"""Audit live canonical deliveries and exercise checkpoint recovery with a scripted verifier."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig
from evidenceroute.research.rlm.checkpoint import load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import observation_json
from evidenceroute.research.rlm.verification import BedrockSemanticVerifier, recover_checkpoint, verification_report
from evidenceroute.research.rlm.verification_records import VerificationLimits

class ReplayedRelationship:
    def send(self, payload, *, timeout_seconds):
        cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
        reviews = []
        for case in cases:
            evidence = case["selected_evidence"]
            assert case["kind"] == "recovery" and {e["year"] for e in evidence} == {2023, 2024}
            if case["company"] == "AMD":
                assert all("water shortages" in e["text"] for e in evidence)
                statement = "Potential power or water shortages could limit data-center expansion."
            else:
                assert all("may depend on the availability and pricing of third-party equipment" in e["text"]
                           for e in evidence)
                statement = "Deployment of certain AI technologies may depend on third-party equipment availability and pricing."
            reviews.append({"ref": case["ref"], "components": [
                {"statement": statement, "status": "supported", "evidence": [e["ref"] for e in evidence]},
            ], "comparison": {"relationship": "more_specific_wording", "relationship_evidence": [e["ref"] for e in evidence],
                              "earlier_year": "supports_risk", "earlier_evidence": [e["ref"] for e in evidence if e["year"] == 2023]}})
        return {"stopReason": "tool_use", "usage": {"inputTokens": 650, "outputTokens": 220},
                "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                    "toolUseId": "offline-review", "name": "submit_verification", "input": {"reviews": reviews}
                }}]}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory of external saved reports and checkpoints.")
    artifacts = parser.parse_args().artifacts
    corpus = CorpusEnvironment()
    audit = {}
    for label, ticker in (("amd", "AMD"), ("amzn-googl", "GOOGL")):
        base = f"rlm-{label}-quality-live"
        original = load_checkpoint(artifacts / f"{base}-checkpoint.json", corpus)
        report = json.loads((artifacts / f"{base}.json").read_text())
        assert original.runs[0].run_id == report["result"]["run_id"]
        excerpts = []
        for node in original.runs[0].coverage.invocations:
            for key, span in node.direct_span_ids:
                chunk = corpus.get_chunk(key)
                a, b = passage_spans(chunk.text)[span]
                spans = list(passage_spans(chunk.text))
                following = spans[spans.index(span) + 1:spans.index(span) + 2]
                excerpts.append(dict(ticker=chunk.ticker, year=chunk.fiscal_year, chunk_id=key, span_id=span,
                                     quote=chunk.text[a:b], chunk_context=chunk.text, next_span=following, source=chunk.source_url))
        verifier = BedrockSemanticVerifier(ReplayedRelationship(), config=BedrockConfig(model="offline-fixture"))
        checked = recover_checkpoint(original, corpus, verifier, limits=VerificationLimits(max_items=2, max_provider_attempts=1),
                                     companies=(ticker,))
        with TemporaryDirectory(prefix="rlm-recovery-") as temporary:
            saved = Path(temporary) / "checkpoint.json"
            save_checkpoint(saved, checked)
            checked = load_checkpoint(saved, corpus)
        final_report = verification_report(checked)
        pair, = final_report["recovered_comparisons"]
        assert pair["verified_relationship"] == "more_specific_wording"
        assert checked.runs == original.runs
        assert checked.verification_runs[:-1] == original.verification_runs
        if ticker == "GOOGL":
            acknowledged = {fid for node in original.runs[0].coverage.invocations
                            for fid in node.findings_acknowledged_by_parent}
            assert pair["finding_ids"][0] not in acknowledged and pair["finding_ids"][1] in acknowledged
        audit[label] = {"acknowledged_excerpts": excerpts, "live_research_usage": report["result"]["usage"],
                        "live_verification_usage": original.verification_usage, "observed_trace": report["result"]["diagnostics"],
                        "offline_checkpoint_only_recovery": final_report}
    print(json.dumps(json.loads(observation_json(audit)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
