"""Read-only comparisons of saved research and verification artifacts."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.checkpoint import load_checkpoint, plan_continuation
from evidenceroute.research.rlm.verification import verification_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory of external saved reports and checkpoints.")
    artifacts = parser.parse_args().artifacts
    corpus = CorpusEnvironment()
    def checkpoint(prefix):
        return load_checkpoint(artifacts / (prefix + "-checkpoint.json"), corpus)
    def saved(name):
        return json.loads((artifacts / (name + ".json")).read_text())
    prefix = "rlm-final-"
    initial, continued, first, second, final = [checkpoint(prefix + name) for name in
        ("corpus-live", "corpus-continued", "corpus-reviewed", "review-pass2", "review-pass3")]
    assert initial.runs == continued.runs[:1]
    assert all(x in continued.findings for x in initial.findings)
    assert initial.discovery_cache is not None
    assert initial.discovery_cache == continued.discovery_cache
    assert initial.plan == continued.plan
    assert {g.ticker for g in plan_continuation(initial).groups} == set(continued.runs[1].target_companies)
    print("initial/continued findings", len(initial.findings), len(continued.findings))
    prior = continued
    for current, filename in zip((first, second, final), ("corpus-reviewed-live", "review-pass2", "review-pass3")):
        assert continued.model_dump(exclude={"verification_runs"}) == current.model_dump(exclude={"verification_runs"})
        assert prior.verification_runs == current.verification_runs[:-1]
        old = verification_report(prior)
        allowed = {i.target_id for i in old["items"] if i.status == "unverified" and
                   (i.kind == "finding" or i.target_id in {c["target_id"] for c in old["conclusions"]})}
        run = current.verification_runs[-1]
        assert {i.target_id for i in run.items} <= allowed
        report = verification_report(current)
        settled = {i.target_id:i for i in old["items"] if i.status != "unverified"}
        assert all(i == settled[i.target_id] for i in report["items"] if i.target_id in settled)
        assert report["answer"] == saved(prefix+filename)["reports"][0]["answer"]
        print(filename, len(allowed), Counter(i.status for i in run.items), run.usage)
        prior = current
    print("current comparisons", len(report["conclusions"]), "historical",len(final.comparisons))
    print("status",Counter((i.kind,i.status) for i in report["items"]))
    print("combined",final.combined_usage)
    print("related",[(c["ticker"],c["verified_relationship"]) for c in report["conclusions"] if c["verified_relationship"]])
    for name in ("rlm-amd-recovered-live","rlm-amzn-googl-recovered-live"):
        c=checkpoint(name)
        assert verification_report(c)["answer"] == saved(name)["reports"][0]["answer"]
        print(name, [(i.status,i.comparison.relationship) for i in c.verification_runs[-1].items])

    # Cross-check the answer projection independently of the evaluator's company totals.
    cited_in_answer = set(re.findall(r"\b[0-9a-f]{24}\b", report["answer"]))
    def chunk_ids(refs):
        return {ref.chunk_id for ref in refs}
    finding_sources = chunk_ids(report["finding_citations"])
    conclusion_sources = chunk_ids(report["conclusion_citations"])
    original_sources = chunk_ids(final.citations)
    assert cited_in_answer <= {e.source.chunk_id for e in final.evidence}
    assert cited_in_answer <= finding_sources | conclusion_sources
    print("printed, original, finding, conclusion, reviewed union, printed missing from original",
          len(cited_in_answer), len(original_sources), len(finding_sources), len(conclusion_sources),
          len(finding_sources | conclusion_sources), len(cited_in_answer - original_sources))

    from collections import defaultdict
    years = defaultdict(set)
    for finding in final.findings:
        years[finding.evidence[0].source.ticker].add(finding.evidence[0].source.fiscal_year)
    print("yield", Counter(tuple(sorted(years[t])) for t in final.plan.requested_companies))
    for run in final.runs:
        batches = [b for b in run.batches if b.usage is not None]
        for name in ("provider_attempts", "corpus_attempts", "delegated_invocations",
                     "reported_tokens", "partition_assignments", "finding_submission_attempts"):
            assert sum(getattr(b.usage, name) for b in batches) == getattr(run.usage, name)
        print("batch companies", [b.companies for b in batches])

    for stage, cp in (("initial", initial), ("continued", continued)):
        name = prefix + "corpus-" + stage + "-live"
        assert saved(name)["final_report"]["answer"] == verification_report(cp)["answer"]


if __name__ == "__main__":
    main()
