"""Recover canonical excerpts and replay the saved AMD tool sequence offline."""
import argparse
import json
from inspect import getfile
from dataclasses import asdict
from pathlib import Path
from subprocess import run as run_command

from demo_rlm_comparative import GUIDANCE, ROOT_GUIDANCE
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.checkpoint import load_checkpoint
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import observation_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory of external saved reports and checkpoints.")
    artifacts = parser.parse_args().artifacts
    corpus = CorpusEnvironment()
    checkpoint = load_checkpoint(artifacts / "rlm-amd-live-checkpoint.json", corpus)
    report = json.loads((artifacts / "rlm-amd-live.json").read_text())["result"]
    run = checkpoint.runs[0]
    node = next(node for node in run.coverage.invocations if node.invocation.depth)
    assert run.run_id == report["run_id"]
    delivered = dict.fromkeys(node.direct_span_ids)
    evidence = {}
    for row in report["scan"]["sections"]:
        for rank, hit in enumerate(row["candidates"], 1):
            chunk_id = hit["chunk_id"]
            chunk = corpus.get_chunk(chunk_id)
            spans = passage_spans(chunk.text)
            evidence.setdefault(chunk.fiscal_year, []).append({
                "rank": rank, "chunk_id": chunk_id, "sec_url": chunk.source_url,
                "hints": {key: chunk.text[slice(*spans[key])] for key in hit["matching_span_ids"]},
                "delivered": {key: chunk.text[slice(*spans[key])] for cid, key in delivered if cid == chunk_id},
                "chunk_context": chunk.text,
            })

    live_steps = report["diagnostics"][1]["decisions"]
    reads = [(op["tool_use_id"], "get_spans", {"chunk_id": op["chunk_ids"][0],
              "span_ids": [sid for cid, sid in delivered if cid == op["chunk_ids"][0]]})
             for op in live_steps[0]["operations"]]
    # Search arguments are absent from the report. These stand-ins exercise history
    # shape and native IDs and deliberately do not claim to replay the exact bytes.
    searches = [(op["tool_use_id"], "search_chunks", {
        "query": "data center power water shortages",
        "filters": {"fiscal_years": [year]},
    }) for op, year in zip(live_steps[1]["operations"], (2024, 2023))]
    child_turns = [
        reads, searches,
        [(live_steps[2]["operations"][0]["tool_use_id"], "submit_findings", {"findings": []})],
        [("offline-child-finish", "finish_research", {"synthesis": {}})],
    ]
    root_turns = [[("offline-root-assign", "delegate_research", {
        "question": node.invocation.question, "goal": node.invocation.goal,
        "partitions": [asdict(pair) for pair in node.invocation.partitions],
    })], [("offline-root-finish", "finish_research", {"synthesis": {}})]]

    class Transport:
        def __init__(self, turns):
            self.turns = iter(turns)
            self.requests = []

        def send(self, payload, *, timeout_seconds):
            self.requests.append(json.loads(observation_json(payload)))
            return {"stopReason": "tool_use", "usage": {"inputTokens": 400, "outputTokens": 150},
                    "output": {"message": {"role": "assistant", "content": [
                        {"toolUse": {"toolUseId": key, "name": name, "input": args}}
                        for key, name, args in next(self.turns)
                    ]}}}

    child = Transport(child_turns)
    root = Transport(root_turns)
    profile = COMPARATIVE_PROFILES["full-corpus"]
    runner = RecursiveResearch(
        corpus, lambda invocation: BedrockRootClient(child if invocation.depth else root, config=BedrockConfig(
            model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", max_retries=0)),
        shared_limits=profile.shared_limits, local_limits=profile.local_limits, finalization=profile.finalization,
        comparative_capture=True, child_guidance=GUIDANCE, max_root_reinvestigation_calls=0,
    )
    result = execute_corpus_research(checkpoint.plan, runner, limits=CorpusLimits.model_validate(report["limits"]),
                                     discovery=checkpoint.discovery, root_guidance=ROOT_GUIDANCE, detailed_trace=True)
    assert result.execution_status == "finished"
    before, capture = child.requests[1:3]
    assert before["messages"] == capture["messages"][:len(before["messages"])]
    def size(value):
        return len(observation_json(value).encode("utf-8"))
    print(json.dumps({
        "baseline_commit": "71c6733", "run_id": run.run_id, "baseline_usage": report["usage"],
        "replay_matches_baseline_source": run_command(["git", "-C", str(Path(getfile(CorpusEnvironment)).resolve().parents[5]),
                                                      "diff", "--quiet", "71c6733", "--", "Code/src"]).returncode == 0,
        "execution_status": checkpoint.execution_status, "research_status": checkpoint.research_status,
        "canonical_evidence_and_ranked_candidates": evidence,
        "live_request_bytes": [turn["attempts"][0]["request_bytes"] for turn in live_steps],
        "live_input_tokens": [turn["attempts"][0]["input_tokens"] for turn in live_steps],
        "reproduction": {
            "before_and_capture": [{key: size(payload[key]) for key in ("messages", "system", "toolConfig")}
                                   | {"request": size(payload)} for payload in (before, capture)],
            "capture_keeps_previous_messages_as_exact_prefix": True,
            "capture_choice": capture["toolConfig"]["toolChoice"],
            "capture_tool_spec": capture["toolConfig"]["tools"],
            "capture_system": capture["system"],
            "reconstructed_search_arguments": True,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
