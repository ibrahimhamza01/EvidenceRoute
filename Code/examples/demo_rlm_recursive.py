"""Scripted offline FY2023/FY2024 delegation using the local canonical corpus."""

import json
from dataclasses import asdict

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.recursive import RecursiveResearch


def _delegate(question, *targets):
    return {
        "type": "tool", "name": "delegate_research",
        "arguments": {"question": question, "goal": "Return one cited excerpt with scoped coverage.",
                      "partitions": [asdict(target) for target in targets]},
    }


def _excerpt(chunk, *, used=()):
    actions = []
    if not used:
        actions.append({"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": chunk.chunk_id}})
    actions.append({
        "type": "finish", "answer": "Collected one excerpt. Filing coverage is partial.",
        "cited_chunk_ids": [chunk.chunk_id], "used_finding_ids": list(used), "analysis_status": "incomplete",
        "findings": [{
            "claim": "Excerpt from the opening Item 1A chunk.",
            "evidence": [{"chunk_id": chunk.chunk_id, "quote": chunk.text[:120].strip()}],
            "analysis_status": "incomplete", "uncertainty": "The example does not analyze the entire section.",
        }],
    })
    return actions


def _summarize(trace):
    return {
        "invocation": asdict(trace.invocation),
        "stop_reason": trace.stop_reason,
        "analysis_status": trace.analysis_status,
        "coverage_status": trace.coverage_status,
        "coverage": [asdict(partition) for partition in trace.partition_coverage],
        "findings": [asdict(finding) for finding in trace.findings],
        "used_finding_ids": trace.used_finding_ids,
        "cited_chunk_ids": trace.cited_chunk_ids,
        "delegations": [{
            "status": step.delegation.status,
            "delivered_finding_ids": step.delegation.delivered_finding_ids,
            "child": _summarize(step.delegation.child),
        } for step in trace.steps if step.delegation and step.delegation.child],
    }


def main() -> int:
    try:
        corpus = CorpusEnvironment()
    except FileNotFoundError:
        print("This offline demo requires locally available canonical manifest and Parquet files.")
        return 2

    filings = {year: corpus.list_filings(ticker="NVDA", fiscal_year=year).items[0] for year in (2023, 2024)}
    old, new = (SectionTarget(filings[year].document_id, "item_1a") for year in (2023, 2024))
    business = SectionTarget(new.document_id, "item_1")
    earlier, later = (corpus.read_chunks(pair.document_id, pair.section_id, limit=1).items[0]
                      for pair in (old, new))
    scripts = {
        "root": [
            _delegate("Collect an FY2023 risk excerpt.", old),
            _delegate("Collect FY2024 evidence from the assigned partitions.", new, business),
            {
                "type": "finish", "answer": "Collected one Item 1A excerpt for each year. Coverage is partial.",
                "cited_chunk_ids": [earlier.chunk_id, later.chunk_id],
                "used_finding_ids": ["child-1:finding-1", "child-2:finding-1"], "analysis_status": "incomplete",
            },
        ],
        "child-1": _excerpt(earlier),
        "child-2": [_delegate("Collect one FY2024 Item 1A excerpt.", new),
                    *_excerpt(later, used=("child-3:finding-1",))],
        "child-3": _excerpt(later),
    }
    runner = RecursiveResearch(corpus, lambda invocation: ScriptedModelClient(scripts[invocation.invocation_id]))
    trace = runner.run("Collect one NVIDIA risk excerpt for FY2023 and FY2024.",
                       partitions=(old, new, business), run_id="nvda-scripted")
    print(json.dumps({
        "mode": "scripted offline", "answer": trace.answer,
        "shared_usage": asdict(trace.shared_budget), "trace": _summarize(trace),
    }, indent=2))
    return 0 if trace.stop_reason == "finished" and len(trace.cited_chunk_ids) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
