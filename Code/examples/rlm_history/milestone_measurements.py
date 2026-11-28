"""Offline comparison of discovery and representative serialized evidence delivery."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from dataclasses import asdict

from demo_rlm_discovery import KNOWN_TARGETS
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.controller import finish_input_schema
from evidenceroute.research.rlm.discovery import DiscoverySpec, discover_research_plan
from evidenceroute.research.rlm.planning import ResearchIntent, resolve_research_plan
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import CorpusToolDispatcher, observation_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Saved acceptance-corpus live report.")
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    result = data["result"]["research"]
    assert result["run_id"] == "ce45e24c55744535b81e80a0756e3428"
    corpus = CorpusEnvironment()
    plan = resolve_research_plan(ResearchIntent.model_validate_json(json.dumps(result["plan"]["intent"])), corpus)
    spec = DiscoverySpec.model_validate_json(json.dumps(result["scan"]["discovery"]))
    distribution = Counter()
    repeated = Counter()
    attempts = data["result"]["interpretation"]["attempts"].copy()
    request_sizes = []
    input_growth = {}
    for node in result["diagnostics"]:
        seen = set()
        sizes = []
        for step in node["decisions"]:
            attempts.extend(step["attempts"])
            sizes.extend(attempt["request_bytes"] for attempt in step["attempts"] if attempt["request_bytes"] is not None)
            for op in step["operations"]:
                distribution[op["tool"]] += 1
                if op["acknowledged"] and op["tool"] in {"get_spans", "get_chunk", "read_chunks", "search_chunks"}:
                    for chunk in op["chunk_ids"]:
                        repeated["deliveries"] += 1
                        repeated["repeated_chunk_id"] += chunk in seen
                        seen.add(chunk)
        request_sizes.extend(sizes)
        input_growth[node["invocation_id"].split(".", 1)[1]] = [sizes[0], max(sizes), sizes[-1]] if sizes else []
    baseline = {
        "scan": {"chunks": sum(row["scanned_chunks"] for row in result["scan"]["sections"]),
                 "characters": result["scan"]["characters_scanned"], "seconds": result["scan"]["elapsed_seconds"],
                 "candidates_by_company": dict(Counter(row["ticker"] for row in result["scan"]["sections"]
                                                      for _ in row["candidates"]))},
        "scope": {"required": len(result["plan"]["targets"]) if "targets" in result["plan"] else len(result["coverage"]["targets"]),
                  "resolved": len(result["scan"]["sections"]), "companies": len(result["groups"])},
        "assignments": dict(Counter(g["outcome"]["assignment_status"] if g["outcome"] else g["pending_reason"]
                                    for g in result["groups"])),
        "tokens": {field: sum(a.get(field) or 0 for a in attempts) for field in ["input_tokens", "output_tokens"]},
        "operation_counts": distribution, "repeat_delivery_counts": repeated,
        "request_bytes_research": {"sum": sum(request_sizes), "median": median(request_sizes),
                                   "max": max(request_sizes), "per_invocation_first_peak_final": input_growth},
        "acknowledged_findings": len(result["acknowledged_child_findings"]),
        "selected_findings": len(result["root_selected_inherited_findings"]), "citations": len(result["citations"]),
        "usage": data["result"]["usage"], "states": [result["execution_status"], result["research_status"]],
        "verification": data["independent_semantic_verification"], "errors": result["errors"],
        "validation": [step["finding_validation"] for n in result["diagnostics"] for step in n["decisions"]
                       if "finding_validation" in step],
    }
    scans = {}
    for label, expanded in [("original", False), ("expanded", True)]:
        runs = [discover_research_plan(corpus, plan, spec.model_copy(update={"expand_context": expanded})) for _ in range(5)]
        scan = runs[-1]
        ids = {hit.chunk_id for row in scan.sections for hit in row.candidates}
        counts = Counter(row.ticker for row in scan.sections for _ in row.candidates)
        scans[label] = {
            "scanned_chunks": sum(row.scanned_chunks for row in scan.sections), "characters": scan.characters_scanned,
            "median_scan_seconds": median(run.elapsed_seconds for run in runs), "matches": sum(row.matched_chunks for row in scan.sections),
            "candidates": len(ids), "known_found": sorted(ids & KNOWN_TARGETS), "known_missing": sorted(KNOWN_TARGETS - ids),
            "by_company": {ticker: counts[ticker] for ticker in plan.requested_companies},
            "matching_sections": sum(bool(row.candidates) for row in scan.sections),
        }

    dispatcher = CorpusToolDispatcher(corpus, max_response_bytes=16000, include_spans=True)
    chunk_id = "043757f52188a8c6fb2a2857"
    chunk = corpus.get_chunk(chunk_id)
    span, (start, end) = next((key, offsets) for key, offsets in passage_spans(chunk.text).items()
                             if "Data centers depend on access to clean water and reliable energy" in chunk.text[slice(*offsets)])
    args = {"chunk_id": chunk_id, "span_ids": [span]}
    size = lambda item: len(item.content.encode("utf-8"))
    full = size(dispatcher.dispatch("get_chunk", {"chunk_id": chunk_id}))
    fresh = size(dispatcher.dispatch("get_spans", args))
    repeat = size(dispatcher.dispatch("get_spans", args, delivered_spans={
        (chunk.document_id, chunk.section_id, chunk.text[start:end]): (chunk_id, span),
    }))
    print(json.dumps({"baseline_live": baseline, "offline_discovery": scans, "offline_evidence_bytes": {
        "whole_chunk": full, "one_exact_span": fresh, "repeated_span_reference": repeat,
        "repeat_savings_over_three_replays_bytes": 3 * (fresh - repeat),
        "finish_schema_bytes": {
            "legacy_child": len(observation_json(finish_input_schema(recursive=True, comparative=True)).encode("utf-8")),
            "legacy_root_reference": len(observation_json(finish_input_schema(
                recursive=True, comparative=True, reference_based=True)).encode("utf-8")),
            "capture_by_reference": len(observation_json(finish_input_schema(
                recursive=True, comparative=True, reference_based=True, compact=True)).encode("utf-8")),
        },
    }}, indent=2))


if __name__ == "__main__":
    main()
