"""Summarize quality, coverage and measured usage from a settled RLM checkpoint."""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.checkpoint import load_checkpoint
from evidenceroute.research.rlm.verification import verification_report
from evidenceroute.research.rlm.tools import observation_json

EMPTY_RECEIPT_BYTES = len(observation_json({"ok": True, "result": {"findings": [], "omitted": 0}}))

CONTROLS = {
    "AMD": {2023: {"043757f52188a8c6fb2a2857"}, 2024: {"483314af3b949a29920abbbf"}},
    "AMZN": {2023: {"b79aa2561327a953e141e2aa", "08d421523afa40cc084a4477"},
             2024: {"4d1474550060251037c5f981", "1852186e04d3918d176cac40"}},
    "GOOGL": {2023: {"15787429383b3e9839f69716"}, 2024: {"62bad042e2e9d8334c2e6267"}},
}


def legacy_metrics(node):
    steps = node["decisions"]
    attempts = [attempt for step in steps for attempt in step["attempts"]]
    ops = [op for step in steps for op in step["operations"] if op["tool"] == "submit_findings"]
    counts_present = all("input_tokens" in a and "output_tokens" in a for a in attempts)
    return {
        "provider_attempts": len(attempts),
        "reported_tokens": sum((a["input_tokens"] or 0) + (a["output_tokens"] or 0) for a in attempts)
        if counts_present else None,
        "provider_seconds": sum(a["latency_seconds"] for a in attempts)
        if all("latency_seconds" in a for a in attempts) else None,
        "elapsed_seconds": None,
        "capture_calls": sum(op["outcome"] in {"executed", "rejected"} for op in ops),
        "empty_captures": sum(op.get("observation_bytes") == EMPTY_RECEIPT_BYTES and op["outcome"] == "executed"
                              and not op.get("accepted_finding_refs") and not op.get("finding_validation") for op in ops),
        "rejected_records": sum(len(op.get("finding_validation", ())) for op in ops),
        "comparison_diagnostics": sum(len(step.get("finding_validation", ())) for step in steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--report", type=Path, action="append", default=[], help="Legacy detailed report for invocation metrics.")
    args = parser.parse_args()
    corpus = CorpusEnvironment()
    checkpoint = load_checkpoint(args.checkpoint, corpus)
    report = verification_report(checkpoint)
    legacy = {}
    for path in args.report:
        initial = json.loads(path.read_text()).get("result") or {}
        legacy.update((node["invocation_id"], legacy_metrics(node)) for node in initial.get("diagnostics", ()))
    findings = {item.finding_id: item for item in checkpoint.findings}
    outcomes = {group.ticker: group for group in checkpoint.assignments}
    partitions = {target.pair: target for target in checkpoint.merged_coverage(corpus).targets}
    leaf_metrics = defaultdict(list)
    missing = Counter()
    for run in checkpoint.runs:
        metrics = {item.invocation_id: item.model_dump() for item in run.invocation_metrics}
        for node in run.coverage.invocations:
            if node.invocation.depth == 0:
                continue
            companies = {partitions[pair].ticker for pair in node.invocation.partitions}
            if len(companies) != 1:
                continue
            ticker, = companies
            measured = metrics.get(node.invocation.invocation_id, legacy.get(node.invocation.invocation_id))
            if measured is None:
                missing[ticker] += 1
            else:
                leaf_metrics[ticker].append(measured)

    companies = []
    for group in report["assignment_evidence"]:
        ticker = group["ticker"]
        measured = leaf_metrics[ticker]
        def total(name):
            return (sum(entry[name] for entry in measured) if measured and missing[ticker] == 0
                    and all(entry.get(name) is not None for entry in measured) else None)

        direct = {key for row in partitions.values() if row.ticker == ticker
                  for key in (*row.direct_span_chunk_ids, *row.whole_chunk_ids)}
        retained = {entry.source.chunk_id for finding in findings.values() for entry in finding.evidence
                    if entry.source.ticker == ticker}
        historical = [item for item in checkpoint.comparisons if item.ticker == ticker]
        recovered = [item for item in report["recovered_comparisons"] if item["ticker"] == ticker]
        current = [item for item in report["conclusions"] if item["ticker"] == ticker]
        statuses = Counter(item.status for item in report["items"]
                           if findings[item.finding_ids[0]].evidence[0].source.ticker == ticker)
        controls = {year: {"text_received": bool(keys & direct), "finding_retained": bool(keys & retained)}
                    for year, keys in CONTROLS.get(ticker, {}).items()}
        companies.append({
            **group, "comparisons_historical": len(historical), "comparisons_recovered": len(recovered),
            "verified_relationships": sorted({entry["verified_relationship"] for entry in (*current, *recovered)
                                              if entry["verified_relationship"]}),
            "verification_statuses": statuses,
            "original_citation_chunks": [ref.chunk_id for ref in checkpoint.citations if ref.ticker == ticker],
            "reviewed_citation_chunks": sorted({ref.chunk_id
                                               for ref in (*report["finding_citations"], *report["conclusion_citations"])
                                               if ref.ticker == ticker}),
            "completed_without_findings": group["execution"] == "completed" and not retained,
            "investigator_metrics": {name: total(name) for name in (
                "capture_calls", "empty_captures", "rejected_records", "comparison_diagnostics", "provider_attempts", "reported_tokens",
                "elapsed_seconds", "provider_seconds")},
            "positive_controls": controls,
        })
    print(json.dumps({
        "required_companies": len(outcomes), "required_targets": checkpoint.plan.required_target_count,
        "execution": checkpoint.execution_status, "research_status": checkpoint.research_status,
        "settled_batches": sum(len(run.batches) for run in checkpoint.runs),
        "research_usage": asdict(report["research_usage"]),
        "verification_usage": report["verification_usage"].model_dump(),
        "outstanding_verification": report["outstanding_verification"], "companies": companies,
        "metric_scope": ("Company metrics are investigator totals. Legacy reported tokens use available input/output "
                         "counters. Root and shared verification usage remains in overall totals. Null means unavailable."),
    }, indent=2))


if __name__ == "__main__":
    main()
