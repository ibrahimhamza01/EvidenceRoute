"""Scan the canonical research inventory and compare a bounded company slice."""

import argparse
import json
from dataclasses import asdict

from demo_rlm_plan import QUESTION, scripted_interpretation
from demo_rlm_recursive import _invocation_report, _usage, _walk
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.comparison import scan_research_plan
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.planned_research import execute_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import observation_json, trace_data

TERMS = ("data center", "data centre", "power shortages", "water shortages", "reliable energy",
         "computing infrastructure", "AI infrastructure", "AI accelerators")
GUIDANCE = (
    "Analyze the same possible AI infrastructure risk across both fiscal years. Explain how the evidence relates "
    "to AI infrastructure or mark it other or uncertain. General AI safety or regulation does not establish "
    "infrastructure relevance. Evaluate whether each passage supports or contradicts the specific risk. "
    "Try a focused earlier-year lookup for counterevidence when needed. Use one finding and one exact span "
    "per passage. Do not assert novelty from incomplete evidence."
)


def _offline_factory(corpus, plan):
    # Fixed regression inputs for the scripted demo only. Live research gets the lexical shortlist.
    chunks = tuple(corpus.get_chunk(key) for key in ("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf"))
    spans = [next(span for span, (start, end) in passage_spans(chunk.text).items()
                  if "Data centers depend on access to clean water and reliable energy" in chunk.text[start:end])
             for chunk in chunks]
    findings = [{"claim": "Power or water shortages can constrain customer data-center expansion.",
                 "uncertainty": "This passage discusses data centers generally and does not establish an AI-specific link.",
                 "analysis_status": "analyzed", "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span}]}
                for chunk, span in zip(chunks, spans)]
    comparison = {"risk": "Power and water constraints on data-center expansion", "same_risk": "yes",
                  "assessments": [{"finding_index": index, "relevance": "infrastructure", "entailment": "supports",
                                   "rationale": "The passage directly identifies the capacity constraint."}
                                  for index in (1, 2)]}

    def factory(invocation):
        if invocation.depth == 0:
            groups = [group for group in plan.groups if group.targets and set(group.targets) <= set(invocation.partitions)]
            actions = [{"type": "tool", "name": "delegate_research", "arguments": {
                "question": f"Compare the same risk for {group.ticker} across the two years.",
                "goal": "Examine supporting evidence and potential counterevidence.",
                "partitions": [asdict(pair) for pair in group.targets],
            }} for group in groups]
            amd_index = next(index for index, group in enumerate(groups, 1) if group.ticker == "AMD")
            return ScriptedModelClient([*actions, {"type": "finish", "answer": "Sampled comparative research.",
                "analysis_status": "incomplete", "cited_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "used_finding_ids": [f"child-{amd_index}:finding-{number}" for number in (1, 2)],
                "findings": findings, "comparisons": [comparison],
            }])
        selected = {pair.document_id for pair in invocation.partitions}
        if any(chunk.document_id not in selected for chunk in chunks):
            return ScriptedModelClient([{"type": "finish", "answer": "No scoped assessment in this scripted example."}])
        # Start with FY2024, then investigate the specific prior-year counterpart.
        reads = [{"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": chunk.chunk_id}}
                 for chunk in reversed(chunks)]
        return ScriptedModelClient([*reads, {"type": "finish", "answer": "The underlying risk appears in both years.",
                                            "findings": findings, "comparisons": [comparison],
                                            "analysis_status": "incomplete"}])

    return factory


def _diagnostics(invocations):
    issues = []
    for node in invocations:
        for step in node.steps:
            provider_errors = [{"code": attempt.error, "response_received": attempt.response_received,
                                "provider_stop_reason": attempt.provider_stop_reason}
                               for attempt in step.model_attempts if attempt.error]
            operation_errors = [{"tool": op.tool_name, "tool_use_id": op.tool_use_id,
                                 "error": op.error, "delivery_error": op.delivery_error}
                                for op in step.batch_operations if op.error or op.delivery_error]
            if step.error or provider_errors or operation_errors:
                issues.append({
                    "invocation_id": node.invocation.invocation_id, "iteration": step.iteration,
                    "stop_reason": node.stop_reason, "budget_stop": node.budget_stop, "error": step.error,
                    "provider_errors": provider_errors, "batch_errors": operation_errors,
                })
    return issues


def _report(result, scan, live, detailed=False):
    trace = result.trace
    invocations = list(_walk(trace)) if trace else []
    assessed = {chunk_id for node in invocations for comparison in node.comparisons
                for chunk_id in comparison.assessed_chunk_ids}
    acknowledged = []
    if trace:
        for step in trace.steps:
            for op in step.operations:
                edge = op.delegation
                if edge and edge.child and op.observation_sent and not op.delivery_error:
                    acknowledged.extend(c for c in edge.child.comparisons
                                        if set(c.finding_ids) <= set(edge.delivered_finding_ids))
    access = {target.pair: target for target in result.coverage.targets}
    rows = [{"ticker": row.ticker, "fiscal_year": row.fiscal_year, "target": asdict(row.pair),
             "inventory_chunks": row.inventory_chunks, "scanned_chunks": row.scanned_chunks,
             "candidate_chunks": len(row.candidates),
             "candidate_spans": sum(len(hit.matching_span_ids) for hit in row.candidates),
             "delivered_chunks": len(access[row.pair].whole_chunk_ids),
             "assessed_chunks": len(set(access[row.pair].whole_chunk_ids) & assessed)} for row in scan.sections]
    return {
        "mode": "live Bedrock comparison" if live else "scripted offline comparison",
        "intent_interpretation": "scripted", "question": result.plan.intent.question,
        "required_targets": result.plan.required_target_count, "required_companies": len(result.plan.requested_companies),
        "research_status": "incomplete", "independent_semantic_verification": False,
        "scan": {"terms": scan.terms, "pages": scan.pages_read, "characters": scan.characters_scanned,
                 "elapsed_seconds": scan.elapsed_seconds, "stop_reason": scan.stop_reason, "by_company_year": rows},
        "groups": result.groups, "coverage": result.coverage,
        "acknowledged_comparisons": acknowledged,
        "root_comparisons": trace.comparisons if trace else (),
        "assessed_chunk_ids": sorted(assessed), "original_citations": result.citations,
        "root_stop_reason": trace.stop_reason if trace else None,
        "root_budget_stop": trace.budget_stop if trace else None,
        "shared_limits": trace.shared_limits if trace else None,
        "shared_usage": trace.shared_budget if trace else None, "provider_usage": _usage(invocations),
        "diagnostics": _diagnostics(invocations),
        **({"trace": _invocation_report(trace, True)} if detailed and trace else {}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize bounded Bedrock inference.")
    parser.add_argument("--trace", action="store_true", help="Include sanitized provider actions and per-step diagnostics.")
    parser.add_argument("--term", action="append", help="Replace default scan terms. Repeat to add terms.")
    parser.add_argument("--max-scan-chunks", type=int, default=30_000)
    parser.add_argument("--scan-deadline-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        corpus = CorpusEnvironment()
        plan = resolve_research_plan(interpret_intent(QUESTION, scripted_interpretation), corpus)
        scan = scan_research_plan(corpus, plan, tuple(args.term or TERMS), max_chunks=args.max_scan_chunks,
                                  deadline_seconds=args.scan_deadline_seconds)
        factory = (lambda _: BedrockRootClient.from_env(disable_retries=True)) if args.live else _offline_factory(corpus, plan)
        runner = RecursiveResearch(
            corpus, factory, comparison_scan=scan, child_guidance=GUIDANCE,
            shared_limits=SharedLimits(max_depth=1, max_delegations=2, max_provider_attempts=18,
                                       max_partitions=4, max_corpus_attempts=12,
                                       max_reported_tokens=100_000, deadline_seconds=180.0),
            local_limits=RootLimits(max_iterations=6, max_tool_calls=6, max_response_bytes=6_000, deadline_seconds=180.0),
            finalization=FinalizationPolicy(child_provider_attempts=6, finalization_attempts=2,
                                           finalization_tool_choice="required"),
        )
        result = execute_research_plan(plan, runner, root_guidance=(
            "Delegate both whole company groups. Ask for a structured comparison including possible prior-year "
            "counterevidence. Reassess the strongest supporting and counterevidence passages using only delivered spans, "
            "then submit your own structured comparison with separate relevance and entailment judgments. "
            "Declare the delivered finding IDs you use and cite the original chunks. Abstain where evidence is insufficient."
        ))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    report = _report(result, scan, args.live, args.trace)
    print(json.dumps(trace_data(json.loads(observation_json(report))), indent=2))
    return 0 if report["acknowledged_comparisons"] and report["root_comparisons"] and report["root_stop_reason"] == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
