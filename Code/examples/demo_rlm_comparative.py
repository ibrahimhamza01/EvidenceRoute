"""Scan the canonical research inventory and compare a bounded company slice."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from demo_rlm_plan import QUESTION, scripted_interpretation
from demo_rlm_recursive import _invocation_report, _usage, _walk
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.comparison import render_comparative_scope
from evidenceroute.research.rlm.discovery import DiscoverySpec, discover_research_plan, scan_research_plan
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.planned_research import execute_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES, ComparativeProfile
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import observation_json, trace_data

TERMS = ("data center", "data centre", "power shortages", "water shortages", "reliable energy",
         "computing infrastructure", "AI infrastructure", "AI accelerators")
DISCOVERY_PHRASES = (
    *TERMS, "artificial intelligence infrastructure", "datacenter", "datacenters", "power supply", "water supply",
    "AI technologies", "supply constraints", "AI services",
)
GUIDANCE = (
    "Analyze the same possible AI infrastructure risk across both fiscal years. Explain how the evidence relates "
    "to AI infrastructure or mark it other or uncertain. Look for a specific risk mechanism, such as a capacity, "
    "power, or supply constraint. General AI regulation and accelerator demand alone do not establish a risk. "
    "Use one concise finding and one exact span per passage. Leave room to read evidence in both years. "
    "Abstain if the available passages do not support a conclusion."
)
ROOT_GUIDANCE = (
    "Ask each company investigator to compare the same risk across both years and look for "
    "prior-year counterevidence. Reassess the strongest delivered spans and report each group's result, "
    "including missing years and abstentions. Use separate relevance and entailment judgments. "
    "Use short delivered finding references so the host can derive original citations. Finish with a partial synthesis "
    "when a company investigation fails. Use the few follow-up calls only for a specific evidence gap."
)


def _offline_factory(corpus, plan, *, compact=False, fail_amd_finish=False):
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
                                   "rationale": "The passage directly identifies the capacity constraint.",
                                   "infrastructure_connection": "The stated shortages could limit data-center expansion."}
                                  for index in (1, 2)]}

    def factory(invocation):
        if invocation.depth == 0:
            groups = [group for group in plan.groups if group.targets and set(group.targets) <= set(invocation.partitions)]
            actions = [{"type": "tool", "name": "delegate_research", "arguments": {
                "question": f"Compare the same risk for {group.ticker} across the two years.",
                "goal": "Examine supporting evidence and potential counterevidence.",
                "partitions": [asdict(pair) for pair in group.targets],
            }} for group in groups]
            if not any(group.ticker == "AMD" for group in groups):
                return ScriptedModelClient([*actions, {"type": "synthesis", "synthesis": {}}])
            if compact:
                return ScriptedModelClient([*actions, {"type": "synthesis", "synthesis": {"cite_refs": ["d1", "d2"]}}])
            assessments = [{"ref": f"d{i}", "relevance": "infrastructure", "entailment": "supports",
                            "rationale": "Power or water shortages can constrain data-center capacity."} for i in (1, 2)]
            return ScriptedModelClient([*actions, {"type": "synthesis", "synthesis": {"analysis_status": "incomplete",
                "comparisons": [{**comparison, "assessments": assessments}],
            }}])
        selected = {pair.document_id for pair in invocation.partitions}
        if any(chunk.document_id not in selected for chunk in chunks):
            if compact:
                return ScriptedModelClient([
                    {"type": "tool", "name": "submit_findings", "arguments": {"findings": []}},
                    {"type": "synthesis", "synthesis": {}},
                ])
            return ScriptedModelClient([{"type": "finish", "answer": "No scoped assessment in this scripted example."}])
        # Start with FY2024, then investigate the specific prior-year counterpart.
        reads = [{"type": "tool", "name": "get_spans", "arguments": {"chunk_id": chunk.chunk_id, "span_ids": [span]}}
                 for chunk, span in reversed(list(zip(chunks, spans)))]
        if compact:
            submit = {"type": "tool", "name": "submit_findings", "arguments": {"findings": findings}}
            if fail_amd_finish:
                bad = {"type": "synthesis", "synthesis": {"cite_refs": ["s8"]}}
                return ScriptedModelClient([*reads, reads[0], submit, bad, bad])
            assessments = [{"ref": f"s{i}", "risk_category": "general_infrastructure", "entailment": "supports"}
                           for i in (1, 2)]
            return ScriptedModelClient([*reads, reads[0], submit, {"type": "synthesis", "synthesis": {
                "comparisons": [{"risk": comparison["risk"], "relationship": "same", "assessments": assessments}],
            }}])
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
                                 "error": op.error, "delivery_error": op.delivery_error,
                                 "observation_acknowledged": op.observation_sent}
                                for op in step.operations if op.error or op.delivery_error or not op.observation_sent]
            if step.error or provider_errors or operation_errors:
                issues.append({
                    "invocation_id": node.invocation.invocation_id, "iteration": step.iteration,
                    "stop_reason": node.stop_reason, "budget_stop": node.budget_stop, "error": step.error,
                    "provider_errors": provider_errors, "operation_errors": operation_errors,
                })
    return issues


def _scope_answer(result, comparisons, answer):
    return render_comparative_scope(
        tuple(comparisons), answer, requested=tuple(group.ticker for group in result.groups),
        admitted={group.ticker for group in result.groups if group.admitted},
        incomplete=result.answer_status == "incomplete",
    )


def _report(result, scan, live, detailed=False):
    trace = result.trace
    invocations = list(_walk(trace)) if trace else []
    finished = bool(trace and trace.stop_reason == "finished")
    admitted = [group for group in result.groups if group.admitted]
    delegated_ids = {fid for group in admitted for fid in group.findings_acknowledged_by_root}
    recursive_success = (
        finished and bool(admitted) and all(group.result_acknowledged for group in admitted)
        and (not delegated_ids or bool(delegated_ids.intersection(trace.used_finding_ids)))
    )
    root_comparisons = trace.comparisons if trace else ()
    assessed = {chunk_id for node in invocations for comparison in node.comparisons
                for chunk_id in comparison.assessed_chunk_ids}
    acknowledged = result.acknowledged_comparisons
    partial = result.partial_comparative_result
    if partial:
        partial = replace(partial, answer=_scope_answer(result, partial.child_comparisons, partial.answer))
    access = {target.pair: target for target in result.coverage.targets}
    rows = []
    for row in scan.sections:
        target = access[row.pair]
        delivered = set(target.whole_chunk_ids) | set(target.direct_span_chunk_ids) | set(target.delegated_excerpt_chunk_ids)
        rows.append({"ticker": row.ticker, "fiscal_year": row.fiscal_year, "target": asdict(row.pair),
                     "inventory_chunks": row.inventory_chunks, "scanned_chunks": row.scanned_chunks,
                     "candidate_chunks": len(row.candidates),
                     "matched_chunks": row.matched_chunks,
                     "candidate_spans": sum(len(hit.matching_span_ids) for hit in row.candidates),
                     "delivered_chunks": len(delivered), "whole_chunks_delivered": len(target.whole_chunk_ids),
                     "direct_span_chunks_delivered": len(target.direct_span_chunk_ids),
                     "assessed_chunks": len(delivered & assessed)})
    return {
        "mode": "live Bedrock comparison" if live else "scripted offline comparison",
        "intent_interpretation": "scripted", "question": result.plan.intent.question,
        "required_targets": result.plan.required_target_count, "required_companies": len(result.plan.requested_companies),
        "execution_status": "success" if finished else "incomplete",
        "recursive_comparative_status": "success" if recursive_success else "incomplete",
        "semantic_review_status": "model_assessed" if root_comparisons or acknowledged else "unassessed",
        "research_status": result.answer_status, "independent_semantic_verification": False,
        "scan": {"terms": scan.terms, "pages": scan.pages_read, "characters": scan.characters_scanned,
                 "metadata_reads": scan.metadata_reads, "unvisited_targets": scan.unvisited_targets,
                 "discovery": scan.discovery, "elapsed_seconds": scan.elapsed_seconds, "stop_reason": scan.stop_reason,
                 "by_company_year": rows},
        "groups": result.groups, "coverage": result.coverage,
        "group_counts": {"admitted": len(admitted),
                         "attempted": sum(group.initial_invocation_id is not None for group in admitted),
                         **{status: sum(group.assignment_status == status for group in result.groups)
                            for status in ("unassigned", "completed", "failed", "unresolved", "not_admitted")}},
        "acknowledged_comparisons": acknowledged,
        "root_comparisons": root_comparisons,
        "root_findings": trace.findings if trace else (),
        "root_selected_inherited_findings": trace.selected_inherited_findings if trace else (),
        "assessed_chunk_ids": sorted(assessed), "original_citations": result.citations,
        "root_cited_years_by_company": {group.ticker: sorted({ref.fiscal_year for ref in result.citations
                                                             if ref.ticker == group.ticker}) for group in admitted},
        "answer": _scope_answer(result, root_comparisons, trace.answer) if finished else None,
        "partial_result": partial,
        "root_stop_reason": trace.stop_reason if trace else None,
        "root_budget_stop": trace.budget_stop if trace else None,
        "initial_assignment_stop": trace.initial_assignment_stop if trace else None,
        "shared_limits": trace.shared_limits if trace else None,
        "shared_usage": trace.shared_budget if trace else None, "provider_usage": _usage(invocations),
        "diagnostics": _diagnostics(invocations),
        **({"trace": _invocation_report(trace, True)} if detailed and trace else {}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly authorize bounded Bedrock inference.")
    parser.add_argument("--trace", action="store_true", help="Include sanitized provider actions and per-step diagnostics.")
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument("--profile", choices=("diagnostic", "expanded"), default="diagnostic")
    profiles.add_argument("--profile-file", type=Path, help="JSON comparative budget profile.")
    parser.add_argument("--term", action="append", help="Replace the example phrases. Repeat for variants.")
    parser.add_argument("--fixed-terms", action="store_true", help="Reproduce the literal-term baseline.")
    parser.add_argument("--max-scan-chunks", type=int, default=30_000)
    parser.add_argument("--max-scan-pages", type=int)
    parser.add_argument("--scan-deadline-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    try:
        profile = (ComparativeProfile.model_validate_json(args.profile_file.read_bytes()) if args.profile_file
                   else COMPARATIVE_PROFILES[args.profile])
        corpus = CorpusEnvironment()
        plan = resolve_research_plan(interpret_intent(QUESTION, scripted_interpretation), corpus)
        options = dict(max_chunks=args.max_scan_chunks, max_pages=args.max_scan_pages,
                       deadline_seconds=args.scan_deadline_seconds)
        if args.fixed_terms:
            scan = scan_research_plan(corpus, plan, tuple(args.term or TERMS), **options)
        else:
            spec = DiscoverySpec.from_intent(plan.intent, tuple(args.term or DISCOVERY_PHRASES))
            scan = discover_research_plan(corpus, plan, spec, **options)
        factory = (lambda _: BedrockRootClient.from_env(disable_retries=True)) if args.live else _offline_factory(corpus, plan)
        runner = RecursiveResearch(
            corpus, factory, comparison_scan=scan, child_guidance=GUIDANCE,
            max_root_reinvestigation_calls=profile.max_root_reinvestigation_calls,
            shared_limits=profile.shared_limits, local_limits=profile.local_limits, finalization=profile.finalization,
        )
        result = execute_research_plan(plan, runner, max_groups=profile.max_groups, root_guidance=ROOT_GUIDANCE)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = _report(result, scan, args.live, args.trace)
    report["profile"] = args.profile if args.profile_file is None else "custom"
    report["configured_limits"] = profile
    print(json.dumps(trace_data(json.loads(observation_json(report))), indent=2))
    return 0 if report["recursive_comparative_status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
