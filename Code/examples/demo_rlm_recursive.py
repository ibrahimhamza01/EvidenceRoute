"""Run scripted NVIDIA delegation offline, or opt in to a bounded Bedrock comparison."""

import argparse
import json
import re
from dataclasses import asdict
from time import monotonic

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import trace_data

YEARS = (2023, 2024)
LIVE_QUESTION = (
    "Compare one documented supply-chain risk disclosure from NVIDIA's FY2023 and FY2024 filings. "
    "Identify similarities and differences supported by the retrieved evidence, and disclose limitations."
)
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _risk_partitions(corpus):
    partitions = {}
    for year in YEARS:
        filings = corpus.list_filings(ticker="NVDA", fiscal_year=year)
        if filings.total != 1:
            raise ValueError(f"Expected exactly one canonical NVIDIA filing for FY{year}.")
        partition = SectionTarget(filings.items[0].document_id, "item_1a")
        try:
            corpus.get_section(partition.document_id, partition.section_id, limit=1)
        except KeyError:
            raise ValueError(f"Canonical NVIDIA FY{year} Item 1A is unavailable.") from None
        partitions[year] = partition
    return partitions


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
        } for step in _operations(trace) if step.delegation and step.delegation.child],
    }


def _run_scripted(corpus, partitions):
    old, new = (partitions[year] for year in YEARS)
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


def _walk(trace):
    yield trace
    for step in _operations(trace):
        if step.delegation and step.delegation.child:
            yield from _walk(step.delegation.child)


def _operations(trace):
    return (operation for step in trace.steps for operation in step.operations)


def _requested_tool_calls(trace):
    return sum(step.provider_tool_request_count or len(step.operations) for step in trace.steps)


def _usage(traces):
    attempts = [attempt for trace in traces for attempt in trace.model_attempts]
    tokens = {}
    for field in TOKEN_FIELDS:
        counts = [getattr(attempt, field) for attempt in attempts if getattr(attempt, field) is not None]
        tokens[field] = {
            "reported_total": sum(counts) if counts else None,
            "missing_attempts": len(attempts) - len(counts),
        }
    provider_latencies = [a.provider_latency_seconds for a in attempts if a.provider_latency_seconds is not None]
    return {
        "provider_attempts": len(attempts),
        "provider_responses_received": sum(a.response_received for a in attempts),
        "provider_retries": sum(max(0, len(step.model_attempts) - 1) for t in traces for step in t.steps),
        "attempts_with_missing_input_or_output": sum(
            a.input_tokens is None or a.output_tokens is None for a in attempts
        ),
        "tokens": tokens,
        "provider_attempt_wall_seconds_sum": sum(a.latency_seconds for a in attempts),
        "reported_provider_latency_seconds_sum": sum(provider_latencies) if provider_latencies else None,
        "attempts_missing_provider_latency": len(attempts) - len(provider_latencies),
    }


def _acknowledged_findings(step):
    delegation = step.delegation
    if not step.observation_sent or step.delivery_error or delegation is None or delegation.child is None:
        return ()
    delivered = set(delegation.delivered_finding_ids)
    return tuple(finding for finding in delegation.child.findings if finding.finding_id in delivered)


def _evidence_delivery(trace):
    direct = sorted({
        ref.chunk_id for step in _operations(trace)
        if step.observation_sent and not step.delivery_error and step.delegation is None and step.tool_name != "get_spans"
        for ref in step.evidence
    })
    spans = sorted({ref.chunk_id for step in _operations(trace) if step.tool_name == "get_spans"
                    and step.observation_sent and not step.delivery_error for ref in step.evidence})
    excerpts = [
        {"finding_id": finding.finding_id, "declared_used": finding.finding_id in trace.used_finding_ids,
         **asdict(item)}
        for step in _operations(trace) for finding in _acknowledged_findings(step) for item in finding.evidence
    ]
    return {"direct_chunk_text_ids": direct, "direct_span_chunk_ids": spans, "delegated_excerpts": excerpts}


def _invocation_report(trace, detailed):
    report = {
        "invocation": asdict(trace.invocation),
        "stop_reason": trace.stop_reason,
        "budget_stop": trace.budget_stop,
        "model_reported_analysis_status": trace.analysis_status,
        "coverage_status": trace.coverage_status,
        "initial_assignment_stop": trace.initial_assignment_stop,
        "partition_coverage": [asdict(item) for item in trace.partition_coverage],
        "inclusive_wall_seconds": trace.elapsed_seconds,
        "model_decisions": len(trace.steps),
        "finalization_decisions": sum(step.phase == "finalization" for step in trace.steps),
        "finalization_reason": next((step.finalization_reason for step in trace.steps if step.finalization_reason), None),
        "local_tool_attempts": trace.tool_calls,
        "requested_tool_calls": _requested_tool_calls(trace),
        "usage": _usage([trace]),
        "findings_generated": [asdict(finding) for finding in trace.findings],
        "declared_used_finding_ids": trace.used_finding_ids,
        "cited_chunk_ids": trace.cited_chunk_ids,
        "evidence_delivered": _evidence_delivery(trace),
        "step_errors": [{"iteration": step.iteration, "code": step.error["code"]}
                        for step in trace.steps if step.error],
        "provider_errors": [attempt.error for attempt in trace.model_attempts if attempt.error],
        "delegations": [],
    }
    for step in _operations(trace):
        if step.delegation is None:
            continue
        delegation = step.delegation
        report["delegations"].append({
            "request": step.validated_arguments,
            "status": delegation.status,
            "initial_assignment": delegation.initial_assignment,
            "error": delegation.error,
            "invocation": asdict(delegation.invocation) if delegation.invocation else None,
            "observation_acknowledged": step.observation_sent,
            "delivery_error": step.delivery_error,
            "delivered_finding_ids": delegation.delivered_finding_ids,
            "child": _invocation_report(delegation.child, detailed) if delegation.child else None,
        })
    # Preserve rejected provider batches even without verbose step tracing.
    report["rejected_provider_requests"] = [{
        "iteration": step.iteration, "requested_count": step.provider_tool_request_count,
        "requests": step.provider_tool_requests,
    } for step in trace.steps if step.provider_tool_requests and not step.operations
         and step.error and step.error["code"] in {"model_error", "invalid_tool_use"}]
    report["tool_operations"] = [{
        "iteration": step.iteration,
        "tool_use_id": operation.tool_use_id,
        "tool_name": operation.tool_name,
        "outcome": operation.outcome,
        "requested_action": operation.requested_action,
        "validated_arguments": operation.validated_arguments,
        "error": operation.error,
        "delivery_error": operation.delivery_error,
        "observation_acknowledged": operation.observation_sent,
        "generated_observation_bytes": operation.observation_bytes,
        "search_discovered_chunk_ids": [ref.chunk_id for ref in operation.search_discovered],
        "generated_observation_chunk_ids": [ref.chunk_id for ref in operation.evidence],
        **({"candidate_chunk_ids": (operation.observation or {}).get("result", {}).get("chunks", [])}
           if operation.tool_name == "list_candidates" else {}),
    } for step in trace.steps for operation in step.operations]
    if detailed:
        report["steps"] = [{
            "iteration": step.iteration,
            "phase": step.phase,
            "assignment_stage": step.assignment_stage,
            "finalization_reason": step.finalization_reason,
            "requested_action": step.requested_action,
            "validated_arguments": step.validated_arguments,
            "model_seconds": step.model_seconds,
            "tool_seconds": step.tool_seconds,
            "observation_acknowledged": step.observation_sent,
            "error": step.error,
            "provider_tool_requests": step.provider_tool_requests,
            "search_discovered_chunk_ids": [ref.chunk_id for ref in step.search_discovered],
            "observation_chunk_ids": [ref.chunk_id for ref in step.evidence],
            "provider_attempts": [
                {key: value for key, value in asdict(attempt).items() if key not in {"model", "response_model"}}
                for attempt in step.model_attempts
            ],
        } for step in trace.steps]
    return report


def _matches_partition(source, year, partition):
    return (source.fiscal_year, source.document_id, source.section_id) == (
        year, partition.document_id, partition.section_id,
    )


def _year_outcome(trace, year, partition):
    children = [
        step for step in _operations(trace) if step.delegation and step.delegation.child
        and step.delegation.child.invocation.partitions == (partition,)
    ]
    generated = [
        finding for step in children if step.delegation.child.stop_reason == "finished"
        for finding in step.delegation.child.findings
        if any(_matches_partition(item.source, year, partition) for item in finding.evidence)
    ]
    delivered_ids = {finding.finding_id for step in children for finding in _acknowledged_findings(step)}
    delivered = [finding for finding in generated if finding.finding_id in delivered_ids]
    used_citations = sorted({
        item.source.chunk_id for finding in delivered if finding.finding_id in trace.used_finding_ids
        for item in finding.evidence if _matches_partition(item.source, year, partition)
        and item.source.chunk_id in trace.cited_chunk_ids
    })
    return {
        "delegated": bool(children),
        "findings_generated": [finding.finding_id for finding in generated],
        "findings_delivered": [finding.finding_id for finding in delivered],
        "citations_from_declared_used_findings": used_citations,
    }


def _live_report(trace, partitions, config, local, elapsed, detailed, finalization):
    traces = list(_walk(trace))
    usage = _usage(traces)
    by_year = {year: _year_outcome(trace, year, pair) for year, pair in partitions.items()}
    synthesis = trace.stop_reason == "finished" and bool(trace.answer and trace.used_finding_ids)
    criteria = {
        "provider_communication": usage["provider_responses_received"] > 0,
        "separate_year_delegations": all(result["delegated"] for result in by_year.values()),
        "child_findings_for_both_years": all(result["findings_generated"] for result in by_year.values()),
        "acknowledged_findings_for_both_years": all(result["findings_delivered"] for result in by_year.values()),
        "synthesis_declares_delivered_findings_used": synthesis,
        "original_citations_from_used_findings_for_both_years": all(
            result["citations_from_declared_used_findings"] for result in by_year.values()
        ),
    }
    evidence = _evidence_delivery(trace)
    return {
        "mode": "live",
        "research_question": LIVE_QUESTION,
        "run_id": trace.invocation.run_id,
        "provider": "bedrock",
        "model_id": re.sub(r"(?<=:)\d{12}(?=:)", "<redacted>", config.model),
        "region": config.region,
        "limits": {
            "local": local.model_dump(), "shared": trace.shared_limits,
            "finalization": finalization.model_dump() if finalization else None,
            "provider": {field: getattr(config, field) for field in (
                "max_retries", "max_output_tokens", "timeout_seconds", "temperature", "top_p",
            )},
        },
        "authorized_partitions": {year: asdict(pair) for year, pair in partitions.items()},
        "status": "success" if all(criteria.values()) else "incomplete",
        "criteria": criteria,
        "year_outcomes": by_year,
        "answer": trace.answer,
        "stop_reason": trace.stop_reason,
        "semantic_claims_verified": False,
        "exhaustive_descendant_coverage_rollup": False,
        "actual_maximum_depth": max(t.invocation.depth for t in traces),
        "wall_clock_seconds": elapsed,
        "shared_usage": asdict(trace.shared_budget),
        "provider_usage": usage,
        "requested_tool_calls": sum(_requested_tool_calls(t) for t in traces),
        "original_citations": [{
            **asdict(ref),
            "direct_chunk_text_delivered": ref.chunk_id in evidence["direct_chunk_text_ids"],
            "delegated_excerpts": [excerpt for excerpt in evidence["delegated_excerpts"]
                                   if excerpt["source"]["chunk_id"] == ref.chunk_id],
        } for ref in trace.coverage.visible_chunks if ref.chunk_id in trace.cited_chunk_ids],
        "trace": _invocation_report(trace, detailed),
    }


def _run_live(corpus, partitions, shared, local, detailed, finalization):
    root_config = None

    def factory(invocation):
        nonlocal root_config
        client = BedrockRootClient.from_env(disable_retries=True)
        if invocation.depth == 0:
            root_config = client.config
        return client

    scope = "; ".join(f"FY{year} = {pair.document_id}, {pair.section_id}" for year, pair in partitions.items())
    prompt = (
        f"{LIVE_QUESTION}\nAuthorized Item 1A assignments: {scope}. Make two separate delegate_research "
        "calls, one per fiscal year, each with only that year's pair. Ask each child to choose a focused "
        "query, start with limit=1, and return one supported finding about a supply disruption or continuity "
        "risk. Use the same goal for both years. A search hit includes chunk text. Ask each child to finish "
        "once it has one adequately supported finding, with a one-sentence answer and a short verbatim "
        "quote and original chunk ID in the finding. Keep the quote out of the answer to avoid repetition. "
        "Write a concise comparison of delivered findings "
        "and declare their IDs in used_finding_ids. "
        "Cite one original chunk from each year's used finding when supported. Report missing evidence "
        "and partial coverage. The excerpts support a limited comparison; avoid claims about exhaustive "
        "changes or the absence of disclosures."
    )
    runner = RecursiveResearch(corpus, factory, shared_limits=shared, local_limits=local,
                               finalization=finalization)
    start = monotonic()
    trace = runner.run(prompt, partitions=tuple(partitions.values()))
    report = _live_report(trace, partitions, root_config, local, monotonic() - start, detailed, finalization)
    print(json.dumps(trace_data(report), indent=2))
    return 0 if report["status"] == "success" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize Bedrock model requests.")
    live = parser.add_argument_group("live experiment settings")
    live.add_argument("--trace", action="store_true", help="Include detailed per-step traces.")
    for name, default in (
        ("max-depth", 1), ("max-delegations", 2), ("max-provider-attempts", 12),
        ("max-corpus-attempts", 8), ("max-partitions", 2), ("max-reported-tokens", 60_000),
        ("max-response-bytes", 6_000), ("max-iterations", 6), ("max-tool-calls", 4),
        ("child-provider-attempts", 4),
    ):
        live.add_argument("--" + name, type=int, default=default, help="Default: %(default)s.")
    live.add_argument("--deadline-seconds", type=float, default=120, help="Shared and local deadline (default: 120s).")
    live.add_argument("--finalization-attempts", type=int, default=1,
                      help="Reserved provider attempts per invocation (default: 1, 0 disables the policy).")
    live.add_argument("--finalization-max-output-tokens", type=int, default=2048,
                      help="Output token cap per protected finish request (default: %(default)s).")
    live.add_argument("--finalization-tool-choice", choices=("auto", "required"), default="auto",
                      help="Use native named finish-tool choice if supported (default: %(default)s).")
    args = parser.parse_args(argv)
    if args.live and args.finalization_attempts == 0 and args.finalization_tool_choice == "required":
        parser.error("Required finish-tool choice needs at least one reserved finalization attempt.")
    try:
        if args.live:
            shared = SharedLimits(
                max_depth=args.max_depth, max_delegations=args.max_delegations,
                max_provider_attempts=args.max_provider_attempts, max_corpus_attempts=args.max_corpus_attempts,
                max_partitions=args.max_partitions, max_reported_tokens=args.max_reported_tokens,
                deadline_seconds=args.deadline_seconds,
            )
            local = RootLimits(
                max_iterations=args.max_iterations, max_tool_calls=args.max_tool_calls,
                max_response_bytes=args.max_response_bytes, deadline_seconds=args.deadline_seconds,
            )
            finalization = FinalizationPolicy(
                child_provider_attempts=args.child_provider_attempts,
                finalization_attempts=args.finalization_attempts,
                finalization_max_output_tokens=args.finalization_max_output_tokens,
                finalization_tool_choice=args.finalization_tool_choice,
            ) if args.finalization_attempts != 0 else None
        corpus = CorpusEnvironment()
        partitions = _risk_partitions(corpus)
        if args.live:
            return _run_live(corpus, partitions, shared, local, args.trace, finalization)
        return _run_scripted(corpus, partitions)
    except FileNotFoundError:
        parser.error("The demo requires locally available canonical manifest and Parquet files.")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
