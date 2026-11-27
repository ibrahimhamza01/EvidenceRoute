"""Schedule independent bounded trees and consolidate their grounded results locally."""

from __future__ import annotations

from collections.abc import Iterator
from copy import copy
from dataclasses import asdict, dataclass, replace
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.research.rlm.budget import BudgetUsage, SharedBudget, SharedLimits
from evidenceroute.research.rlm.comparison import Comparison, render_comparative_scope, render_comparisons
from evidenceroute.research.rlm.discovery import CorpusScan, DiscoverySpec, discover_research_plan, scan_research_plan
from evidenceroute.research.rlm.controller import ExecutionTrace
from evidenceroute.research.rlm.findings import Finding, FindingEvidence
from evidenceroute.research.rlm.planned_research import GroupOutcome, admit_company_groups, execute_research_plan
from evidenceroute.research.rlm.planning import CompanyPartition, ResearchPlan
from evidenceroute.research.rlm.synthesis import AnswerDisposition
from evidenceroute.research.rlm.publication import PublicationState, PublishedConclusion, publish_research_findings
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.research_coverage import ResearchCoverage, aggregate_research_trees
from evidenceroute.research.rlm.tools import EvidenceRef


class CorpusLimits(BaseModel):
    """Overall allowances for sequential trees and their optional local lexical scan."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    max_batches: int = Field(ge=0)
    max_provider_attempts: int = Field(ge=0)
    max_reported_tokens: int = Field(ge=0)
    max_corpus_attempts: int = Field(ge=0)
    max_delegations: int = Field(ge=0)
    max_partitions: int = Field(ge=0)
    deadline_seconds: float = Field(gt=0)
    max_scan_chunks: int = Field(default=30_000, ge=1)
    max_scan_pages: int | None = Field(default=None, ge=0)
    scan_deadline_seconds: float = Field(default=30.0, gt=0)


@dataclass(frozen=True)
class CorpusGroupOutcome:
    ticker: str
    batch_number: int | None
    outcome: GroupOutcome | None
    pending_reason: str | None


@dataclass(frozen=True)
class BatchOutcome:
    number: int
    companies: tuple[str, ...]
    usage: BudgetUsage
    stop_reason: str
    budget_stop: str | None
    setup_or_execution_error: str | None = None


@dataclass(frozen=True)
class CorpusUsage:
    provider_attempts: int
    reported_tokens: int
    attempts_with_missing_usage: int
    estimated_tokens: int
    attempts_with_missing_estimate: int
    corpus_attempts: int
    delegated_invocations: int
    partition_assignments: int
    lexical_pages_read: int
    lexical_metadata_reads: int
    lexical_chunks_scanned: int
    elapsed_seconds: float
    finding_submission_attempts: int = 0
    accepted_submissions: int = 0
    rejected_submission_records: int = 0
    unused_capture_reservations: int = 0


@dataclass(frozen=True)
class CorpusResearchResult:
    plan: ResearchPlan
    limits: CorpusLimits
    run_id: str
    groups: tuple[CorpusGroupOutcome, ...]
    batches: tuple[BatchOutcome, ...]
    coverage: ResearchCoverage
    scan: CorpusScan | None
    usage: CorpusUsage
    root_findings: tuple[Finding, ...]
    root_selected_inherited_findings: tuple[Finding, ...]
    acknowledged_child_findings: tuple[Finding, ...]
    retained_unacknowledged_child_findings: tuple[Finding, ...]
    root_comparisons: tuple[Comparison, ...]
    acknowledged_child_comparisons: tuple[Comparison, ...]
    evidence: tuple[FindingEvidence, ...]
    citations: tuple[EvidenceRef, ...]
    assessed_chunk_ids: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]
    diagnostics: tuple[dict, ...]
    stop_reason: str
    execution_status: Literal["finished", "incomplete"]
    research_status: Literal["incomplete", "unverified"]
    answer: str
    comparative_capture: bool = False
    answer_citations: tuple[EvidenceRef, ...] = ()
    publication_state: PublicationState = "insufficient_evidence"

    answer_disposition: AnswerDisposition | None = None
    published_conclusions: tuple[PublishedConclusion, ...] = ()

    @property
    def group_counts(self) -> dict[str, int]:
        return {
            "requested": len(self.groups),
            "scheduled": sum(group.batch_number is not None for group in self.groups),
            "attempted": sum(bool(group.outcome and group.outcome.initial_invocation_id) for group in self.groups),
            "completed": sum(bool(group.outcome and (group.outcome.assignment_status == "completed"
                                                     or group.outcome.result_acknowledged)) for group in self.groups),
            "failed": sum(bool(group.outcome and group.outcome.assignment_status == "failed") for group in self.groups),
            "pending": sum(group.pending_reason is not None for group in self.groups),
        }


# Allow one full configured allocation before launch. Surplus is available to later trees only after settlement.
_RESOURCES = {
    "max_provider_attempts": "provider_attempts",
    "max_reported_tokens": "reported_tokens",
    "max_corpus_attempts": "corpus_attempts",
    "max_delegations": "delegated_invocations",
    "max_partitions": "partition_assignments",
}


def _schedule(
    plan: ResearchPlan, runner: RecursiveResearch, max_groups: int,
) -> tuple[list[tuple[CompanyPartition, ...]], dict[str, str]]:
    batches = []
    remaining = plan
    while True:
        admitted, blocked = admit_company_groups(remaining, runner, max_groups=max_groups, delegation_first=True)
        if not admitted:
            for missing in plan.missing_requirements:
                blocked[missing.company] = "unresolved_requirements"
            return batches, blocked
        batches.append(tuple(admitted))
        selected = {group.ticker for group in admitted}
        remaining = replace(remaining, groups=tuple(group for group in remaining.groups if group.ticker not in selected))


def _walk(trace: ExecutionTrace) -> Iterator[ExecutionTrace]:
    yield trace
    for step in trace.steps:
        for operation in step.operations:
            if operation.delegation and operation.delegation.child:
                yield from _walk(operation.delegation.child)


def _diagnostics(node: ExecutionTrace, detailed: bool) -> dict:
    decisions = []
    for step in node.steps:
        operations = []
        for op in step.operations:
            operation = {"tool": op.tool_name, "outcome": op.outcome, "observation_bytes": op.observation_bytes,
                         "attempted_delivery": op.observation_attempted, "acknowledged": op.observation_sent,
                         "error": (op.error or {}).get("code"), "delivery_error": op.delivery_error}
            if detailed:
                operation.update(tool_use_id=op.tool_use_id, chunk_ids=[ref.chunk_id for ref in op.evidence])
            if op.tool_name == "submit_findings" and op.outcome in {"executed", "rejected"}:
                operation["capture_rejected_records"] = len(op.finding_validation)
            if op.finding_validation:
                operation["finding_validation"] = [asdict(issue) for issue in op.finding_validation]
            if op.accepted_finding_refs:
                submitted = (op.observation_attempted or op.observation_sent) and not op.delivery_error
                operation.update(accepted_finding_refs=op.accepted_finding_refs,
                                 offered_finding_refs=op.offered_finding_refs,
                                 submitted_finding_refs=op.offered_finding_refs if submitted else (),
                                 acknowledged_finding_refs=op.acknowledged_finding_refs)
            if op.prior_finding_references:
                offered = tuple(fid for _, fid in op.prior_finding_references)
                submitted = (op.observation_attempted or op.observation_sent) and not op.delivery_error
                operation.update(offered_prior_finding_ids=offered,
                                 submitted_prior_finding_ids=offered if submitted else (),
                                 acknowledged_prior_finding_ids=op.acknowledged_prior_finding_ids,
                                 used_prior_finding_ids=sorted(set(node.used_finding_ids) & set(op.acknowledged_prior_finding_ids)))
            edge = op.delegation
            if edge and edge.child:
                submitted = (op.observation_attempted or op.observation_sent) and not op.delivery_error
                operation.update(child_id=edge.invocation.invocation_id, full_observation_bytes=edge.full_observation_bytes,
                                 retained_finding_ids=[item.finding_id for item in edge.child.transferable_findings],
                                 child_status=edge.status, child_error=edge.error,
                                 offered_finding_ids=edge.offered_finding_ids,
                                 omitted_finding_ids=tuple(item.finding_id for item in edge.child.transferable_findings
                                                          if item.finding_id not in edge.offered_finding_ids),
                                 finding_references=dict(edge.finding_references),
                                 submitted_finding_ids=edge.offered_finding_ids if submitted else (),
                                 acknowledged_finding_ids=edge.delivered_finding_ids,
                                 used_finding_ids=sorted(set(node.used_finding_ids) & set(edge.delivered_finding_ids)))
            operations.append(operation)
        decision = {"iteration": step.iteration, "phase": step.phase, "assignment_stage": step.assignment_stage,
                    "finalization_reason": step.finalization_reason, "recovery": step.recovery,
                    "envelope_bytes": step.envelope_bytes, "error": (step.error or {}).get("code"), "operations": operations,
                    "attempts": [{"stop_reason": a.provider_stop_reason, "output_cap": a.output_cap, "error": a.error,
                                  "request_bytes": a.request_bytes, "timeout_seconds": a.timeout_seconds,
                                  **({"input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                                      "estimated_tokens": a.estimated_tokens, "latency_seconds": a.latency_seconds}
                                     if detailed else {})} for a in step.model_attempts]}
        if detailed:
            decision.update(model_seconds=step.model_seconds, requested_tool_count=step.provider_tool_request_count,
                            offered_tool_names=step.offered_tool_names)
        if step.finding_validation and not step.operations:
            decision["finding_validation"] = [asdict(issue) for issue in step.finding_validation]
        decisions.append(decision)
    attempts = node.model_attempts
    submissions = [op for step in node.steps for op in step.operations if op.tool_name == "submit_findings"]
    metrics = {
        "provider_attempts": len(attempts),
        "reported_tokens": sum((a.input_tokens or 0) + (a.output_tokens or 0) +
                               (a.cache_read_input_tokens or 0) + (a.cache_creation_input_tokens or 0) for a in attempts),
        "attempts_with_missing_usage": sum(a.input_tokens is None or a.output_tokens is None for a in attempts),
        "elapsed_seconds": node.elapsed_seconds,
        "provider_seconds": sum(a.latency_seconds for a in attempts),
        "empty_captures": sum(op.outcome == "executed" and not op.accepted_finding_refs and not op.finding_validation
                              and (op.observation or {}).get("ok", False) for op in submissions),
        "capture_calls": sum(op.outcome in {"executed", "rejected"} for op in submissions),
        "rejected_records": sum(len(op.finding_validation) for op in submissions),
        "comparison_diagnostics": sum(len(step.finding_validation) for step in node.steps if not step.operations),
    }
    return {"invocation_id": node.invocation.invocation_id, "stop_reason": node.stop_reason,
            "budget_stop": node.budget_stop, "metrics": metrics, "generated_finding_ids": [item.finding_id for item in node.findings],
            "selected_local_finding_ids": node.selected_local_finding_ids,
            "unused_capture_reservation": node.unused_capture_reservation,
            "decisions": decisions}


def execute_corpus_research(
    plan: ResearchPlan, runner: RecursiveResearch, *, limits: CorpusLimits, max_groups: int = 2,
    scan_terms: tuple[str, ...] | None = None, root_guidance: str | None = None,
    detailed_trace: bool = False, discovery: DiscoverySpec | None = None, _deadline: float | None = None,
    _target_companies: tuple[str, ...] | None = None, _cached_scan: CorpusScan | None = None,
) -> CorpusResearchResult:
    """Run a resolved plan, settling one bounded tree before admitting the next."""
    if runner.comparison_scan is not None or runner.discovery_scan is not None:
        raise ValueError("Pass discovery or scan_terms so the lexical scan is charged to this corpus execution.")
    if discovery is not None and scan_terms is not None:
        raise ValueError("Choose either normalized discovery or the literal-term baseline.")
    review_discovery = discovery is not None and plan.intent.objective == "review"
    comparative = scan_terms is not None or (discovery is not None and not review_discovery)
    if comparative and (plan.intent.objective != "year_over_year" or len(set(plan.intent.fiscal_years)) != 2):
        raise ValueError("Comparative execution requires a two-year comparison.")
    clock = runner.clock
    started = clock()
    deadline = started + limits.deadline_seconds
    if _deadline is not None:
        deadline = min(deadline, _deadline)
    run_id = uuid4().hex
    work_plan = plan
    if _target_companies is not None:
        selected_companies = set(_target_companies)
        if not selected_companies.issubset(group.ticker for group in plan.groups):
            raise ValueError("Targeted companies must be resolved in the original plan.")
        work_plan = replace(plan, groups=tuple(group for group in plan.groups if group.ticker in selected_companies),
                            requested_companies=tuple(ticker for ticker in plan.requested_companies if ticker in selected_companies),
                            missing_requirements=tuple(row for row in plan.missing_requirements if row.company in selected_companies))
    schedule, pending = _schedule(work_plan, runner, max_groups)
    pending.update(dict.fromkeys(set(plan.requested_companies) - set(work_plan.requested_companies), "not_selected"))
    scheduled = {group.ticker: number for number, groups in enumerate(schedule, 1) for group in groups}
    scan = None
    remaining_seconds = deadline - clock()

    def tree_limits(groups: tuple[CompanyPartition, ...]) -> SharedLimits:
        if comparative and runner.comparative_capture:
            return runner.shared_limits.model_copy(update={
                "max_delegations": len(groups),
                "max_partitions": sum(len(group.targets) for group in groups),
            })
        return runner.shared_limits

    scan_allowed = _target_companies is None or (
        bool(schedule) and limits.max_batches > 0
        and all(getattr(tree_limits(schedule[0]), cap) <= getattr(limits, cap) for cap in _RESOURCES)
    )
    reused_scan = scan_allowed and _cached_scan is not None
    if reused_scan:
        scan = _cached_scan
    elif scan_allowed and (comparative or review_discovery) and remaining_seconds > 0:
        options = dict(max_chunks=limits.max_scan_chunks, max_pages=limits.max_scan_pages,
                       deadline_seconds=min(limits.scan_deadline_seconds, remaining_seconds), clock=clock)
        if discovery is not None:
            scan = discover_research_plan(runner.corpus, work_plan, discovery, **options)
        else:
            scan = scan_research_plan(runner.corpus, work_plan, scan_terms, **options)
    if scan is not None:
        runner = copy(runner)
        if comparative:
            runner.comparison_scan = scan
        else:
            runner.discovery_scan = scan

    if comparative and runner.comparative_capture and scan is not None:
        rank: dict[str, list[int]] = {}
        for row in scan.sections:
            rank.setdefault(row.ticker, []).append(len(row.candidates))
        work_plan = replace(work_plan, groups=tuple(sorted(work_plan.groups, key=lambda group: (
            -sum(bool(count) for count in rank.get(group.ticker, ())),
            -sum(rank.get(group.ticker, ())), group.ticker,
        ))))
        schedule, blocked = _schedule(work_plan, runner, max_groups)
        pending.update(blocked)
        scheduled = {group.ticker: number for number, groups in enumerate(schedule, 1) for group in groups}

    batches = []
    traces = []
    results = []
    outcomes: dict[str, GroupOutcome] = {}
    errors: list[tuple[str, str]] = []
    used = dict.fromkeys(_RESOURCES.values(), 0)
    stop = "schedule_completed"
    for number, groups in enumerate(schedule, 1):
        if clock() >= deadline:
            stop = "deadline"
            break
        if len(batches) >= limits.max_batches:
            stop = "batch_limit"
            break
        shared_limits = tree_limits(groups)
        batch_runner = runner
        if shared_limits is not runner.shared_limits:
            batch_runner = copy(runner)
            batch_runner.shared_limits = shared_limits
        blocked_resource = next((cap for cap, counter in _RESOURCES.items()
                                 if used[counter] + getattr(batch_runner.shared_limits, cap) > getattr(limits, cap)), None)
        if blocked_resource is not None:
            stop = f"insufficient_{blocked_resource.removeprefix('max_')}"
            break

        companies = tuple(group.ticker for group in groups)
        batch_plan = replace(plan, groups=groups, requested_companies=companies, missing_requirements=())
        budget = SharedBudget(batch_runner.shared_limits, clock, finalization=runner.finalization, deadline=deadline)
        prefix = f"{run_id}.batch-{number}."
        try:
            result = execute_research_plan(batch_plan, batch_runner, max_groups=max_groups, root_guidance=root_guidance,
                                           delegation_first=True, _budget=budget, _id_prefix=prefix,
                                           _requested_scope=(len(plan.requested_companies), plan.required_target_count))
        except Exception as exc:
            # A host failure may occur after provider or corpus work. Settle the actual budget either way.
            error = type(exc).__name__
            errors.append((f"{prefix}root", error))
            batch = BatchOutcome(number, companies, budget.snapshot(), "batch_error", None, error)
            pending.update(dict.fromkeys(companies, "batch_error"))
        else:
            trace = result.trace
            assert trace is not None
            traces.append(trace)
            results.append(result)
            batch = BatchOutcome(number, companies, budget.snapshot(), trace.stop_reason, trace.budget_stop)
            for group in result.groups:
                outcomes[group.ticker] = group
                if group.assignment_status == "failed" and (
                    (comparative and runner.comparative_capture) or not group.result_acknowledged
                ):
                    pending[group.ticker] = "child_failed"
                elif not group.result_acknowledged:
                    pending[group.ticker] = group.assignment_pending_reason or "child_result_unacknowledged"
                elif trace.stop_reason != "finished":
                    pending[group.ticker] = "batch_root_unfinished"
        batches.append(batch)
        for counter in used:
            used[counter] += getattr(batch.usage, counter)

    groups = tuple(CorpusGroupOutcome(ticker, scheduled.get(ticker), outcomes.get(ticker),
                                     pending.get(ticker, stop if scheduled.get(ticker, 0) > len(batches) else None))
                   for ticker in plan.requested_companies)
    coverage = aggregate_research_trees(runner.corpus, plan, traces)
    root_findings = tuple(finding for trace in traces for finding in trace.findings)
    selected = tuple(finding for trace in traces if trace.stop_reason == "finished"
                     for finding in trace.selected_inherited_findings)
    roots = {trace.invocation.invocation_id for trace in traces}
    child_findings = tuple(finding for child in coverage.invocations if child.invocation.parent_id in roots
                           for finding in child.acknowledged_findings)
    if not comparative:
        used_ids = {finding_id for trace in traces if trace.stop_reason == "finished" for finding_id in trace.used_finding_ids}
        selected = tuple({finding.finding_id: finding for finding in (*selected, *child_findings, *runner._prior_findings)
                          if finding.finding_id in used_ids}.values())
    retained = tuple(finding for child in coverage.invocations if child.invocation.parent_id is not None
                     for finding in child.findings
                     if finding.finding_id not in child.findings_acknowledged_by_parent)
    root_comparisons = tuple(item for trace in traces if trace.stop_reason == "finished" for item in trace.comparisons)
    child_comparisons = tuple(item for result in results for item in result.acknowledged_comparisons)
    citations = {ref.chunk_id: ref for result in results for ref in result.citations}
    estimates = []
    diagnostics = []
    accepted_submissions = rejected_submissions = unused_reservations = 0
    for trace in traces:
        for node in _walk(trace):
            diagnostics.append(_diagnostics(node, detailed_trace))
            invocation_id = node.invocation.invocation_id
            unused_reservations += node.unused_capture_reservation
            submissions = [op for step in node.steps for op in step.operations if op.tool_name == "submit_findings"]
            accepted_submissions += len({ref for op in submissions for ref in op.accepted_finding_refs})
            rejected_submissions += sum(len(op.finding_validation) for op in submissions)
            for step in node.steps:
                if step.error:
                    errors.append((invocation_id, step.error["code"]))
                for op in step.operations:
                    if op is not step and op.error:
                        errors.append((invocation_id, op.error["code"]))
                    if op.delivery_error:
                        errors.append((invocation_id, op.delivery_error))
                for attempt in step.model_attempts:
                    if attempt.error:
                        errors.append((invocation_id, attempt.error))
                    if attempt.estimated_tokens is not None:
                        estimates.append(attempt.estimated_tokens)
    usage = CorpusUsage(**used, attempts_with_missing_usage=sum(b.usage.attempts_with_missing_usage for b in batches),
                        estimated_tokens=sum(estimates), attempts_with_missing_estimate=used["provider_attempts"] - len(estimates),
                        lexical_pages_read=scan.pages_read if scan and not reused_scan else 0,
                        lexical_metadata_reads=scan.metadata_reads if scan and not reused_scan else 0,
                        lexical_chunks_scanned=sum(row.scanned_chunks for row in scan.sections) if scan and not reused_scan else 0,
                        finding_submission_attempts=sum(b.usage.finding_submission_attempts for b in batches),
                        accepted_submissions=accepted_submissions, rejected_submission_records=rejected_submissions,
                        unused_capture_reservations=unused_reservations, elapsed_seconds=clock() - started)
    evidence = tuple(dict.fromkeys(item for finding in (*root_findings, *child_findings, *selected) for item in finding.evidence))
    assessed = {key for item in (*root_comparisons, *child_comparisons) for key in item.assessed_chunk_ids}
    assessed.update(item.source.chunk_id for finding in (*root_findings, *child_findings)
                    if finding.analysis_status in {"analyzed", "conflicting"} for item in finding.evidence)
    finished = bool(batches) and all(group.pending_reason is None for group in groups)
    comparable = {item.ticker for item in root_comparisons if item.outcome == "comparable_both_years"}
    unresolved_comparisons = comparative and (
        comparable != set(plan.requested_companies) or any(item.outcome != "comparable_both_years" for item in root_comparisons)
    )
    incomplete = not finished or unresolved_comparisons or any(result.answer_status == "incomplete" for result in results)
    pending_groups = ", ".join(group.ticker for group in groups if group.pending_reason) or "None."
    finished_roots = sum(batch.stop_reason == "finished" for batch in batches)
    answer = (f"Execution has {finished_roots} finished batch roots out of {len(schedule)} scheduled batches.\n"
              f"Pending execution or scope. {pending_groups}")
    if retained:
        answer += f"\nRetained child findings without parent acknowledgement. {len(retained)}."
    compact_comparative = comparative and runner.comparative_capture
    published = ()
    disposition = None
    if comparative and not compact_comparative:
        assessment = "Batch-root comparisons.\n" + render_comparisons(root_comparisons)
        if child_comparisons:
            assessment += "\n\nAcknowledged child comparisons.\n" + render_comparisons(child_comparisons)
        answer += "\n\n" + render_comparative_scope(
            root_comparisons, assessment, requested=plan.requested_companies,
            admitted={group.ticker for group in groups if group.outcome and group.outcome.admitted}, incomplete=incomplete,
        )
        sources = {item.source.chunk_id: item.source for item in evidence}
        cited_ids = dict.fromkeys(chunk_id for item in (*root_comparisons, *child_comparisons)
                                  for chunk_id in item.assessed_chunk_ids)
        answer_citations = tuple(sources[chunk_id] for chunk_id in cited_ids)
        publication_state: PublicationState = "comparison_evidence" if answer_citations else "insufficient_evidence"
    else:
        publication = publish_research_findings(
            coverage.invocations, incomplete=incomplete, compact_comparative=compact_comparative,
            prior_findings=runner._prior_findings, generic_plan=None if comparative else plan,
        )
        answer += "\n\n" + publication.answer
        answer_citations = publication.answer_citations
        publication_state = publication.publication_state
        disposition = publication.answer_disposition
        published = publication.published_conclusions
    return CorpusResearchResult(plan, limits, run_id, groups, tuple(batches), coverage, scan, usage,
                                root_findings, selected, child_findings, retained, root_comparisons, child_comparisons, evidence,
                                tuple(citations[key] for key in sorted(citations)), tuple(sorted(assessed)), tuple(errors),
                                tuple(diagnostics), stop,
                                "finished" if finished else "incomplete", "incomplete" if incomplete else "unverified",
                                answer, compact_comparative, answer_citations, publication_state, disposition, published)
