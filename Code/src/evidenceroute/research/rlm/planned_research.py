"""Admit whole company groups and run one scoped recursive investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from evidenceroute.research.rlm.budget import SharedBudget
from evidenceroute.research.rlm.comparison import Comparison, render_comparisons
from evidenceroute.research.rlm.controller import ExecutionTrace
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.planning import CompanyPartition, ResearchPlan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.research_coverage import AccessStatus, ResearchCoverage, aggregate_research_coverage
from evidenceroute.research.rlm.tools import MAX_TOOL_BATCH, EvidenceRef, observation_json


@dataclass(frozen=True)
class GroupOutcome:
    ticker: str
    pending_reason: str | None
    root_delegate_ids: tuple[str, ...]
    failed_invocation_ids: tuple[str, ...]
    access_status: AccessStatus
    findings_acknowledged_by_root: tuple[str, ...]
    admitted: bool
    assignment_status: Literal["unresolved", "not_admitted", "unassigned", "completed", "failed"]
    initial_invocation_id: str | None
    initial_result_acknowledged: bool
    result_acknowledged: bool
    assignment_pending_reason: str | None


@dataclass(frozen=True)
class PartialComparativeResult:
    answer: str
    child_findings: tuple[Finding, ...]
    child_comparisons: tuple[Comparison, ...]
    failed_groups: tuple[str, ...]
    targets_by_access: dict[AccessStatus, int]
    unresolved_targets: int
    status: Literal["acknowledged_child_evidence"] = "acknowledged_child_evidence"
    root_synthesis: Literal[False] = False


@dataclass(frozen=True)
class PlannedResearchResult:
    plan: ResearchPlan
    groups: tuple[GroupOutcome, ...]
    coverage: ResearchCoverage
    trace: ExecutionTrace | None
    citations: tuple[EvidenceRef, ...]
    answer_status: Literal["incomplete", "unverified"]

    @property
    def acknowledged_comparisons(self) -> tuple[Comparison, ...]:
        comparisons = []
        for step in self.trace.steps if self.trace else ():
            for operation in step.operations:
                edge = operation.delegation
                if not edge or not edge.child or not operation.observation_sent or operation.delivery_error:
                    continue
                delivered = set(edge.delivered_finding_ids)
                comparisons.extend(edge.child.comparisons[i] for i in edge.offered_comparison_indices
                                   if delivered.issuperset(edge.child.comparisons[i].finding_ids))
        return tuple(comparisons)

    @property
    def partial_comparative_result(self) -> PartialComparativeResult | None:
        if not self.trace or self.trace.stop_reason == "finished":
            return None
        findings = tuple(finding for child in self.coverage.invocations
                         if child.invocation.parent_id == self.trace.invocation.invocation_id
                         for finding in child.findings if finding.finding_id in child.findings_acknowledged_by_parent)
        if not findings:
            return None
        comparisons = self.acknowledged_comparisons
        return PartialComparativeResult(
            "Partial result from acknowledged child evidence. Root synthesis did not finish.\n" + render_comparisons(comparisons),
            findings, comparisons,
            tuple(group.ticker for group in self.groups if group.assignment_status == "failed" and not group.result_acknowledged),
            {status: sum(target.status == status for target in self.coverage.targets)
             for status in ("none", "partial", "complete")},
            len(self.coverage.missing_requirements),
        )


def admit_company_groups(
    plan: ResearchPlan, runner: RecursiveResearch, *, max_groups: int, delegation_first: bool,
) -> tuple[list[CompanyPartition], dict[str, str]]:
    if type(max_groups) is not int or max_groups < 0:
        raise ValueError("max_groups must be a nonnegative integer.")

    limits = runner.shared_limits
    local = runner.local_limits
    # Preflight with the same allocation rules as the runner, including slots
    # held for prospective children even when fewer groups are selected.
    preflight = SharedBudget(limits, finalization=runner.finalization)
    with preflight.invocation_allocation(root=True):
        allocation = preflight.active_allocation
        investigation_turns = allocation.investigation_left if allocation else limits.max_provider_attempts - 1
    finish_decisions = allocation.finalization_left if allocation else 1
    assignments_per_turn = 1 if delegation_first else MAX_TOOL_BATCH
    root_slots = min(local.max_tool_calls,
                     assignments_per_turn * min(local.max_iterations - finish_decisions, investigation_turns))

    unresolved = {entry.company for entry in plan.missing_requirements}
    pending: dict[str, str] = {}
    admitted: list[CompanyPartition] = []
    pairs_left = limits.max_partitions
    for group in sorted(plan.groups, key=lambda item: item.ticker):
        if group.ticker in unresolved or not group.targets:
            reason = "unresolved_requirements"
        elif limits.max_depth < 1:
            reason = "depth_limit"
        elif len(admitted) >= max_groups:
            reason = "group_limit"
        elif len(admitted) >= limits.max_delegations:
            reason = "delegation_limit"
        elif len(admitted) >= root_slots:
            reason = "root_work_limit"
        elif len(group.targets) > pairs_left:
            reason = "partition_limit"
        else:
            admitted.append(group)
            pairs_left -= len(group.targets)
            continue
        pending[group.ticker] = reason
    return admitted, pending


def execute_research_plan(
    plan: ResearchPlan, runner: RecursiveResearch, *, max_groups: int = 2, root_guidance: str | None = None,
    delegation_first: bool | None = None, _budget: SharedBudget | None = None, _id_prefix: str = "",
    _requested_scope: tuple[int, int] | None = None,
) -> PlannedResearchResult:
    if delegation_first is None:
        delegation_first = plan.intent.comprehensive
    admitted, pending = admit_company_groups(plan, runner, max_groups=max_groups, delegation_first=delegation_first)
    unresolved = {entry.company for entry in plan.missing_requirements}

    trace = None
    if admitted:
        manifest = observation_json([
            {"ticker": group.ticker, "partitions": [asdict(pair) for pair in group.targets]} for group in admitted
        ])
        company_count = _requested_scope[0] if _requested_scope else len(plan.requested_companies)
        question = (
            f"{plan.intent.question}\nObjective: {plan.intent.objective}. Topic: {plan.intent.topic}\n"
            f"This run admits {len(admitted)} of {company_count} requested companies. "
            f"Use one whole group per delegation: {manifest}\n"
            "Compare the same topic across the assigned filings. Synthesize only supported findings and cite "
            "original chunks. Disclose unexamined targets and unresolved or conflicting evidence. "
            "A search miss cannot establish that a risk was absent or newly introduced."
        )
        if _requested_scope:
            question += f"\nThe full research plan requires {_requested_scope[1]} filing-section targets."
        if root_guidance:
            question += f"\nSynthesis guidance: {root_guidance}"
        trace = runner.run(question, partitions=tuple(pair for group in admitted for pair in group.targets),
                           root_delegation_groups=tuple(group.targets for group in admitted),
                           delegation_first=delegation_first, _budget=_budget, _id_prefix=_id_prefix)

    coverage = aggregate_research_coverage(runner.corpus, plan, trace)
    by_pair = {target.pair: target for target in coverage.targets}
    root_delegations = {}
    for outcome in coverage.invocations:
        if trace is not None and outcome.invocation.parent_id == trace.invocation.invocation_id:
            root_delegations.setdefault(frozenset(outcome.invocation.partitions or ()), []).append(outcome)

    admitted_tickers = {group.ticker for group in admitted}
    resolved = {group.ticker: group for group in plan.groups}
    groups = []
    for ticker in plan.requested_companies:
        pairs = resolved[ticker].targets if ticker in resolved else ()
        delegates = root_delegations.get(frozenset(pairs), ())
        first = delegates[0] if delegates else None
        acknowledged = {child.invocation.invocation_id for child in delegates
                        if child.parent_acknowledged and child.delegation_status in {"finished", "incomplete"}
                        and (not child.findings or child.findings_acknowledged_by_parent)}
        is_admitted = ticker in admitted_tickers
        if ticker in unresolved:
            assignment_status = "unresolved"
        elif not is_admitted:
            assignment_status = "not_admitted"
        elif first is None:
            assignment_status = "unassigned"
        else:
            assignment_status = "completed" if first.stop_reason == "finished" else "failed"
        targets = [by_pair[pair] for pair in pairs]
        status = "none"
        if targets and all(target.status == "complete" for target in targets) and ticker not in unresolved:
            status = "complete"
        elif any(target.status != "none" for target in targets):
            status = "partial"
        groups.append(GroupOutcome(
            ticker, "unresolved_requirements" if ticker in unresolved else pending.get(ticker),
            tuple(child.invocation.invocation_id for child in delegates),
            tuple(sorted({invocation_id for target in targets for invocation_id in target.failed_invocation_ids})),
            status, tuple(finding_id for child in delegates for finding_id in child.findings_acknowledged_by_parent),
            is_admitted, assignment_status, first.invocation.invocation_id if first else None,
            bool(first and first.invocation.invocation_id in acknowledged), bool(acknowledged),
            (trace.initial_assignment_stop or "no_initial_attempt") if trace and assignment_status == "unassigned" else None,
        ))

    citations = tuple(ref for ref in trace.coverage.visible_chunks if ref.chunk_id in trace.cited_chunk_ids) if trace else ()
    incomplete = (
        not trace or trace.stop_reason != "finished" or trace.analysis_status != "analyzed"
        or bool(coverage.missing_requirements)
        or any(group.access_status != "complete" or group.pending_reason or group.failed_invocation_ids
               or not set(group.findings_acknowledged_by_root).intersection(trace.used_finding_ids) for group in groups)
        or any(outcome.analysis_status != "analyzed"
               or any(finding.analysis_status != "analyzed" or finding.conflict for finding in outcome.findings)
               for outcome in coverage.invocations)
    )
    return PlannedResearchResult(plan, tuple(groups), coverage, trace, citations,
                                 "incomplete" if incomplete else "unverified")
