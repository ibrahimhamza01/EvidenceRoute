"""Admit whole company groups and run one scoped recursive investigation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from evidenceroute.research.rlm.budget import SharedBudget
from evidenceroute.research.rlm.controller import ExecutionTrace
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


@dataclass(frozen=True)
class PlannedResearchResult:
    plan: ResearchPlan
    groups: tuple[GroupOutcome, ...]
    coverage: ResearchCoverage
    trace: ExecutionTrace | None
    citations: tuple[EvidenceRef, ...]
    answer_status: Literal["incomplete", "unverified"]


def execute_research_plan(
    plan: ResearchPlan, runner: RecursiveResearch, *, max_groups: int = 2, root_guidance: str | None = None,
) -> PlannedResearchResult:
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
    root_slots = min(local.max_tool_calls, MAX_TOOL_BATCH * min(local.max_iterations - 1, investigation_turns))

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

    trace = None
    if admitted:
        manifest = observation_json([
            {"ticker": group.ticker, "partitions": [asdict(pair) for pair in group.targets]} for group in admitted
        ])
        question = (
            f"{plan.intent.question}\nObjective: {plan.intent.objective}. Topic: {plan.intent.topic}\n"
            f"This run admits {len(admitted)} of {len(plan.requested_companies)} requested companies. "
            f"Use one whole group per delegation: {manifest}\n"
            "Compare the same topic across the assigned filings. Synthesize only supported findings and cite "
            "original chunks. Disclose unexamined targets and unresolved or conflicting evidence. "
            "A search miss cannot establish that a risk was absent or newly introduced."
        )
        if root_guidance:
            question += f"\nSynthesis guidance: {root_guidance}"
        trace = runner.run(question, partitions=tuple(pair for group in admitted for pair in group.targets),
                           root_delegation_groups=tuple(group.targets for group in admitted))

    coverage = aggregate_research_coverage(runner.corpus, plan, trace)
    by_pair = {target.pair: target for target in coverage.targets}
    root_delegations = {}
    for outcome in coverage.invocations:
        if outcome.invocation.parent_id == "root":
            root_delegations.setdefault(frozenset(outcome.invocation.partitions or ()), []).append(outcome)

    groups = []
    for group in plan.groups:
        delegates = root_delegations.get(frozenset(group.targets), ())
        targets = [by_pair[pair] for pair in group.targets]
        status = "none"
        if targets and all(target.status == "complete" for target in targets) and group.ticker not in unresolved:
            status = "complete"
        elif any(target.status != "none" for target in targets):
            status = "partial"
        groups.append(GroupOutcome(
            group.ticker, pending.get(group.ticker), tuple(child.invocation.invocation_id for child in delegates),
            tuple(sorted({invocation_id for target in targets for invocation_id in target.failed_invocation_ids})),
            status, tuple(finding_id for child in delegates for finding_id in child.findings_acknowledged_by_parent),
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
