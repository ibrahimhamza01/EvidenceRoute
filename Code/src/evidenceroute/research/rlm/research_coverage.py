"""Roll up acknowledged access across a run without widening citation eligibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from evidenceroute.research.rlm.controller import ExecutionTrace
from evidenceroute.research.rlm.coverage import SectionTarget, _merge_ranges
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import AnalysisStatus, Finding
from evidenceroute.research.rlm.planning import MissingRequirement, ResearchPlan
from evidenceroute.research.rlm.recursive import DelegationTrace, Invocation

AccessStatus = Literal["none", "partial", "complete"]


@dataclass(frozen=True)
class SectionAccess:
    pair: SectionTarget
    start: int
    end: int


@dataclass(frozen=True)
class InvocationOutcome:
    invocation: Invocation
    stop_reason: str | None
    budget_stop: str | None
    analysis_status: AnalysisStatus | None
    delegation_status: str | None
    delegation_error: str | None
    parent_acknowledged: bool
    findings: tuple[Finding, ...]
    findings_acknowledged_by_parent: tuple[str, ...]
    used_finding_ids: tuple[str, ...]
    cited_chunk_ids: tuple[str, ...]
    whole_chunk_ids: tuple[str, ...]
    section_windows: tuple[SectionAccess, ...]


@dataclass(frozen=True)
class TargetCoverage:
    pair: SectionTarget
    ticker: str
    fiscal_year: int
    admitted: bool
    assigned_child_ids: tuple[str, ...]
    failed_invocation_ids: tuple[str, ...]
    status: AccessStatus
    discovered_chunk_ids: tuple[str, ...]
    whole_chunk_ids: tuple[str, ...]
    chunk_inventory_count: int
    chunk_inventory_complete: bool
    character_ranges: tuple[tuple[int, int], ...]
    section_character_count: int
    section_text_complete: bool
    delegated_excerpt_chunk_ids: tuple[str, ...]
    direct_span_chunk_ids: tuple[str, ...]
    generated_finding_ids: tuple[str, ...]
    acknowledged_finding_ids: tuple[str, ...]
    used_finding_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearchCoverage:
    targets: tuple[TargetCoverage, ...]
    missing_requirements: tuple[MissingRequirement, ...]
    invocations: tuple[InvocationOutcome, ...]
    rejected_delegations: tuple[tuple[str, str | None], ...]

    @property
    def required_target_count(self) -> int:
        return len(self.targets) + len(self.missing_requirements)


@dataclass
class _Access:
    chunks: set[str] = field(default_factory=set)
    discovered: set[str] = field(default_factory=set)
    ranges: list[tuple[int, int]] = field(default_factory=list)
    excerpts: set[str] = field(default_factory=set)
    direct_spans: set[str] = field(default_factory=set)
    assigned: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    generated: set[str] = field(default_factory=set)
    acknowledged: set[str] = field(default_factory=set)
    used: set[str] = field(default_factory=set)


def aggregate_research_coverage(
    corpus: CorpusEnvironment, plan: ResearchPlan, trace: ExecutionTrace | None,
) -> ResearchCoverage:
    """Traverse invocation edges once. Only successful acknowledged operations add access."""
    access = {pair: _Access() for pair in plan.resolved_targets}
    outcomes: list[InvocationOutcome] = []
    rejected: list[tuple[str, str | None]] = []

    def visit(
        invocation: Invocation, run: ExecutionTrace | None, edge: DelegationTrace | None = None,
        acknowledged: bool = False, parent_used: tuple[str, ...] = (),
    ) -> None:
        findings = run.findings if run else ()
        delivered = edge.delivered_finding_ids if edge is not None and acknowledged else ()
        if invocation.parent_id is not None:
            for pair in invocation.partitions or ():
                if pair in access:
                    access[pair].assigned.append(invocation.invocation_id)
        if run is None or run.stop_reason != "finished":
            for pair in invocation.partitions or ():
                if pair in access:
                    access[pair].failed.append(invocation.invocation_id)

        for finding in findings:
            for item in finding.evidence:
                source = item.source
                target = access.get(SectionTarget(source.document_id, source.section_id))
                if target is None:
                    continue
                target.generated.add(finding.finding_id)
                if finding.finding_id in delivered:
                    target.acknowledged.add(finding.finding_id)
                    target.excerpts.add(source.chunk_id)
                    if finding.finding_id in parent_used:
                        target.used.add(finding.finding_id)

        chunks: set[str] = set()
        windows: list[SectionAccess] = []
        children = []
        for step in run.steps if run else ():
            for operation in step.operations:
                observation = operation.observation or {}
                sent = operation.observation_sent and not operation.delivery_error
                delegation = operation.delegation
                if delegation is not None:
                    if delegation.invocation is None:
                        rejected.append((invocation.invocation_id, delegation.error))
                    else:
                        children.append((delegation, sent))
                    continue
                for ref in operation.search_discovered:
                    target = access.get(SectionTarget(ref.document_id, ref.section_id))
                    if target is not None:
                        target.discovered.add(ref.chunk_id)
                if not sent or not observation.get("ok"):
                    continue
                if operation.tool_name in {"search_chunks", "read_chunks", "get_chunk"}:
                    for ref in operation.evidence:
                        target = access.get(SectionTarget(ref.document_id, ref.section_id))
                        if target is not None:
                            target.chunks.add(ref.chunk_id)
                            chunks.add(ref.chunk_id)
                elif operation.tool_name == "get_section":
                    result = observation["result"]
                    meta = result["metadata"]
                    pair = SectionTarget(meta["document_id"], meta["section_id"])
                    target = access.get(pair)
                    if target is not None and result["text"]:
                        start = result["offset"]
                        end = start + len(result["text"])
                        target.ranges.append((start, end))
                        windows.append(SectionAccess(pair, start, end))
                elif operation.tool_name == "get_spans":
                    for ref in operation.evidence:
                        target = access.get(SectionTarget(ref.document_id, ref.section_id))
                        if target is not None:
                            target.direct_spans.add(ref.chunk_id)

        outcomes.append(InvocationOutcome(
            invocation, run.stop_reason if run else None, run.budget_stop if run else None,
            run.analysis_status if run else None, edge.status if edge else None, edge.error if edge else None,
            acknowledged, findings, delivered, run.used_finding_ids if run else (),
            run.cited_chunk_ids if run else (), tuple(sorted(chunks)), tuple(windows),
        ))
        for child, child_acknowledged in children:
            assert child.invocation is not None
            visit(child.invocation, child.child, child, child_acknowledged, run.used_finding_ids)

    admitted = ()
    if trace is not None:
        if trace.invocation is None or trace.invocation.partitions is None:
            raise ValueError("Research coverage requires an explicitly scoped recursive trace.")
        admitted = trace.invocation.partitions
        if not set(admitted).issubset(access):
            raise ValueError("The recursive trace includes targets outside the research plan.")
        visit(trace.invocation, trace)

    targets = []
    for pair, seen in access.items():
        meta = corpus.get_section(pair.document_id, pair.section_id, limit=1).metadata
        ranges = _merge_ranges(seen.ranges)
        chunks_complete = len(seen.chunks) == meta.chunk_count
        text_complete = ranges == ((0, meta.char_count),)
        status = "complete" if chunks_complete or text_complete else "partial" if seen.chunks or ranges or seen.direct_spans else "none"
        targets.append(TargetCoverage(
            pair, meta.ticker, meta.fiscal_year, pair in admitted, tuple(seen.assigned), tuple(seen.failed),
            status, tuple(sorted(seen.discovered)), tuple(sorted(seen.chunks)), meta.chunk_count, chunks_complete,
            ranges, meta.char_count, text_complete, tuple(sorted(seen.excerpts)), tuple(sorted(seen.direct_spans)),
            tuple(sorted(seen.generated)),
            tuple(sorted(seen.acknowledged)), tuple(sorted(seen.used)),
        ))
    return ResearchCoverage(tuple(targets), plan.missing_requirements, tuple(outcomes), tuple(rejected))
