"""Roll up acknowledged access across a run without widening citation eligibility."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from evidenceroute.research.rlm.controller import ExecutionTrace
from evidenceroute.research.rlm.coverage import SectionTarget, _merge_ranges
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import AnalysisStatus, Finding, FindingValidationIssue
from evidenceroute.research.rlm.planning import MissingRequirement, ResearchPlan
from evidenceroute.research.rlm.recursive import DelegationTrace, Invocation
from evidenceroute.research.rlm.synthesis import Conclusion
from evidenceroute.research.rlm.tools import READ_PRIOR_FINDINGS

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
    selected_inherited_finding_ids: tuple[str, ...] = ()
    selected_local_finding_ids: tuple[str, ...] = ()
    forwarded_findings: tuple[Finding, ...] = ()
    direct_span_ids: tuple[tuple[str, str], ...] = ()
    acknowledged_prior_finding_ids: tuple[str, ...] = ()
    acknowledged_local_finding_ids: tuple[str, ...] = ()
    conclusions: tuple[Conclusion, ...] = ()
    withheld_conclusions: bool = False
    conclusion_rejections: tuple[FindingValidationIssue, ...] = ()

    @property
    def acknowledged_findings(self) -> tuple[Finding, ...]:
        delivered = set(self.findings_acknowledged_by_parent)
        return tuple(finding for finding in (*self.findings, *self.forwarded_findings)
                     if finding.finding_id in delivered)


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
    delivered_span_ids: tuple[tuple[str, str], ...] = ()


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
    spans: set[tuple[str, str]] = field(default_factory=set)


def aggregate_research_coverage(
    corpus: CorpusEnvironment, plan: ResearchPlan, trace: ExecutionTrace | None,
) -> ResearchCoverage:
    return aggregate_research_trees(corpus, plan, () if trace is None else (trace,))


def aggregate_research_trees(
    corpus: CorpusEnvironment, plan: ResearchPlan, traces: Iterable[ExecutionTrace],
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

        generated = {finding.finding_id for finding in findings}
        for finding in run.transferable_findings if run else ():
            for item in finding.evidence:
                source = item.source
                target = access.get(SectionTarget(source.document_id, source.section_id))
                if target is None:
                    continue
                if finding.finding_id in generated:
                    target.generated.add(finding.finding_id)
                if finding.finding_id in delivered:
                    target.acknowledged.add(finding.finding_id)
                    target.excerpts.add(source.chunk_id)
                    if item.span_id is not None:
                        target.spans.add((source.chunk_id, item.span_id))
                    if finding.finding_id in parent_used:
                        target.used.add(finding.finding_id)

        chunks: set[str] = set()
        spans: set[tuple[str, str]] = set()
        prior_ids: set[str] = set()
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
                            delivered_spans = {(ref.chunk_id, span) for span in observation["span_ids"][ref.chunk_id]}
                            target.spans.update(delivered_spans)
                            spans.update(delivered_spans)
                elif operation.tool_name == READ_PRIOR_FINDINGS:
                    prior_ids.update(operation.acknowledged_prior_finding_ids)
                    for record in observation["result"]["findings"]:
                        fid = record["finding_id"]
                        for item in record["evidence"]:
                            source = item["source"]
                            target = access[SectionTarget(source["document_id"], source["section_id"])]
                            target.acknowledged.add(fid)
                            target.excerpts.add(source["chunk_id"])
                            if item["span_id"] is not None:
                                target.spans.add((source["chunk_id"], item["span_id"]))
                            if fid in run.used_finding_ids:
                                target.used.add(fid)

        outcomes.append(InvocationOutcome(
            invocation, run.stop_reason if run else None, run.budget_stop if run else None,
            run.analysis_status if run else None, edge.status if edge else None, edge.error if edge else None,
            acknowledged, findings, delivered, run.used_finding_ids if run else (),
            run.cited_chunk_ids if run else (), tuple(sorted(chunks)), tuple(windows),
            tuple(finding.finding_id for finding in run.selected_inherited_findings) if run else (),
            run.selected_local_finding_ids if run else (),
            run.selected_inherited_findings if run and invocation.parent_id is not None else (),
            tuple(sorted(spans)), tuple(sorted(prior_ids)),
            tuple(dict.fromkeys(f"{invocation.invocation_id}:finding-{ref[1:]}"
                                for step in run.steps for op in step.operations for ref in op.acknowledged_finding_refs))
            if run and run.conclusions else (),
            run.conclusions if run else (), run.withheld_conclusions if run else False,
            tuple(issue for step in run.steps for issue in step.finding_validation
                  if issue.path.startswith("synthesis.conclusions")) if run and run.withheld_conclusions else (),
        ))
        for child, child_acknowledged in children:
            assert child.invocation is not None
            visit(child.invocation, child.child, child, child_acknowledged, run.used_finding_ids)

    admitted: set[SectionTarget] = set()
    for trace in traces:
        if trace.invocation is None or trace.invocation.partitions is None:
            raise ValueError("Research coverage requires an explicitly scoped recursive trace.")
        if not set(trace.invocation.partitions).issubset(access):
            raise ValueError("The recursive trace includes targets outside the research plan.")
        admitted.update(trace.invocation.partitions)
        visit(trace.invocation, trace)

    return ResearchCoverage(_target_coverage(corpus, access, admitted), plan.missing_requirements,
                            tuple(outcomes), tuple(rejected))


def _target_coverage(corpus: CorpusEnvironment, access: dict[SectionTarget, _Access],
                     admitted: set[SectionTarget]) -> tuple[TargetCoverage, ...]:
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
            tuple(sorted(seen.acknowledged)), tuple(sorted(seen.used)), tuple(sorted(seen.spans)),
        ))
    return tuple(targets)


def merge_research_coverage(
    corpus: CorpusEnvironment, plan: ResearchPlan, reports: Iterable[ResearchCoverage],
) -> ResearchCoverage:
    """Union access over the original filing-section denominator."""
    access = {pair: _Access() for pair in plan.resolved_targets}
    admitted: set[SectionTarget] = set()
    invocations: dict[str, InvocationOutcome] = {}
    rejected = []
    for report in reports:
        for node in report.invocations:
            invocation_id = node.invocation.invocation_id
            if invocation_id in invocations:
                raise ValueError("Duplicate invocation in coverage history.")
            invocations[invocation_id] = node
        rejected.extend(report.rejected_delegations)
        for row in report.targets:
            seen = access[row.pair]
            if row.admitted:
                admitted.add(row.pair)
            seen.chunks.update(row.whole_chunk_ids)
            seen.discovered.update(row.discovered_chunk_ids)
            seen.ranges.extend(row.character_ranges)
            seen.excerpts.update(row.delegated_excerpt_chunk_ids)
            seen.direct_spans.update(row.direct_span_chunk_ids)
            seen.assigned.extend(row.assigned_child_ids)
            seen.failed.extend(row.failed_invocation_ids)
            seen.generated.update(row.generated_finding_ids)
            seen.acknowledged.update(row.acknowledged_finding_ids)
            seen.used.update(row.used_finding_ids)
            seen.spans.update(row.delivered_span_ids)
    return ResearchCoverage(_target_coverage(corpus, access, admitted), plan.missing_requirements,
                            tuple(invocations.values()), tuple(rejected))
