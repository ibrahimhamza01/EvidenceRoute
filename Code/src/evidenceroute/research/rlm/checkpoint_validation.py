"""Check durable research state against scope, canonical evidence and recorded delivery."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import fields
from math import isfinite
from typing import TYPE_CHECKING

from evidenceroute.research.rlm.budget import BudgetUsage
from evidenceroute.research.rlm.comparison import Comparison, ResolvedReferenceAssessment
from evidenceroute.research.rlm.corpus_research import CorpusUsage
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import MAX_FINDINGS, Finding, FindingEvidence
from evidenceroute.research.rlm.planning import ResearchPlan, resolve_research_plan
from evidenceroute.research.rlm.research_coverage import InvocationOutcome, _Access, _target_coverage
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import EvidenceRef
from evidenceroute.schemas import Chunk

if TYPE_CHECKING:
    from evidenceroute.research.rlm.checkpoint import CheckpointRun, ResearchCheckpoint

CHECKPOINT_VERSION = 1


class CheckpointError(ValueError):
    """A checkpoint is incompatible, malformed or internally inconsistent."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


class _CanonicalEvidence:
    def __init__(self, corpus: CorpusEnvironment):
        self.corpus = corpus
        self._chunks: dict[str, Chunk] = {}
        self._spans: dict[str, dict[str, tuple[int, int]]] = {}

    def chunk(self, chunk_id: str, scope: set[SectionTarget]) -> Chunk:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            try:
                chunk = self.corpus.get_chunk(chunk_id)
            except KeyError:
                raise CheckpointError("Checkpoint references an unknown chunk.") from None
            self._chunks[chunk_id] = chunk
        _require(SectionTarget(chunk.document_id, chunk.section_id) in scope,
                 "Checkpoint chunk lies outside its authorized scope.")
        return chunk

    def span(self, chunk: Chunk, span_id: str) -> str:
        if chunk.chunk_id not in self._spans:
            self._spans[chunk.chunk_id] = passage_spans(chunk.text)
        bounds = self._spans[chunk.chunk_id].get(span_id)
        if bounds is None:
            raise CheckpointError("Checkpoint references a stale or unknown span.")
        return chunk.text[slice(*bounds)]

    def finding(self, finding: Finding, scope: set[SectionTarget]) -> None:
        _require(
            1 <= len(finding.evidence) <= 3 and len(set(finding.evidence)) == len(finding.evidence)
            and 0 < len(finding.claim.strip()) <= 700
            and all(note is None or len(note) <= 400 for note in (finding.uncertainty, finding.conflict)),
            "Checkpoint finding has an invalid claim or evidence shape.",
        )
        targets = set()
        for item in finding.evidence:
            chunk = self.chunk(item.source.chunk_id, scope)
            _require(item.source == EvidenceRef.from_chunk(chunk) and finding.company_name == chunk.company_name,
                     "Checkpoint finding metadata differs from its canonical source.")
            valid_quote = (item.quote == self.span(chunk, item.span_id) if item.span_id is not None
                           else 0 < len(item.quote) <= 500 and bool(item.quote.strip()) and item.quote in chunk.text)
            _require(valid_quote, "Checkpoint finding excerpt differs from canonical text.")
            targets.add(SectionTarget(chunk.document_id, chunk.section_id))
        _require(len(targets) == 1, "Checkpoint finding spans multiple filing sections.")


def _validate_usage(usage: CorpusUsage | BudgetUsage) -> None:
    for field in fields(usage):
        value = getattr(usage, field.name)
        _require(value >= 0 and (not isinstance(value, float) or isfinite(value)), "Checkpoint has invalid usage counters.")
    _require(usage.finding_submission_attempts <= usage.corpus_attempts
             and usage.attempts_with_missing_usage <= usage.provider_attempts,
             "Checkpoint has inconsistent usage counters.")
    if isinstance(usage, BudgetUsage):
        _require(usage.investigation_provider_attempts + usage.finalization_provider_attempts == usage.provider_attempts,
                 "Checkpoint provider phase counts differ from total attempts.")
    else:
        _require(usage.attempts_with_missing_estimate <= usage.provider_attempts, "Checkpoint has invalid estimate counters.")


def _validate_budget(run: CheckpointRun) -> None:
    _validate_usage(run.usage)
    _require(len(run.batches) <= run.limits.max_batches, "Checkpoint exceeds its authorized batch allowance.")
    for index, batch in enumerate(run.batches, 1):
        _require(batch.number == index and bool(batch.companies), "Checkpoint batch numbering or scope is invalid.")
        _validate_usage(batch.usage)
    for field in ("provider_attempts", "reported_tokens", "corpus_attempts", "partition_assignments",
                  "delegated_invocations", "attempts_with_missing_usage", "finding_submission_attempts"):
        _require(getattr(run.usage, field) == sum(getattr(batch.usage, field) for batch in run.batches),
                 "Checkpoint batch usage differs from run usage.")
    limits = run.limits
    _require(run.usage.provider_attempts <= limits.max_provider_attempts
             and run.usage.corpus_attempts <= limits.max_corpus_attempts
             and run.usage.delegated_invocations <= limits.max_delegations
             and run.usage.partition_assignments <= limits.max_partitions
             and run.usage.lexical_chunks_scanned <= limits.max_scan_chunks
             and (limits.max_scan_pages is None or run.usage.lexical_pages_read <= limits.max_scan_pages),
             "Checkpoint usage exceeds an authorized hard limit.")


def _record_delivered(finding: Finding, access: dict[SectionTarget, _Access], used: bool) -> None:
    for item in finding.evidence:
        source = item.source
        seen = access[SectionTarget(source.document_id, source.section_id)]
        seen.acknowledged.add(finding.finding_id)
        seen.excerpts.add(source.chunk_id)
        if item.span_id is not None:
            seen.spans.add((source.chunk_id, item.span_id))
        if used:
            seen.used.add(finding.finding_id)


def _has_evidence(item: FindingEvidence, node: InvocationOutcome, excerpts: tuple[FindingEvidence, ...]) -> bool:
    chunk = item.source.chunk_id
    if chunk in node.whole_chunk_ids:
        return True
    if item.span_id is not None:
        return ((chunk, item.span_id) in node.direct_span_ids
                or any(other.source.chunk_id == chunk and other.span_id == item.span_id for other in excerpts))
    return any(other.source.chunk_id == chunk and item.quote in other.quote for other in excerpts)


def _validate_coverage(
    run: CheckpointRun, plan: ResearchPlan, canonical: _CanonicalEvidence, previous: dict[str, Finding],
) -> tuple[dict[str, InvocationOutcome], dict[str, Finding]]:
    coverage = run.coverage
    _require(tuple(row.pair for row in coverage.targets) == plan.resolved_targets
             and coverage.missing_requirements == plan.missing_requirements,
             "Checkpoint coverage differs from the original plan denominator.")
    nodes = {node.invocation.invocation_id: node for node in coverage.invocations}
    _require(len(nodes) == len(coverage.invocations), "Checkpoint contains duplicate invocation IDs.")
    _require(all(invocation_id in nodes for invocation_id, _ in coverage.rejected_delegations),
             "Checkpoint rejected-delegation history references an unknown invocation.")
    access = {pair: _Access() for pair in plan.resolved_targets}
    groups = {group.ticker: group for group in plan.groups}
    batches = {batch.number: batch for batch in run.batches}
    roots = {}
    children = defaultdict(list)
    findings = {}
    admitted: set[SectionTarget] = set()

    for invocation_id, node in nodes.items():
        inv = node.invocation
        match = re.fullmatch(rf"{run.run_id}\.batch-([1-9][0-9]{{0,8}})\.(root|child-[1-9][0-9]{{0,8}})", invocation_id)
        _require(match is not None, "Checkpoint invocation ID lies outside its run namespace.")
        batch = batches.get(int(match.group(1)))
        _require(batch is not None and batch.stop_reason != "batch_error", "Checkpoint invocation has no executed batch.")
        scope = set(inv.partitions or ())
        batch_scope = {pair for ticker in batch.companies for pair in groups[ticker].targets}
        _require(bool(scope) and len(scope) == len(inv.partitions) and scope.issubset(batch_scope),
                 "Checkpoint invocation scope differs from its authorized batch.")
        if inv.parent_id is None:
            _require(match.group(2) == "root" and inv.depth == 0 and scope == batch_scope
                     and node.stop_reason == batch.stop_reason and not node.parent_acknowledged
                     and inv.question.startswith(f"{plan.intent.question}\nObjective: {plan.intent.objective}. Topic: {plan.intent.topic}\n"),
                     "Checkpoint root does not match its scheduled batch.")
            roots[batch.number] = node
            admitted.update(scope)
        else:
            parent = nodes.get(inv.parent_id)
            _require(parent is not None and match.group(2) != "root"
                     and inv.parent_id.startswith(f"{run.run_id}.batch-{batch.number}.")
                     and inv.run_id == parent.invocation.run_id and inv.depth == parent.invocation.depth + 1
                     and scope.issubset(parent.invocation.partitions or ())
                     and (inv.depth == 1 or scope != set(parent.invocation.partitions or ())),
                     "Checkpoint child authorization is inconsistent with its parent.")
            if inv.depth == 1:
                _require(any(scope == set(groups[ticker].targets) for ticker in batch.companies),
                         "Checkpoint initial child must match an authorized whole company group.")
            children[inv.parent_id].append(node)
            for pair in scope:
                access[pair].assigned.append(invocation_id)
        if node.stop_reason != "finished":
            for pair in scope:
                access[pair].failed.append(invocation_id)
        _require(len(node.findings) <= MAX_FINDINGS, "Checkpoint invocation exceeds its finding limit.")
        for index, finding in enumerate(node.findings, 1):
            _require(finding.finding_id == f"{invocation_id}:finding-{index}" and finding.finding_id not in previous,
                     "Checkpoint finding ID is inconsistent with its originating invocation.")
            canonical.finding(finding, scope)
            findings[finding.finding_id] = finding
            pair = SectionTarget(finding.evidence[0].source.document_id, finding.evidence[0].source.section_id)
            access[pair].generated.add(finding.finding_id)

        for chunk_id in node.whole_chunk_ids:
            chunk = canonical.chunk(chunk_id, scope)
            access[SectionTarget(chunk.document_id, chunk.section_id)].chunks.add(chunk_id)
        for chunk_id, span_id in node.direct_span_ids:
            chunk = canonical.chunk(chunk_id, scope)
            canonical.span(chunk, span_id)
            seen = access[SectionTarget(chunk.document_id, chunk.section_id)]
            seen.direct_spans.add(chunk_id)
            seen.spans.add((chunk_id, span_id))
        for window in node.section_windows:
            _require(window.pair in scope, "Checkpoint coverage window lies outside its authorized scope.")
            meta = canonical.corpus.get_section(window.pair.document_id, window.pair.section_id, limit=1).metadata
            _require(0 <= window.start < window.end <= meta.char_count, "Checkpoint coverage window is out of bounds.")
            access[window.pair].ranges.append((window.start, window.end))

    _require(set(roots) == {batch.number for batch in run.batches if batch.stop_reason != "batch_error"},
             "Checkpoint is missing an executed root.")
    known = previous | findings
    for invocation_id, node in nodes.items():
        inv = node.invocation
        own = {item.finding_id for item in node.findings}
        available = set(node.acknowledged_prior_finding_ids)
        _require(available.issubset(previous), "Checkpoint references a prior finding outside earlier runs.")
        scope = set(inv.partitions)
        for fid in available:
            canonical.finding(previous[fid], scope)
            _record_delivered(previous[fid], access, fid in node.used_finding_ids)
        available.update(fid for child in children[invocation_id] for fid in child.findings_acknowledged_by_parent)
        _require(available.issubset(known) and set(node.used_finding_ids).issubset(available),
                 "Checkpoint references a finding without recorded delivery.")
        inherited = set(node.selected_inherited_finding_ids)
        _require(inherited.issubset(node.used_finding_ids) and set(node.selected_local_finding_ids).issubset(own)
                 and (node.stop_reason == "finished" or not (
                     inherited or node.selected_local_finding_ids or node.used_finding_ids or node.cited_chunk_ids)),
                 "Checkpoint terminal selection is inconsistent with its invocation.")
        expected_forwarded = inherited if inv.parent_id is not None else set()
        _require(len(node.forwarded_findings) == len(expected_forwarded)
                 and {item.finding_id for item in node.forwarded_findings} == expected_forwarded
                 and all(known[item.finding_id] == item for item in node.forwarded_findings),
                 "Checkpoint forwarded finding differs from the retained record.")
        delivered = set(node.findings_acknowledged_by_parent)
        _require(delivered.issubset(own | expected_forwarded)
                 and (not delivered or inv.parent_id is not None and node.parent_acknowledged),
                 "Checkpoint child acknowledgement is inconsistent with its retained findings.")
        if delivered:
            used = nodes[inv.parent_id].used_finding_ids
            for fid in delivered:
                _record_delivered(known[fid], access, fid in used)
        excerpts = tuple(item for fid in available for item in known[fid].evidence)
        for finding in node.findings:
            _require(all(_has_evidence(item, node, excerpts) for item in finding.evidence),
                     "Checkpoint finding lacks recorded evidence access in its originating invocation.")
        citation_ids = set(node.whole_chunk_ids) | {chunk_id for chunk_id, _ in node.direct_span_ids}
        citation_ids.update(item.source.chunk_id for fid in node.used_finding_ids for item in known[fid].evidence)
        _require(set(node.cited_chunk_ids).issubset(citation_ids),
                 "Checkpoint citation lacks recorded evidence access in its invocation.")

    for row in coverage.targets:
        for chunk_id in row.discovered_chunk_ids:
            canonical.chunk(chunk_id, {row.pair})
            access[row.pair].discovered.add(chunk_id)
    _require(coverage.targets == _target_coverage(canonical.corpus, access, admitted),
             "Checkpoint coverage summary differs from its access records.")
    return nodes, findings


def _validate_results(run: CheckpointRun, plan: ResearchPlan, canonical: _CanonicalEvidence,
                      nodes: dict[str, InvocationOutcome], known: dict[str, Finding]) -> None:
    cited = {chunk_id for node in nodes.values() if node.invocation.parent_id is None for chunk_id in node.cited_chunk_ids}
    scope = set(plan.resolved_targets)
    refs = tuple(EvidenceRef.from_chunk(canonical.chunk(chunk_id, scope)) for chunk_id in sorted(cited))
    _require(run.citations == refs, "Checkpoint citations differ from the recorded root selections.")

    years = tuple(sorted(set(plan.intent.fiscal_years)))
    selected = set(run.target_companies)
    root_selections = [set(node.selected_local_finding_ids) | set(node.selected_inherited_finding_ids)
                       for node in nodes.values() if node.invocation.depth == 0 and node.stop_reason == "finished"]
    child_selections = [set(node.findings_acknowledged_by_parent)
                        for node in nodes.values() if node.invocation.depth == 1 and node.stop_reason == "finished"]
    for comparisons, permitted in ((run.root_comparisons, root_selections),
                                    (run.acknowledged_child_comparisons, child_selections)):
        for comparison in comparisons:
            _validate_comparison(comparison, plan.intent.objective, selected, years, known, permitted)

    _validate_assignments(run, plan, nodes, selected)


def _validate_comparison(
    comparison: Comparison, objective: str, companies: set[str], years: tuple[int, ...],
    known: dict[str, Finding], permitted: list[set[str]],
) -> None:
    _require(objective == "year_over_year" and comparison.years == years
             and comparison.ticker in companies and bool(comparison.finding_ids)
             and 1 <= len(comparison.assessments) <= 8
             and len(set(comparison.finding_ids)) == len(comparison.finding_ids)
             and any(group.issuperset(comparison.finding_ids) for group in permitted),
             "Checkpoint comparison has incompatible scope or finding references.")
    sources = [item.source for fid in comparison.finding_ids for item in known[fid].evidence]
    evidence_years = tuple(sorted({source.fiscal_year for source in sources}))
    _require(all(source.ticker == comparison.ticker and source.fiscal_year in years for source in sources)
             and all(item.span_id is not None for fid in comparison.finding_ids for item in known[fid].evidence)
             and comparison.assessed_chunk_ids == tuple(sorted({source.chunk_id for source in sources}))
             and evidence_years == comparison.evidence_years
             and comparison.both_years_contributed == (evidence_years == years)
             and set(comparison.ai_specific_years + comparison.ai_contextual_years).issubset(evidence_years),
             "Checkpoint comparison evidence differs from its retained findings.")
    for item in comparison.assessments:
        if isinstance(item, ResolvedReferenceAssessment):
            _require(item.finding_id in comparison.finding_ids and (
                item.context_finding_id is None or item.context_finding_id in comparison.finding_ids),
                "Checkpoint comparison assessment references an unknown finding.")


def _validate_assignments(run: CheckpointRun, plan: ResearchPlan,
                          nodes: dict[str, InvocationOutcome], selected: set[str]) -> None:
    _require(tuple(group.ticker for group in run.groups) == plan.requested_companies,
             "Checkpoint assignment history differs from the original plan.")
    assigned = {ticker: batch.number for batch in run.batches for ticker in batch.companies}
    scopes = {group.ticker: set(group.targets) for group in plan.groups}
    targets = {row.pair: row for row in run.coverage.targets}
    for group in run.groups:
        if group.ticker not in selected:
            _require(group.pending_reason == "not_selected" and group.outcome is None and group.batch_number is None,
                     "Checkpoint records work for an unselected assignment.")
            continue
        outcome = group.outcome
        if outcome is None:
            _require(group.pending_reason is not None, "Checkpoint marks an unattempted assignment as complete.")
            continue
        tick_scope = scopes.get(group.ticker, set())
        root_id = f"{run.run_id}.batch-{group.batch_number}.root"
        expected_children = tuple(node for node in nodes.values()
                                  if node.invocation.depth == 1 and set(node.invocation.partitions) == tick_scope)
        _require(root_id in nodes and outcome.ticker == group.ticker and group.batch_number == assigned.get(group.ticker)
                 and outcome.root_delegate_ids == tuple(node.invocation.invocation_id for node in expected_children)
                 and outcome.findings_acknowledged_by_root == tuple(fid for node in expected_children
                                                                   for fid in node.findings_acknowledged_by_parent),
                 "Checkpoint assignment does not match its recorded invocations.")
        first = expected_children[0] if expected_children else None
        acknowledged = [node for node in expected_children if node.parent_acknowledged
                        and node.delegation_status in {"finished", "incomplete"}
                        and (not (node.findings or node.forwarded_findings) or node.findings_acknowledged_by_parent)]
        status = ("completed" if first.stop_reason == "finished" else "failed") if first else "unassigned"
        rows = [targets[pair] for pair in tick_scope]
        failed = {entry for row in rows for entry in row.failed_invocation_ids}
        access_status = ("complete" if rows and all(row.status == "complete" for row in rows)
                         else "partial" if any(row.status != "none" for row in rows) else "none")
        _require(outcome.assignment_status == status and outcome.admitted
                 and outcome.access_status == access_status
                 and outcome.initial_invocation_id == (first.invocation.invocation_id if first else None)
                 and outcome.initial_result_acknowledged == (first in acknowledged)
                 and outcome.result_acknowledged == bool(acknowledged)
                 and set(outcome.failed_invocation_ids) == failed,
                 "Checkpoint assignment status differs from its delivery records.")
        if status == "failed" and not acknowledged:
            pending = "child_failed"
        elif not acknowledged:
            pending = outcome.assignment_pending_reason or "child_result_unacknowledged"
        else:
            pending = None if nodes[root_id].stop_reason == "finished" else "batch_root_unfinished"
        _require(group.pending_reason == pending, "Checkpoint pending assignment status differs from execution.")
    expected = "finished" if run.batches and all(group.pending_reason is None for group in run.groups) else "incomplete"
    _require(run.execution_status == expected and (
        run.research_status == "incomplete" or expected == "finished"
        and all(row.status == "complete" for row in run.coverage.targets)),
        "Checkpoint marks incomplete coverage or execution as complete.")


def validate_checkpoint(checkpoint: ResearchCheckpoint, corpus: CorpusEnvironment) -> None:
    _require(type(checkpoint.version) is int and checkpoint.version == CHECKPOINT_VERSION,
             "Unsupported checkpoint schema version.")
    _require(checkpoint.corpus_identity == corpus.snapshot_identity, "Checkpoint corpus identity does not match the loaded snapshot.")
    plan = checkpoint.plan
    _require(plan == resolve_research_plan(plan.intent, corpus), "Checkpoint plan differs from its canonical original scope.")
    _require(not (checkpoint.discovery and checkpoint.scan_terms), "Checkpoint has conflicting discovery policies.")
    if checkpoint.scan_terms or checkpoint.discovery and plan.intent.objective == "year_over_year":
        _require(plan.intent.objective == "year_over_year" and len(set(plan.intent.fiscal_years)) == 2,
                 "Checkpoint comparative discovery policy requires a two-year plan.")
    if checkpoint.interpretation is not None:
        usage = checkpoint.interpretation.usage
        _validate_usage(usage)
        _require(usage.provider_attempts <= checkpoint.interpretation.limits.max_provider_attempts
                 and checkpoint.interpretation.charged_tokens >= usage.reported_tokens
                 and checkpoint.runs[0].limits.max_provider_attempts + usage.provider_attempts
                 == checkpoint.interpretation.overall_limits.max_provider_attempts
                 and checkpoint.runs[0].limits.max_reported_tokens + checkpoint.interpretation.charged_tokens
                 == checkpoint.interpretation.overall_limits.max_reported_tokens,
                 "Checkpoint interpretation usage does not settle against its authorized limits.")
    canonical = _CanonicalEvidence(corpus)
    findings: dict[str, Finding] = {}
    seen_runs = set()
    complete_groups = {group.ticker for group in plan.groups if group.targets} - {row.company for row in plan.missing_requirements}
    for index, run in enumerate(checkpoint.runs):
        _require(run.run_id not in seen_runs, "Checkpoint contains duplicate run IDs.")
        seen_runs.add(run.run_id)
        selected = set(run.target_companies)
        _require(tuple(sorted(selected)) == run.target_companies and (
            selected == set(plan.requested_companies) if index == 0 else bool(selected) and selected.issubset(complete_groups)),
            "Checkpoint continuation selection lies outside its authorized scope.")
        scheduled = [ticker for batch in run.batches for ticker in batch.companies]
        _require(len(set(scheduled)) == len(scheduled) and set(scheduled).issubset(selected & complete_groups),
                 "Checkpoint batch scope exceeds the selected assignments.")
        _validate_budget(run)
        nodes, new = _validate_coverage(run, plan, canonical, findings)
        findings.update(new)
        _validate_results(run, plan, canonical, nodes, findings)
