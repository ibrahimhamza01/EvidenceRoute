"""Sequential model-directed delegation with scoped tools and shared limits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from evidenceroute.research.rlm.budget import BudgetExceeded, FinalizationPolicy, SharedBudget, SharedLimits
from evidenceroute.research.rlm.comparison import Comparison, CorpusScan, resolve_comparison
from evidenceroute.research.rlm.controller import (
    ExecutionTrace, RootController, RootLimits, RootModelClient, ToolTrace, _FinishAction,
)
from evidenceroute.research.rlm.coverage import CoverageReport, ResearchScope, SectionTarget
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import (
    AnalysisStatus, Finding, FindingEvidence, PartitionCoverage, SpanEvidenceDraft, partition_status,
)
from evidenceroute.research.rlm.scoped import ScopedCorpus
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import (
    DispatchResult, EvidenceRef, Identifier, StrictInput, error_observation, observation_json, validation_message,
)


class _Partition(StrictInput):
    document_id: Identifier
    section_id: Identifier


class _Delegate(StrictInput):
    question: str = Field(min_length=1, max_length=1_500)
    goal: str = Field(min_length=1, max_length=750)
    partitions: list[_Partition] = Field(min_length=1, max_length=500)


@dataclass(frozen=True)
class Invocation:
    run_id: str
    invocation_id: str
    parent_id: str | None
    depth: int
    question: str
    goal: str | None
    partitions: tuple[SectionTarget, ...] | None


@dataclass
class DelegationTrace:
    status: Literal["rejected", "failed", "incomplete", "finished", "oversized"]
    error: str | None = None
    invocation: Invocation | None = None
    child: ExecutionTrace | None = None
    delivered_finding_ids: tuple[str, ...] = ()


class RecursiveResearch:
    """Start a run with a fresh client per invocation and a single shared budget."""

    def __init__(
        self, corpus: CorpusEnvironment, model_factory: Callable[[Invocation], RootModelClient], *,
        shared_limits: SharedLimits | None = None, local_limits: RootLimits | None = None,
        finalization: FinalizationPolicy | None = None,
        child_guidance: str | None = None,
        comparison_scan: CorpusScan | None = None,
        clock: Callable[[], float] = monotonic,
    ):
        self.corpus = corpus
        self.model_factory = model_factory
        self.shared_limits = shared_limits if shared_limits is not None else SharedLimits()
        self.local_limits = local_limits if local_limits is not None else RootLimits()
        self.clock = clock
        self.finalization = finalization
        self.child_guidance = child_guidance
        self.comparison_scan = comparison_scan

    def run(
        self, question: str, *, partitions: tuple[SectionTarget, ...] | None = None,
        run_id: str | None = None,
        root_delegation_groups: tuple[tuple[SectionTarget, ...], ...] | None = None,
    ) -> ExecutionTrace:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question must be a non-blank string.")
        budget = SharedBudget(self.shared_limits, self.clock, finalization=self.finalization)
        if partitions is not None:
            partitions = tuple(partitions)
            if not partitions or len(partitions) > self.shared_limits.max_partitions:
                raise ValueError("Root authorization must fit the partition limit and be nonempty.")
            self._validate_partitions(partitions)
        groups = None
        if root_delegation_groups is not None:
            groups = frozenset(frozenset(group) for group in root_delegation_groups)
            if not groups or any(not group or not group.issubset(partitions or ()) for group in groups):
                raise ValueError("Root delegation groups must be nonempty subsets of authorized pairs.")
        invocation = Invocation(run_id or uuid4().hex, "root", None, 0, question, None, partitions)
        return self._invoke(invocation, budget, [], budget.deadline, groups)

    def _validate_partitions(self, partitions: tuple[SectionTarget, ...]) -> None:
        if len(set(partitions)) != len(partitions):
            raise ValueError("Partition pairs must be unique.")
        for pair in partitions:
            self.corpus.get_section(pair.document_id, pair.section_id, limit=1)

    def _invoke(
        self, invocation: Invocation, budget: SharedBudget, clients: list[RootModelClient],
        parent_deadline: float,
        root_groups: frozenset[frozenset[SectionTarget]] | None = None,
    ) -> ExecutionTrace:
        with budget.invocation_allocation(root=invocation.depth == 0):
            return self._run_invocation(invocation, budget, clients, parent_deadline, root_groups)

    def _run_invocation(
        self, invocation: Invocation, budget: SharedBudget, clients: list[RootModelClient],
        parent_deadline: float,
        root_groups: frozenset[frozenset[SectionTarget]] | None,
    ) -> ExecutionTrace:
        model = self.model_factory(invocation)
        bind = getattr(model, "bind_execution_budget", None)
        if any(model is previous for previous in clients) or not callable(bind):
            raise ValueError("The factory must return a fresh, budget-aware model client.")
        bind(budget)
        if self.comparison_scan is not None:
            enable = getattr(model, "enable_comparative_finish", None)
            if callable(enable):
                enable()
        clients.append(model)
        budget.model_sessions += 1
        session = RecursiveSession(self, invocation, budget, clients, parent_deadline, root_groups)
        partitions = invocation.partitions
        view = self.corpus if partitions is None else ScopedCorpus(self.corpus, partitions)
        scope = ResearchScope() if partitions is None else ResearchScope(
            tuple(sorted({pair.document_id for pair in partitions})), partitions,
        )
        prompt = invocation.question
        if invocation.goal is not None:
            prompt += f"\nAnalysis goal: {invocation.goal}"
        if invocation.depth > 0 and self.child_guidance:
            prompt += f"\nInvestigation guidance: {self.child_guidance}"
        if partitions is not None:
            prompt += "\nAuthorized partitions: " + observation_json([vars(pair) for pair in partitions])
        if self.comparison_scan is not None:
            shortlist = self.comparison_scan.shortlist(partitions or ())
            prompt += (
                "\nLexical candidates (chunk IDs only, at most three per section): " + observation_json(shortlist)
                + "\nRead candidates and surrounding context as needed. Choose chunk_id and span_id from acknowledged "
                "evidence to create findings. To compare the same risk, reference your own findings by one-based index "
                "in comparisons. Evaluate relevance to the topic separately from what the passage supports or contradicts. "
                "Inspect possible counterevidence in the earlier year. If unclear, use remaining scoped tool calls to "
                "investigate that specific uncertainty, then report any unresolved conflict or gap. No lexical miss proves absence."
            )
        if invocation.depth < budget.limits.max_depth:
            prompt += "\nUse delegate_research only for a narrower subproblem."
        evidence_instruction = ("source span IDs" if self.comparison_scan is not None else "exact quotes")
        prompt += (
            f"\nSubmit findings via finish_research with {evidence_instruction} and original chunk IDs. "
            "Report uncertainty, conflicts and incomplete analysis. "
            "Name used_finding_ids when synthesizing delegated evidence. Search misses do not establish absence."
        )
        return RootController(view, model, limits=self.local_limits, clock=self.clock,
                              include_spans=self.comparison_scan is not None).run(
            prompt, scope=scope, _recursive=session,
        )


class RecursiveSession:
    def __init__(
        self, runner: RecursiveResearch, invocation: Invocation, budget: SharedBudget,
        clients: list[RootModelClient], parent_deadline: float,
        root_groups: frozenset[frozenset[SectionTarget]] | None,
    ):
        self.runner = runner
        self.invocation = invocation
        self.budget = budget
        self.clients = clients
        self.parent_deadline = parent_deadline
        self.root_groups = root_groups
        self.direct_ids: set[str] = set()
        self.direct_spans: dict[str, set[str]] = {}
        self.delivered_findings: dict[str, Finding] = {}
        self.findings: tuple[Finding, ...] = ()
        self.comparisons: tuple[Comparison, ...] = ()
        self.used_finding_ids: tuple[str, ...] = ()
        self.analysis_status: AnalysisStatus = "incomplete"

    def can_delegate(self) -> bool:
        if self.invocation.depth and len(self.invocation.partitions or ()) <= 1:
            return False
        try:
            self.budget.check_child(self.invocation.depth + 1, 1)
        except BudgetExceeded:
            return False
        return True

    def tool_schema(self) -> dict:
        return {
            "name": "delegate_research",
            "description": (
                "Investigate a narrower question in a fresh conversation with exact document and section pairs. "
                f"Current depth {self.invocation.depth}, maximum {self.budget.limits.max_depth}. "
                "All delegates share the run budget. Returned findings include original excerpts and coverage."
            ),
            "input_schema": _Delegate.model_json_schema(),
        }

    def delegate(self, arguments: dict, deadline: float) -> tuple[DispatchResult, DelegationTrace]:
        cap = self.runner.local_limits.max_response_bytes
        validated = None
        invocation = None

        def reject(code: str, message: str, *, status: Literal["rejected", "failed"] = "rejected"):
            return (DispatchResult(validated, error_observation(code, message, cap)),
                    DelegationTrace(status, code, invocation=invocation))

        try:
            request = _Delegate.model_validate(arguments)
        except ValidationError as exc:
            return reject("invalid_arguments", validation_message(exc))
        validated = request.model_dump(mode="json")
        pairs = tuple(SectionTarget(p.document_id, p.section_id) for p in request.partitions)
        parent_pairs = self.invocation.partitions
        depth = self.invocation.depth + 1
        try:
            self.budget.check_child(depth, len(pairs))
            if parent_pairs is not None and not set(pairs).issubset(parent_pairs):
                return reject("scope_violation", "Delegate only authorized document and section pairs.")
            if self.root_groups is not None and frozenset(pairs) not in self.root_groups:
                return reject("group_violation", "Delegate one whole admitted company group at a time.")
            if self.invocation.depth > 0 and set(pairs) == set(parent_pairs or ()):
                return reject("scope_violation", "A descendant must delegate a strict subset.")
            self.runner._validate_partitions(pairs)
            if self.runner.clock() >= deadline:
                raise BudgetExceeded("deadline")
            number = self.budget.reserve_child(depth, len(pairs))
        except BudgetExceeded as exc:
            return reject(str(exc), "Shared budget does not permit this delegation.")
        except (KeyError, ValueError):
            return reject("invalid_partitions", "Use unique canonical document and section pairs.")

        invocation = Invocation(self.invocation.run_id, f"child-{number}", self.invocation.invocation_id,
                                depth, request.question, request.goal, pairs)
        try:
            child = self.runner._invoke(invocation, self.budget, self.clients, deadline)
        except Exception:
            return reject("child_execution_error", "Child invocation could not run.", status="failed")
        successful = child.stop_reason == "finished"
        complete = (
            successful and bool(child.findings) and child.analysis_status == "analyzed"
            and all(p.status == "complete" for p in child.partition_coverage)
            and all(finding.analysis_status == "analyzed" for finding in child.findings)
        )
        status = ("finished" if complete else "incomplete") if successful else "failed"
        findings = child.findings if successful else ()
        body = {
            "ok": successful, "result": {
                "child_invocation_id": invocation.invocation_id, "status": status,
                "stop_reason": child.stop_reason, "budget_stop": child.budget_stop,
                "analysis_status": child.analysis_status, "coverage": child.partition_coverage,
                "findings": findings,
            },
        }
        if self.runner.comparison_scan is not None and successful:
            body["result"]["comparisons"] = child.comparisons
        if not successful:
            body["error"] = {"code": "child_not_finished", "message": "Inspect the child stop reason and coverage."}
        content = observation_json(body)
        delegation = DelegationTrace(status, invocation=invocation, child=child)
        if len(content.encode("utf-8")) > cap:
            delegation.status = "oversized"
            delegation.error = "response_too_large"
            content = error_observation(
                "response_too_large", "Child findings exceeded the cap. Try a smaller delegated task.",
                cap, max_response_bytes=cap, response_bytes=len(content.encode("utf-8")),
            )
            return DispatchResult(validated, content), delegation
        refs = tuple(dict.fromkeys(item.source for finding in findings for item in finding.evidence))
        return DispatchResult(validated, content, refs), delegation

    def record_delivery(self, step: ToolTrace) -> None:
        if step.delegation is None:
            self.direct_ids.update(ref.chunk_id for ref in step.evidence)
            for chunk_id, spans in (step.observation or {}).get("span_ids", {}).items():
                self.direct_spans.setdefault(chunk_id, set()).update(spans)
        elif step.delegation.child is not None and step.delegation.status in {"finished", "incomplete"}:
            findings = step.delegation.child.findings
            self.delivered_findings.update((finding.finding_id, finding) for finding in findings)
            step.delegation.delivered_finding_ids = tuple(finding.finding_id for finding in findings)

    def finish(self, action: _FinishAction, coverage: CoverageReport) -> str | None:
        scan = self.runner.comparison_scan
        if action.comparisons and scan is None:
            return "Comparisons require comparative mode."
        used = tuple(dict.fromkeys(action.used_finding_ids))
        if any(finding_id not in self.delivered_findings for finding_id in used):
            return "Use only delivered finding IDs."
        used_evidence = [item for finding_id in used for item in self.delivered_findings[finding_id].evidence]
        selected_ids = {item.source.chunk_id for item in used_evidence}
        cited = set(action.cited_chunk_ids)
        if not cited.issubset(self.direct_ids | selected_ids):
            return "Declare the delivered finding IDs used by delegated citations."

        findings = []
        for index, draft in enumerate(action.findings, start=1):
            evidence = []
            for evidence_index, item in enumerate(draft.evidence, start=1):
                location = f"Finding {index}, evidence {evidence_index}"
                if isinstance(item, SpanEvidenceDraft):
                    if scan is None:
                        return f"{location}: span selection requires comparative mode."
                    inherited = next((ref for ref in used_evidence if ref.source.chunk_id == item.chunk_id
                                      and ref.span_id == item.span_id), None)
                    if item.span_id in self.direct_spans.get(item.chunk_id, ()):
                        chunk = self.runner.corpus.get_chunk(item.chunk_id)
                        start, end = passage_spans(chunk.text)[item.span_id]
                        evidence.append(FindingEvidence(EvidenceRef.from_chunk(chunk), chunk.text[start:end], item.span_id))
                    elif inherited is not None:
                        evidence.append(inherited)
                    else:
                        return f"{location}: select a span from an acknowledged chunk or a declared delivered finding."
                    continue
                if item.chunk_id not in self.direct_ids:
                    if item.chunk_id not in selected_ids:
                        return f"{location}: use an observed chunk or a declared delivered excerpt, or omit it."
                    if not any(item.chunk_id == observed.source.chunk_id and item.quote in observed.quote
                               for observed in used_evidence):
                        return (f"{location} (chunk {item.chunk_id}): copy a contiguous quote exactly from "
                                "the delivered excerpt or omit it.")
                chunk = self.runner.corpus.get_chunk(item.chunk_id)
                if item.quote not in chunk.text:
                    return (f"{location} (chunk {item.chunk_id}): copy a contiguous quote exactly from "
                            "the delivered chunk text, including punctuation and spacing, or omit it.")
                evidence.append(FindingEvidence(EvidenceRef.from_chunk(chunk), item.quote))
            source = evidence[0].source
            target = SectionTarget(source.document_id, source.section_id)
            if any((item.source.document_id, item.source.section_id) != (target.document_id, target.section_id)
                   for item in evidence):
                return "Each finding must cite evidence from one document and section."
            findings.append(Finding(
                f"{self.invocation.invocation_id}:finding-{index}", self.runner.corpus.get_filing(source.document_id).company_name,
                draft.claim, tuple(evidence),
                draft.analysis_status, partition_status(coverage, target), draft.uncertainty, draft.conflict,
            ))
        referenced = cited | {item.source.chunk_id for finding in findings for item in finding.evidence}
        if any(not any(item.source.chunk_id in referenced for item in self.delivered_findings[fid].evidence)
               for fid in used):
            return "Each used finding must contribute a chunk to the answer or a new finding."
        if scan is not None:
            try:
                self.comparisons = tuple(resolve_comparison(draft, findings, scan.years) for draft in action.comparisons)
            except ValueError as exc:
                return str(exc)
        self.findings = tuple(findings)
        self.used_finding_ids = used
        self.analysis_status = (action.analysis_status or "incomplete") if referenced else "incomplete"
        return None

    def trace_fields(self, coverage: CoverageReport) -> dict:
        partitions = tuple(PartitionCoverage(pair, partition_status(coverage, pair))
                           for pair in self.invocation.partitions or ())
        if self.invocation.partitions is None:
            coverage_status = "unknown"
        elif all(pair.status == "complete" for pair in partitions):
            coverage_status = "complete"
        elif any(pair.status != "missing" for pair in partitions):
            coverage_status = "partial"
        else:
            coverage_status = "missing"
        return {
            "invocation": self.invocation,
            "shared_limits": self.budget.limits.model_dump(mode="json"),
            "shared_budget": self.budget.snapshot(),
            "findings": self.findings,
            "comparisons": self.comparisons,
            "used_finding_ids": self.used_finding_ids,
            "analysis_status": self.analysis_status,
            "partition_coverage": partitions,
            "coverage_status": coverage_status,
        }
