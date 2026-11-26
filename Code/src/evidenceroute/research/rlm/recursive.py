"""Sequential model-directed delegation with scoped tools and shared limits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationError

from evidenceroute.research.rlm.budget import BudgetExceeded, FinalizationPolicy, SharedBudget, SharedLimits
from evidenceroute.research.rlm.comparison import Comparison, render_comparisons, resolve_comparison, resolve_reference_comparison
from evidenceroute.research.rlm.discovery import CandidatePageRequest, CorpusScan
from evidenceroute.research.rlm.controller import (
    ExecutionTrace, RootController, RootLimits, RootModelClient, ToolTrace, _FinishAction, _ReferenceSynthesis,
)
from evidenceroute.research.rlm.coverage import CoverageReport, ResearchScope, SectionTarget
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import (
    MAX_FINDINGS, MAX_FINDINGS_PER_SUBMISSION, AnalysisStatus, EvidenceDraft, Finding, FindingDraft, FindingEvidence,
    FindingValidationError, FindingValidationIssue, PartitionCoverage, SpanEvidenceDraft, SpanFindingDraft,
    SubmitFindings, finding_schema_issues, partition_status, render_findings,
)
from evidenceroute.research.rlm.scoped import ScopedCorpus
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import (
    SUBMIT_FINDINGS, DispatchResult, EvidenceRef, Identifier, StrictInput, error_observation, observation_json,
    validation_message,
)


class _Partition(StrictInput):
    document_id: Identifier
    section_id: Identifier


class _Delegate(StrictInput):
    question: str = Field(min_length=1, max_length=1_500)
    goal: str = Field(min_length=1, max_length=750)
    partitions: list[_Partition] = Field(min_length=1, max_length=500)


FindingKey = tuple[str, frozenset[FindingEvidence], AnalysisStatus, str | None, str | None]


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
    initial_assignment: bool = False
    offered_finding_ids: tuple[str, ...] = ()
    offered_comparison_indices: tuple[int, ...] = ()
    full_observation_bytes: int | None = None
    finding_references: tuple[tuple[str, str], ...] = ()


class RecursiveResearch:
    """Start a run with a fresh client per invocation and a single shared budget."""

    def __init__(
        self, corpus: CorpusEnvironment, model_factory: Callable[[Invocation], RootModelClient], *,
        shared_limits: SharedLimits | None = None, local_limits: RootLimits | None = None,
        finalization: FinalizationPolicy | None = None,
        child_guidance: str | None = None,
        comparison_scan: CorpusScan | None = None,
        discovery_scan: CorpusScan | None = None,
        max_root_reinvestigation_calls: int | None = None,
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
        self.discovery_scan = discovery_scan
        if max_root_reinvestigation_calls is not None and (
            type(max_root_reinvestigation_calls) is not int or max_root_reinvestigation_calls < 0
        ):
            raise ValueError("Root reinvestigation call limit must be a nonnegative integer.")
        self.max_root_reinvestigation_calls = max_root_reinvestigation_calls

    def run(
        self, question: str, *, partitions: tuple[SectionTarget, ...] | None = None,
        run_id: str | None = None,
        root_delegation_groups: tuple[tuple[SectionTarget, ...], ...] | None = None,
        delegation_first: bool = False,
        _budget: SharedBudget | None = None,
        _id_prefix: str = "",
    ) -> ExecutionTrace:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question must be a non-blank string.")
        budget = _budget if _budget is not None else SharedBudget(self.shared_limits, self.clock, finalization=self.finalization)
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
        if delegation_first:
            if (not groups or len(groups) != len(root_delegation_groups)
                    or any(len(group) != len(set(group)) for group in root_delegation_groups)
                    or sum(map(len, groups)) != len(partitions or ()) or set().union(*groups) != set(partitions or ())):
                raise ValueError("Delegation-first research requires disjoint whole groups covering the root scope.")
            if len(groups) > self.shared_limits.max_delegations or self.shared_limits.max_depth < 1:
                raise ValueError("Initial groups must fit the shared delegation and depth limits.")
            budget.initial_delegations_left = len(groups)
            budget.initial_partitions_left = len(partitions)
        invocation = Invocation(run_id or uuid4().hex, f"{_id_prefix}root", None, 0, question, None, partitions)
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
        session = RecursiveSession(self, invocation, budget, clients, parent_deadline, root_groups)
        if self.comparison_scan is not None:
            enable = getattr(model, "enable_comparative_finish", None)
            if callable(enable):
                enable()
        clients.append(model)
        budget.model_sessions += 1
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
        if self.discovery_scan is not None and not session.delegation_first:
            shortlist = self.discovery_scan.shortlist(partitions or ())
            prompt += (
                "\nLexical candidates (IDs, at most three per section): " + observation_json(shortlist)
                + "\nUse list_candidates for scoped IDs and get_spans or get_chunk to read promising evidence. "
                "Use scoped search_chunks for gaps or different wording. No lexical miss proves absence."
            )
        if self.comparison_scan is not None and not session.delegation_first:
            shortlist = self.comparison_scan.shortlist(partitions or ())
            prompt += (
                "\nLexical candidates (IDs and span hints, at most three per section): " + observation_json(shortlist)
                + "\nUse get_spans for matching passages and list_candidates with next_offset for more scoped IDs. "
                "Use scoped search_chunks for gaps or different wording, including sections with no candidates. "
                "Read promising evidence in each fiscal "
                "year and check the same risk in both filings. Choose chunk_id and span_id from acknowledged "
                "evidence to create findings. Evaluate relevance to the topic separately from what the passage supports or contradicts. "
                "Accelerator demand alone does not establish an infrastructure risk. Explain the connection in the "
                "assessment and state separately whether the exact source links it to AI infrastructure. "
                "Mark relevance other or uncertain where appropriate and abstain if support is insufficient. "
                "Investigate counterevidence and report gaps. No lexical miss proves absence."
            )
        if session.delegation_first:
            prompt += ("\nStart with one delegate_research call per company, keeping all its fiscal years together. "
                       "Once every admitted group has an attempt, synthesize delivered results or investigate specific gaps.")
        elif invocation.depth < budget.limits.max_depth:
            prompt += "\nUse delegate_research only for a narrower subproblem."
        prompt += (
            "\nUse submit_findings as soon as you have supported claims from acknowledged chunk and span IDs. "
            "Keep evidence for each finding in one document and section. In reference synthesis, select the returned sN "
            "references and dN references from delivered child findings. The host derives citations and renders the answer. "
            "Report uncertainty and conflicts. Abstain when evidence is insufficient. Search misses do not establish absence."
        )
        if self.comparison_scan is not None:
            prompt += (" Reassess short references in synthesis.comparisons. In a legacy finish, use one-based "
                       "finding indices for comparisons.")
        return RootController(view, model, limits=self.local_limits, clock=self.clock,
                              include_spans=True, comparative=self.comparison_scan is not None).run(
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
        self.delegation_first = invocation.depth == 0 and budget.initial_delegations_left > 0
        self.unassigned_groups = set(root_groups or ()) if self.delegation_first else set()
        self.direct_ids: set[str] = set()
        self.direct_spans: dict[str, set[str]] = {}
        self.delivered_findings: dict[str, Finding] = {}
        self.reference_synthesis = runner.discovery_scan is not None or (
            invocation.depth == 0 and runner.comparison_scan is not None
        )
        self.acknowledged_references: dict[str, Finding] = {}
        self.submitted_findings: dict[str, Finding] = {}
        self.retained_findings: dict[FindingKey, Finding] = {}
        self.next_reference = 1
        self.selected_inherited_findings: tuple[Finding, ...] = ()
        self.selected_local_finding_ids: tuple[str, ...] = ()
        self.comparisons: tuple[Comparison, ...] = ()
        self.used_finding_ids: tuple[str, ...] = ()
        self.analysis_status: AnalysisStatus = "incomplete"

    def can_delegate(self, *, initial_assignment: bool = False) -> bool:
        if self.invocation.depth and len(self.invocation.partitions or ()) <= 1:
            return False
        groups = self.root_groups if self.delegation_first else None
        if initial_assignment:
            groups = self.unassigned_groups
        if initial_assignment and not groups:
            return False
        try:
            self.budget.check_child(self.invocation.depth + 1, min(map(len, groups)) if groups else 1,
                                    initial_assignment=initial_assignment)
        except BudgetExceeded:
            return False
        return True

    def candidate_schema(self) -> dict | None:
        if (self.runner.comparison_scan is None and self.runner.discovery_scan is None
                or self.invocation.partitions is None):
            return None
        return {
            "name": "list_candidates",
            "description": "Page through precomputed lexical candidate IDs and span hints for one authorized section. "
                           "Use get_spans or get_chunk to read exact evidence.",
            "input_schema": CandidatePageRequest.model_json_schema(),
        }

    def list_candidates(self, arguments: dict) -> DispatchResult:
        scan = self.runner.comparison_scan or self.runner.discovery_scan
        cap = self.runner.local_limits.max_response_bytes
        if scan is None or self.invocation.partitions is None:
            return DispatchResult(None, error_observation("unknown_tool", "Candidate pages require a scoped scan.", cap))
        try:
            request = CandidatePageRequest.model_validate(arguments)
        except ValidationError as exc:
            return DispatchResult(None, error_observation("invalid_arguments", validation_message(exc), cap))
        validated = request.model_dump(mode="json")
        pair = SectionTarget(request.document_id, request.section_id)
        try:
            if pair not in self.invocation.partitions:
                raise KeyError(pair)
            page = scan.candidate_page(pair, request.offset, request.limit)
        except KeyError:
            return DispatchResult(validated, error_observation("unknown_id", "No candidate inventory for this section.", cap))
        content = observation_json({"ok": True, "result": page})
        if len(content.encode("utf-8")) > cap:
            content = error_observation("response_too_large", "Try a smaller candidate page at the same offset.", cap)
        return DispatchResult(validated, content)

    def tool_schema(self) -> dict:
        return {
            "name": "delegate_research",
            "description": (
                "Investigate a narrower question in a fresh conversation with exact document and section pairs. "
                f"Current depth {self.invocation.depth}, maximum {self.budget.limits.max_depth}. "
                "All delegates share the run budget. Returned findings include original excerpts and coverage."
            ) + (" Use one whole admitted company group." if self.root_groups is not None else ""),
            "input_schema": _Delegate.model_json_schema(),
        }

    def finding_schema(self) -> dict:
        return {
            "name": SUBMIT_FINDINGS,
            "description": f"Retain up to {MAX_FINDINGS_PER_SUBMISSION} supported findings per call, "
                           f"up to {MAX_FINDINGS} unique local findings per invocation. "
                           "Supply a claim, analysis status and acknowledged chunk/span IDs. "
                           "Use returned sN references in finish_research; citations and source details are derived.",
            "input_schema": SubmitFindings.model_json_schema(),
        }

    def submit_findings(
        self, arguments: dict, coverage: CoverageReport, deadline: float, *, batch_share: int | None,
    ) -> tuple[DispatchResult, tuple[str, ...], tuple[FindingValidationIssue, ...]]:
        cap = self.runner.local_limits.max_response_bytes
        drafts = arguments.get("findings")
        if (arguments.keys() != {"findings"} or not isinstance(drafts, list)
                or not 1 <= len(drafts) <= MAX_FINDINGS_PER_SUBMISSION):
            issue = FindingValidationIssue("findings", "schema", "invalid_submission")
            content = error_observation("invalid_arguments",
                                        f"Supply a findings list with 1 to {MAX_FINDINGS_PER_SUBMISSION} items.", cap)
            return DispatchResult(None, content), (), (issue,)

        receipts = []
        accepted = []
        issues = []
        for index, raw in enumerate(drafts):
            path = f"findings[{index}]"
            try:
                if self.runner.clock() >= deadline:
                    raise FindingValidationError(path, "limit", "deadline", "Finding validation deadline reached.")
                draft = SpanFindingDraft.model_validate(raw)
                staged: dict[FindingKey, Finding] = {}
                finding = self._stage_finding(draft, coverage, (), staged, path, index)
                if self.runner.clock() >= deadline:
                    raise FindingValidationError(path, "limit", "deadline", "Finding validation deadline reached.")
            except ValidationError as exc:
                issue = finding_schema_issues(exc, path)[0]
            except FindingValidationError as exc:
                issue = exc.issue
            else:
                ref = f"s{finding.finding_id.rsplit('-', 1)[-1]}"
                self.retained_findings.update(staged)
                self.submitted_findings[ref] = finding
                accepted.append(ref)
                receipts.append({"index": index, "status": "accepted" if staged else "duplicate", "ref": ref})
                continue
            issues.append(issue)
            receipts.append({"index": index, "status": "rejected", "code": issue.code, "path": issue.path})

        share = min(cap, batch_share) if batch_share is not None else cap
        for count in range(len(receipts), -1, -1):
            content = observation_json({"ok": True, "result": {
                "findings": receipts[:count], "omitted": len(receipts) - count,
            }})
            if len(content.encode("utf-8")) <= share:
                self.reference_synthesis |= any("ref" in receipt for receipt in receipts[:count])
                return DispatchResult({}, content), tuple(dict.fromkeys(accepted)), tuple(issues)
        content = error_observation("finding_receipt_deferred", "Findings are retained in the invocation.", cap)
        return DispatchResult({}, content), tuple(dict.fromkeys(accepted)), tuple(issues)

    def delegate(
        self, arguments: dict, deadline: float, *, initial_assignment: bool = False, batch_share: int | None = None,
    ) -> tuple[DispatchResult, DelegationTrace]:
        cap = self.runner.local_limits.max_response_bytes
        validated = None
        invocation = None

        def reject(code: str, message: str, *, status: Literal["rejected", "failed"] = "rejected"):
            return (DispatchResult(validated, error_observation(code, message, cap)),
                    DelegationTrace(status, code, invocation=invocation, initial_assignment=initial_assignment))

        try:
            request = _Delegate.model_validate(arguments)
        except ValidationError as exc:
            return reject("invalid_arguments", validation_message(exc))
        validated = request.model_dump(mode="json")
        pairs = tuple(SectionTarget(p.document_id, p.section_id) for p in request.partitions)
        group = frozenset(pairs)
        parent_pairs = self.invocation.partitions
        depth = self.invocation.depth + 1
        try:
            if not initial_assignment:
                self.budget.check_child(depth, len(pairs))
            if parent_pairs is not None and not group.issubset(parent_pairs):
                return reject("scope_violation", "Delegate only authorized document and section pairs.")
            if self.root_groups is not None and group not in self.root_groups:
                return reject("group_violation", "Delegate one whole admitted company group at a time.")
            if initial_assignment and group not in self.unassigned_groups:
                return reject("duplicate_initial_assignment", "Choose a whole company group without an initial attempt.")
            if self.invocation.depth > 0 and group == frozenset(parent_pairs or ()):
                return reject("scope_violation", "A descendant must delegate a strict subset.")
            self.runner._validate_partitions(pairs)
            if self.runner.clock() >= deadline:
                raise BudgetExceeded("deadline")
            number = self.budget.reserve_child(depth, len(pairs), initial_assignment=initial_assignment)
        except BudgetExceeded as exc:
            return reject(str(exc), "Shared budget does not permit this delegation.")
        except (KeyError, ValueError):
            return reject("invalid_partitions", "Use unique canonical document and section pairs.")

        if initial_assignment:
            self.unassigned_groups.remove(group)
        namespace, separator, _ = self.invocation.invocation_id.rpartition(".")
        invocation = Invocation(self.invocation.run_id, f"{namespace}{separator}child-{number}", self.invocation.invocation_id,
                                depth, request.question, request.goal, pairs)
        try:
            child = self.runner._invoke(invocation, self.budget, self.clients, deadline)
        except Exception:
            return reject("child_execution_error", "Child invocation could not run.", status="failed")
        successful = child.stop_reason == "finished"
        findings = child.transferable_findings
        complete = (
            successful and bool(findings) and child.analysis_status == "analyzed"
            and all(p.status == "complete" for p in child.partition_coverage)
            and all(finding.analysis_status == "analyzed" for finding in findings)
        )
        status = ("finished" if complete else "incomplete") if successful else "failed"
        references = {}
        finding_records = {}
        for finding in findings:
            ref = f"d{self.next_reference}"
            self.next_reference += 1
            references[finding.finding_id] = ref
            finding_records[finding.finding_id] = {"ref": ref, **asdict(finding)}
        body = {
            "ok": successful or bool(findings), "result": {
                "child_invocation_id": invocation.invocation_id, "status": status,
                "stop_reason": child.stop_reason, "budget_stop": child.budget_stop,
                "analysis_status": child.analysis_status, "coverage": child.partition_coverage,
                "findings": tuple(finding_records.values()),
            },
        }
        if self.runner.comparison_scan is not None and successful:
            body["result"]["comparisons"] = child.comparisons
        if not body["ok"]:
            body["error"] = {"code": "child_not_finished", "message": "Inspect the child stop reason and coverage."}
        content = observation_json(body)
        delegation = DelegationTrace(status, error=None if successful else "child_not_finished",
                                     invocation=invocation, child=child, initial_assignment=initial_assignment)
        delegation.full_observation_bytes = len(content.encode("utf-8"))
        if body["ok"] and (self.runner.comparison_scan is not None or self.runner.discovery_scan is not None
                           or not successful or child.selected_inherited_findings
                           or any(op.accepted_finding_refs for step in child.steps for op in step.operations)):
            share = min(cap, batch_share) if batch_share is not None else cap
            if delegation.full_observation_bytes > share:
                content, findings, indices = self._compact_child_result(body["result"], child, share, finding_records)
                delegation.offered_comparison_indices = indices
                if content is None:
                    if successful:
                        delegation.status = "oversized"
                    delegation.error = "child_result_deferred"
                    content = error_observation("child_result_deferred", "Validated findings are retained in the report.", cap)
            else:
                delegation.offered_comparison_indices = tuple(range(len(child.comparisons)))
        if len(content.encode("utf-8")) > cap:
            if successful:
                delegation.status = "oversized"
            delegation.error = "response_too_large"
            content = error_observation(
                "response_too_large", "Child findings exceeded the cap. Try a smaller delegated task.",
                cap, max_response_bytes=cap, response_bytes=len(content.encode("utf-8")),
            )
            return DispatchResult(validated, content), delegation
        delegation.offered_finding_ids = tuple(finding.finding_id for finding in findings)
        delegation.finding_references = tuple((references[fid], fid) for fid in delegation.offered_finding_ids)
        self.reference_synthesis |= bool(delegation.finding_references)
        refs = tuple(dict.fromkeys(item.source for finding in findings for item in finding.evidence))
        return DispatchResult(validated, content, refs), delegation

    @staticmethod
    def _compact_child_result(
        result: dict, child: ExecutionTrace, cap: int, finding_records: dict[str, dict],
    ) -> tuple[str | None, tuple[Finding, ...], tuple[int, ...]]:
        """Keep whole validated findings and send a bounded summary when the full result cannot fit."""
        findings: list[Finding] = []
        indices: list[int] = []
        summaries: list[dict] = []
        record = {key: result[key] for key in ("child_invocation_id", "status", "analysis_status", "stop_reason", "budget_stop")}
        retained = child.transferable_findings

        def encode() -> str:
            return observation_json({"ok": True, "result": {**record,
                "findings": [finding_records[finding.finding_id] for finding in findings], "comparisons": summaries,
                "omitted_findings": len(retained) - len(findings),
                "omitted_comparisons": len(child.comparisons) - len(summaries)}})

        content = encode()
        if len(content.encode("utf-8")) > cap:
            return None, (), ()
        for finding in retained:
            findings.append(finding)
            candidate = encode()
            if len(candidate.encode("utf-8")) > cap:
                findings.pop()
            else:
                content = candidate
        offered = {finding.finding_id for finding in findings}
        for index, comparison in enumerate(child.comparisons):
            if not offered.issuperset(comparison.finding_ids):
                continue
            summaries.append({"risk": comparison.risk, "outcome": comparison.outcome,
                              "finding_ids": comparison.finding_ids, "evidence_years": comparison.evidence_years})
            candidate = encode()
            if len(candidate.encode("utf-8")) > cap:
                summaries.pop()
            else:
                indices.append(index)
                content = candidate
        return content, tuple(findings), tuple(indices)

    def record_delivery(self, step: ToolTrace) -> None:
        if not (step.observation or {}).get("ok"):
            return
        if step.delegation is None:
            if step.tool_name == SUBMIT_FINDINGS:
                step.acknowledged_finding_refs = step.offered_finding_refs
                self.acknowledged_references.update((ref, self.submitted_findings[ref])
                                                    for ref in step.acknowledged_finding_refs)
                return
            if step.tool_name != "get_spans":
                self.direct_ids.update(ref.chunk_id for ref in step.evidence)
            for chunk_id, spans in (step.observation or {}).get("span_ids", {}).items():
                self.direct_spans.setdefault(chunk_id, set()).update(spans)
        elif step.delegation.child is not None:
            offered = set(step.delegation.offered_finding_ids)
            findings = [finding for finding in step.delegation.child.transferable_findings if finding.finding_id in offered]
            self.delivered_findings.update((finding.finding_id, finding) for finding in findings)
            step.delegation.delivered_finding_ids = tuple(finding.finding_id for finding in findings)
            self.acknowledged_references.update((ref, self.delivered_findings[fid])
                                                for ref, fid in step.delegation.finding_references if fid in offered)

    def finish(self, action: _FinishAction, coverage: CoverageReport) -> None:
        scan = self.runner.comparison_scan
        if action.comparisons and scan is None:
            raise FindingValidationError("comparisons", "mode", "comparative_required", "Comparisons require comparative mode.")
        used = tuple(dict.fromkeys(action.used_finding_ids))
        for index, finding_id in enumerate(action.used_finding_ids):
            if finding_id not in self.delivered_findings:
                raise FindingValidationError(f"used_finding_ids[{index}]", "provenance", "unacknowledged_finding",
                                             "Use only delivered finding IDs.")
        used_evidence = tuple(item for finding_id in used for item in self.delivered_findings[finding_id].evidence)
        selected_ids = {item.source.chunk_id for item in used_evidence}
        cited = set(action.cited_chunk_ids)
        for index, chunk_id in enumerate(action.cited_chunk_ids):
            if chunk_id not in self.direct_ids and chunk_id not in self.direct_spans and chunk_id not in selected_ids:
                raise FindingValidationError(f"cited_chunk_ids[{index}]", "provenance", "undeclared_finding",
                                             "Declare the delivered finding IDs used by delegated citations.")

        staged: dict[FindingKey, Finding] = {}
        findings = [
            self._stage_finding(draft, coverage, used_evidence, staged, f"findings[{index}]", index)
            for index, draft in enumerate(action.findings)
        ]
        referenced = cited | {item.source.chunk_id for finding in findings for item in finding.evidence}
        for index, finding_id in enumerate(action.used_finding_ids):
            if not any(item.source.chunk_id in referenced for item in self.delivered_findings[finding_id].evidence):
                raise FindingValidationError(f"used_finding_ids[{index}]", "provenance", "unused_finding",
                                             "Each used finding must contribute a chunk to the answer or a new finding.")
        comparisons = []
        if scan is not None:
            for index, draft in enumerate(action.comparisons):
                try:
                    comparisons.append(resolve_comparison(draft, findings, scan.years))
                except ValueError as exc:
                    raise FindingValidationError(f"comparisons[{index}]", "comparison", "invalid_comparison", str(exc)) from exc
        self.retained_findings.update(staged)
        self.comparisons = tuple(comparisons)
        self.selected_local_finding_ids = tuple(dict.fromkeys(finding.finding_id for finding in findings))
        self.used_finding_ids = used
        self.analysis_status = (action.analysis_status or "incomplete") if referenced and not self.unassigned_groups else "incomplete"

    def finish_references(self, action: _ReferenceSynthesis, coverage: CoverageReport) -> tuple[str, ...]:
        scan = self.runner.comparison_scan
        if action.comparisons and scan is None:
            raise FindingValidationError("synthesis.comparisons", "mode", "comparative_required",
                                         "Comparisons require comparative mode.")
        staged: dict[FindingKey, Finding] = {}
        new = [
            self._stage_finding(draft, coverage, (), staged, f"synthesis.new_findings[{index}]", index)
            for index, draft in enumerate(action.new_findings)
        ]
        available = {**self.acknowledged_references, **{f"n{i}": f for i, f in enumerate(new, 1)}}
        selected = {finding.finding_id for finding in new}
        for index, ref in enumerate(action.cite_refs):
            self._check_reference(ref, available, f"synthesis.cite_refs[{index}]")
            selected.add(available[ref].finding_id)
        comparisons = []
        if scan is not None:
            for index, draft in enumerate(action.comparisons):
                path = f"synthesis.comparisons[{index}]"
                for position, assessment in enumerate(draft.assessments):
                    self._check_reference(assessment.ref, available, f"{path}.assessments[{position}].ref")
                    if assessment.context_ref is not None:
                        self._check_reference(assessment.context_ref, available, f"{path}.assessments[{position}].context_ref")
                try:
                    comparisons.append(resolve_reference_comparison(draft, available, scan.years))
                except ValueError as exc:
                    raise FindingValidationError(path, "comparison", "invalid_comparison", str(exc)) from exc
        selected.update(fid for item in comparisons for fid in item.finding_ids)
        inherited = tuple(dict.fromkeys(finding for ref, finding in self.acknowledged_references.items()
                                        if ref.startswith("d") and finding.finding_id in selected))
        local = tuple(finding for finding in (*self.retained_findings.values(), *staged.values())
                      if finding.finding_id in selected)
        sources = [item.source for finding in (*local, *inherited) for item in finding.evidence]
        if any((scan is not None and source.fiscal_year not in scan.years)
               or (self.invocation.partitions is not None
                   and SectionTarget(source.document_id, source.section_id) not in self.invocation.partitions)
               for source in sources):
            raise FindingValidationError("synthesis", "scope", "outside_scope",
                                         "Selected findings must use authorized partitions and requested fiscal years.")
        citations = tuple(sorted({source.chunk_id for source in sources}))
        if len(citations) > 50 or len(inherited) > 32:
            raise FindingValidationError("synthesis", "limit", "synthesis_limit",
                                         "Select fewer findings to stay within citation and inherited finding limits.")
        if not set(citations).issubset(ref.chunk_id for ref in coverage.visible_chunks):
            raise FindingValidationError("synthesis", "provenance", "unacknowledged_evidence",
                                         "Citations require acknowledged evidence.")
        self.retained_findings.update(staged)
        self.comparisons = tuple(comparisons)
        self.selected_local_finding_ids = tuple(item.finding_id for item in local)
        self.selected_inherited_findings = inherited
        self.used_finding_ids = tuple(item.finding_id for item in inherited)
        self.analysis_status = (action.analysis_status or "incomplete") if citations and not self.unassigned_groups else "incomplete"
        return citations

    @staticmethod
    def _check_reference(ref: str, available: dict[str, Finding], path: str) -> None:
        if ref not in available:
            raise FindingValidationError(path, "provenance", "unacknowledged_finding_reference",
                                         f"Unknown or unacknowledged finding reference: {ref}.")

    def _stage_finding(
        self, draft: FindingDraft, coverage: CoverageReport, used_evidence: tuple[FindingEvidence, ...],
        staged: dict[FindingKey, Finding], path: str, index: int,
    ) -> Finding:
        evidence = []
        for position, item in enumerate(draft.evidence):
            field = f"{path}.evidence[{position}]"
            location = f"Finding {index + 1}, evidence {position + 1}"
            resolved = self._resolve_evidence(item, used_evidence, field, location)
            source = resolved.source
            if evidence and (source.document_id, source.section_id) != (
                evidence[0].source.document_id, evidence[0].source.section_id,
            ):
                raise FindingValidationError(field, "scope", "mixed_finding_scope",
                                             "Each finding must cite evidence from one document and section.")
            evidence.append(resolved)
        source = evidence[0].source
        target = SectionTarget(source.document_id, source.section_id)
        key = (draft.claim, frozenset(evidence), draft.analysis_status, draft.uncertainty, draft.conflict)
        existing = self.retained_findings.get(key) or staged.get(key)
        if existing is not None:
            return existing
        count = len(self.retained_findings) + len(staged)
        if count >= MAX_FINDINGS:
            raise FindingValidationError(path, "limit", "finding_limit", f"At most {MAX_FINDINGS} local findings can be retained.")
        finding = Finding(
            finding_id=f"{self.invocation.invocation_id}:finding-{count + 1}",
            company_name=self.runner.corpus.get_filing(source.document_id).company_name,
            claim=draft.claim, evidence=tuple(dict.fromkeys(evidence)),
            analysis_status=draft.analysis_status, coverage_status=partition_status(coverage, target),
            uncertainty=draft.uncertainty, conflict=draft.conflict,
        )
        staged[key] = finding
        return finding

    def _resolve_evidence(
        self, item: EvidenceDraft | SpanEvidenceDraft, used_evidence: tuple[FindingEvidence, ...],
        field: str, location: str,
    ) -> FindingEvidence:
        if isinstance(item, SpanEvidenceDraft):
            if item.span_id in self.direct_spans.get(item.chunk_id, ()):
                chunk = self.runner.corpus.get_chunk(item.chunk_id)
                start, end = passage_spans(chunk.text)[item.span_id]
                return FindingEvidence(EvidenceRef.from_chunk(chunk), chunk.text[start:end], item.span_id)
            for observed in used_evidence:
                if observed.source.chunk_id == item.chunk_id and observed.span_id == item.span_id:
                    return observed
            raise FindingValidationError(f"{field}.span_id", "provenance", "unacknowledged_span",
                                         f"{location}: select a span from an acknowledged chunk or a declared delivered finding.")
        if item.chunk_id not in self.direct_ids:
            excerpts = [observed for observed in used_evidence if item.chunk_id == observed.source.chunk_id]
            if not excerpts:
                raise FindingValidationError(f"{field}.chunk_id", "provenance", "unacknowledged_chunk",
                                             f"{location}: use an observed chunk or a declared delivered excerpt, or omit it.")
            if not any(item.quote in observed.quote for observed in excerpts):
                raise FindingValidationError(f"{field}.quote", "provenance", "quote_outside_excerpt",
                                             f"{location} (chunk {item.chunk_id}): copy a contiguous quote exactly from "
                                             "the delivered excerpt or omit it.")
        chunk = self.runner.corpus.get_chunk(item.chunk_id)
        if item.quote not in chunk.text:
            raise FindingValidationError(f"{field}.quote", "provenance", "quote_mismatch",
                                         f"{location} (chunk {item.chunk_id}): copy a contiguous quote exactly from "
                                         "the delivered chunk text, including punctuation and spacing, or omit it.")
        return FindingEvidence(EvidenceRef.from_chunk(chunk), item.quote)

    def render_answer(self, draft: str) -> str:
        if self.runner.comparison_scan is not None:
            return render_comparisons(self.comparisons)
        if draft:
            return draft
        local = (finding for finding in self.retained_findings.values()
                 if finding.finding_id in self.selected_local_finding_ids)
        return render_findings((*local, *self.selected_inherited_findings)) or "No findings were selected. Research remains incomplete."

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
            "findings": tuple(self.retained_findings.values()),
            "comparisons": self.comparisons,
            "used_finding_ids": self.used_finding_ids,
            "selected_inherited_findings": self.selected_inherited_findings,
            "selected_local_finding_ids": self.selected_local_finding_ids,
            "analysis_status": self.analysis_status,
            "partition_coverage": partitions,
            "coverage_status": coverage_status,
        }
