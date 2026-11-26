"""Bounded model-directed corpus inspection with an offline scripted client."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from evidenceroute.research.rlm.budget import BudgetExceeded, BudgetUsage, SharedBudget
from evidenceroute.research.rlm.comparison import Comparison, ComparisonDraft, FindingReference, ReferenceComparisonDraft
from evidenceroute.research.rlm.coverage import CoverageReport, CoverageTracker, ResearchScope
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import AnalysisStatus, Finding, FindingDraft, PartitionCoverage, SpanEvidenceDraft
from evidenceroute.research.rlm.tools import (
    MIN_RESPONSE_BYTES,
    MAX_PROTOCOL_TOOL_BATCH,
    MAX_TOOL_BATCH,
    TOOL_USE_ID_PATTERN,
    CorpusToolDispatcher,
    DispatchResult,
    EvidenceRef,
    Identifier,
    StrictInput,
    error_observation,
    observation_json,
    tool_schemas,
    trace_data,
    validation_message,
)

if TYPE_CHECKING:
    from evidenceroute.research.rlm.recursive import DelegationTrace, Invocation, RecursiveSession
    from evidenceroute.research.rlm.scoped import ScopedCorpus


class _ToolAction(StrictInput):
    type: Literal["tool"]
    name: Identifier
    arguments: dict[str, JsonValue]


class _BatchCall(StrictInput):
    model_config = ConfigDict(str_strip_whitespace=False)

    tool_use_id: str = Field(pattern=TOOL_USE_ID_PATTERN)
    name: JsonValue
    arguments: JsonValue


class _ToolBatch(StrictInput):
    type: Literal["tool_batch"]
    calls: list[_BatchCall] = Field(min_length=2, max_length=MAX_PROTOCOL_TOOL_BATCH)


class _FinishAction(StrictInput):
    type: Literal["finish"]
    answer: str = Field(min_length=1, max_length=12_000)
    cited_chunk_ids: list[Identifier] = Field(default_factory=list, max_length=50)
    findings: list[FindingDraft] = Field(default_factory=list, max_length=8)
    used_finding_ids: list[Identifier] = Field(default_factory=list, max_length=32)
    analysis_status: AnalysisStatus | None = None
    comparisons: list[ComparisonDraft] = Field(default_factory=list, max_length=3)


class _NewSpanFinding(FindingDraft):
    evidence: list[SpanEvidenceDraft] = Field(min_length=1, max_length=3, description="Directly acknowledged root spans in one document and section.")


class _ReferenceSynthesis(StrictInput):
    comparisons: list[ReferenceComparisonDraft] = Field(default_factory=list, max_length=3)
    new_findings: list[_NewSpanFinding] = Field(default_factory=list, max_length=8)
    cite_refs: list[FindingReference] = Field(default_factory=list, max_length=8,
        description="Optional references for evidence outside comparisons. Comparison citations are automatic.")
    analysis_status: AnalysisStatus | None = None


class _ReferenceFinishAction(StrictInput):
    type: Literal["synthesis"]
    synthesis: _ReferenceSynthesis


_ACTION = TypeAdapter(Annotated[_ToolAction | _ToolBatch | _FinishAction | _ReferenceFinishAction, Field(discriminator="type")])


def action_schema() -> dict:
    """JSON Schema for the root model's tool and finish decisions."""
    return _ACTION.json_schema()


def finish_input_schema(*, recursive: bool = False, comparative: bool = False, reference_based: bool = False) -> dict:
    """Describe a finish payload whose action type is supplied by the adapter."""
    schema = (_ReferenceFinishAction if reference_based else _FinishAction).model_json_schema()
    del schema["properties"]["type"]
    schema["required"].remove("type")
    if reference_based:
        return schema
    if not comparative:
        del schema["properties"]["comparisons"]
        schema["$defs"]["FindingDraft"]["properties"]["evidence"]["items"] = {"$ref": "#/$defs/EvidenceDraft"}
        for name in ("ComparisonDraft", "EvidenceAssessment", "SpanEvidenceDraft"):
            schema["$defs"].pop(name, None)
    if not recursive:
        for field in ("findings", "used_finding_ids", "analysis_status"):
            del schema["properties"][field]
        schema.pop("$defs", None)
    return schema


class RootModelClient(Protocol):
    def next_action(
        self, question: str, tools: list[dict], observation: str | None, *,
        remaining_seconds: float,
    ) -> JsonValue:
        """Return one decision, using remaining_seconds as a cooperative timeout."""
        ...


class ScriptedModelClient:
    """Consume predetermined decisions in order and record supplied observations."""

    def __init__(self, decisions: Iterable[JsonValue]):
        self._decisions = deque(decisions)
        self.observations: list[str | None] = []
        self._budget_bound = False

    def bind_execution_budget(self, budget: SharedBudget) -> None:
        """Scripted decisions do not use a provider transport."""
        if self._budget_bound or self.observations:
            raise ValueError("Create a fresh scripted client for each invocation.")
        self._budget_bound = True

    def next_action(
        self, question: str, tools: list[dict], observation: str | None, *,
        remaining_seconds: float,
    ) -> JsonValue:
        self.observations.append(observation)
        if not self._decisions:
            raise ValueError("Scripted model ran out of decisions.")
        return self._decisions.popleft()


class RootLimits(StrictInput):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    max_iterations: int = Field(default=20, ge=1)
    max_tool_calls: int = Field(default=12, ge=0)
    max_response_bytes: int = Field(default=24_000, ge=MIN_RESPONSE_BYTES)
    deadline_seconds: float = Field(default=30.0, gt=0)


StopReason = Literal["finished", "iteration_limit", "tool_call_limit", "deadline", "model_error",
                     "budget_exhausted", "batch_observation_limit"]


@dataclass(frozen=True)
class ModelAttempt:
    """One provider request, including unsuccessful attempts and retries."""

    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    error: str | None = None
    response_model: str | None = None
    provider_latency_seconds: float | None = None
    response_received: bool = False
    provider_stop_reason: str | None = None
    request_bytes: int | None = None
    estimated_tokens: int | None = None
    timeout_seconds: float | None = None
    output_cap: int | None = None


@dataclass(kw_only=True)
class ToolTrace:
    tool_seconds: float | None = None
    requested_action: JsonValue = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    outcome: Literal["executed", "rejected", "skipped"] | None = None
    validated_arguments: dict | None = None
    observation: dict | None = None
    observation_sent: bool = False
    evidence: tuple[EvidenceRef, ...] = ()
    search_discovered: tuple[EvidenceRef, ...] = ()
    error: dict | None = None
    delegation: DelegationTrace | None = None
    delivery_error: str | None = None
    observation_bytes: int | None = None
    observation_attempted: bool = False

    def record(self, result: DispatchResult) -> None:
        self.validated_arguments = trace_data(result.arguments)
        self.observation = json.loads(result.content)
        self.observation_bytes = len(result.content.encode("utf-8"))
        self.error = self.observation.get("error")
        self.evidence = result.evidence
        self.search_discovered = result.search_discovered


@dataclass
class StepTrace(ToolTrace):
    iteration: int
    model_seconds: float = 0
    model_attempts: tuple[ModelAttempt, ...] = ()
    provider_tool_requests: tuple[dict, ...] = ()
    provider_tool_request_count: int = 0
    batch_operations: tuple[ToolTrace, ...] = ()
    phase: Literal["investigation", "finalization"] = "investigation"
    finalization_reason: str | None = None
    assignment_stage: Literal["assignment", "synthesis", "incomplete_finalization"] | None = None
    recovery: str | None = None
    envelope_bytes: int | None = None
    offered_tool_names: tuple[str, ...] = ()

    @property
    def operations(self) -> tuple[ToolTrace, ...]:
        return self.batch_operations or ((self,) if self.tool_name is not None else ())


@dataclass(frozen=True)
class ExecutionTrace:
    question: str
    steps: tuple[StepTrace, ...]
    tool_calls: int
    elapsed_seconds: float
    stop_reason: StopReason
    coverage: CoverageReport
    answer: str | None
    cited_chunk_ids: tuple[str, ...]
    invocation: Invocation | None = None
    shared_limits: dict[str, int | float] | None = None
    shared_budget: BudgetUsage | None = None
    budget_stop: str | None = None
    findings: tuple[Finding, ...] = ()
    used_finding_ids: tuple[str, ...] = ()
    analysis_status: AnalysisStatus | None = None
    partition_coverage: tuple[PartitionCoverage, ...] = ()
    coverage_status: Literal["unknown", "complete", "partial", "missing"] | None = None
    comparisons: tuple[Comparison, ...] = ()
    initial_assignment_stop: str | None = None
    selected_inherited_findings: tuple[Finding, ...] = ()

    @property
    def model_attempts(self) -> tuple[ModelAttempt, ...]:
        return tuple(attempt for step in self.steps for attempt in step.model_attempts)


class RootController:
    """Run a bounded sequence of structured decisions and corpus observations."""

    def __init__(
        self, corpus: CorpusEnvironment | ScopedCorpus, model: RootModelClient, *,
        limits: RootLimits | None = None, clock: Callable[[], float] = monotonic, include_spans: bool = False,
    ):
        self._corpus = corpus
        self._model = model
        self._limits = limits if limits is not None else RootLimits()
        self._clock = clock
        self._comparative = include_spans
        self._dispatcher = CorpusToolDispatcher(corpus, max_response_bytes=self._limits.max_response_bytes,
                                                include_spans=include_spans)

    def run(self, question: str, *, scope: ResearchScope | None = None,
            _recursive: RecursiveSession | None = None) -> ExecutionTrace:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question must be a non-blank string.")
        start = self._clock()
        deadline = start + self._limits.deadline_seconds
        if _recursive is not None:
            deadline = min(deadline, _recursive.parent_deadline, _recursive.budget.deadline)
        scope = scope if scope is not None else ResearchScope()
        scope.validate(self._corpus)
        coverage = CoverageTracker(scope)
        steps: list[StepTrace] = []
        calls = 0
        pending: str | None = None
        replayed_observation: StepTrace | None = None
        answer = None
        citations: tuple[str, ...] = ()
        stop: StopReason = "iteration_limit"
        budget_stop = None
        tools = tool_schemas(max_response_bytes=self._limits.max_response_bytes, comparative=self._comparative)
        if self._comparative and scope.required_sections:
            tools = [tool for tool in tools if tool["name"] not in {"list_companies", "list_filings", "get_filing", "list_sections"}]
        if _recursive is not None:
            if _recursive.invocation.depth < _recursive.budget.limits.max_depth:
                tools.append(_recursive.tool_schema())
            candidate_tool = _recursive.candidate_schema()
            if candidate_tool is not None:
                tools.append(candidate_tool)
        finalization_reason = None
        replay_reason = None
        tool_limit = self._limits.max_tool_calls
        reinvestigation_limit = _recursive.runner.max_root_reinvestigation_calls if _recursive else None

        for iteration in range(1, self._limits.max_iterations + 1):
            model_start = self._clock()
            if model_start >= deadline:
                stop = "deadline"
                break
            offered_tools = tools
            assignment_stage = None
            if _recursive is not None:
                budget = _recursive.budget
                allocation = budget.active_allocation
                unassigned = bool(_recursive.unassigned_groups)
                if _recursive.delegation_first and not unassigned and reinvestigation_limit is not None:
                    tool_limit = min(tool_limit, calls + reinvestigation_limit)
                    reinvestigation_limit = None
                tool_slots = tool_limit - calls
                reinvestigation_exhausted = not tool_slots and tool_limit < self._limits.max_tool_calls
                corpus_slots = budget.limits.max_corpus_attempts - budget.corpus_attempts
                can_delegate = (allocation is not None or unassigned) and _recursive.can_delegate(initial_assignment=unassigned)
                if finalization_reason is None and (allocation is not None or unassigned or reinvestigation_exhausted):
                    if allocation is not None and not allocation.investigation_left:
                        finalization_reason = "investigation_attempt_limit"
                    elif unassigned and allocation is None and 0 < budget.provider_attempts == budget.limits.max_provider_attempts - 1:
                        finalization_reason = "reserved_last_provider_attempt"
                    elif not tool_slots:
                        finalization_reason = "root_reinvestigation_limit" if reinvestigation_exhausted else "tool_call_limit"
                    elif unassigned and not can_delegate:
                        finalization_reason = "initial_assignment_unavailable"
                    elif not corpus_slots and not can_delegate:
                        finalization_reason = "corpus_attempt_limit"
                    elif self._limits.max_iterations - iteration + 1 <= (allocation.finalization_left if allocation else 1):
                        finalization_reason = "reserved_local_decisions"
                if allocation is not None:
                    allocation.finalizing = finalization_reason is not None
                    if allocation.finalizing and not allocation.finalization_left:
                        stop, budget_stop = "budget_exhausted", "finalization_attempt_limit"
                        break
                if finalization_reason:
                    offered_tools = []
                    notice = (
                        f"Finalization only ({finalization_reason}). Call finish_research now using delivered "
                        "evidence. If evidence is insufficient, report an incomplete result or abstain."
                    )
                    if _recursive.reference_synthesis:
                        notice += " Submit comparisons by short finding reference and omit answer prose."
                        if replay_reason == "truncated_response":
                            notice += " Prioritize one comparison per company and reference the strongest evidence."
                    else:
                        notice += " Keep the answer and findings concise and avoid repeating quotes."
                        if replay_reason == "truncated_response" and self._comparative:
                            notice += (" The previous output was truncated. Prioritize one comparison per company "
                                       "using one exact span per finding. Keep each finding in one document and section.")
                elif unassigned:
                    offered_tools = [tool for tool in tools if tool["name"] == "delegate_research"]
                    notice = (
                        f"Initial assignment has {len(_recursive.unassigned_groups)} unassigned company group(s). "
                        "Use delegate_research for one unassigned whole group per turn. Keep its fiscal years together."
                    )
                else:
                    notice = ""
                if allocation is not None and not finalization_reason:
                    if not unassigned:
                        offered_tools = [tool for tool in tools if (
                            can_delegate if tool["name"] == "delegate_research" else corpus_slots
                        )]
                    notice += (" " if notice else "") + (
                        f"Investigation: {allocation.investigation_left} provider attempt(s) for this invocation, "
                        f"{tool_slots} local tool slot(s), {corpus_slots} shared corpus attempt(s), "
                        f"{budget.limits.max_provider_attempts - budget.provider_attempts} shared provider attempt(s) "
                        f"including reserves, {budget.limits.max_reported_tokens - budget.reported_tokens} "
                        f"reported-token headroom, {deadline - model_start:.0f}s remaining. "
                        f"{allocation.finalization_left} attempt(s) reserved for this invocation's finish."
                    )
                    if not unassigned:
                        notice += " Finish once the evidence is sufficient."
                if _recursive.delegation_first:
                    if not unassigned:
                        assignment_stage = "synthesis"
                    elif finalization_reason:
                        assignment_stage = "incomplete_finalization"
                        notice += f" Report that {len(_recursive.unassigned_groups)} admitted group(s) had no attempt."
                    else:
                        assignment_stage = "assignment"
                    set_finish = getattr(self._model, "set_finish_allowed", None)
                    if callable(set_finish):
                        set_finish(assignment_stage != "assignment")
                notify = getattr(self._model, "set_research_budget_notice", None)
                if callable(notify) and (allocation is not None or _recursive.delegation_first):
                    notify(notice)
            if _recursive is not None:
                try:
                    _recursive.budget.start_decision()
                except BudgetExceeded as exc:
                    budget_stop = str(exc)
                    stop = "deadline" if budget_stop == "deadline" else "budget_exhausted"
                    break
            step = StepTrace(
                iteration, phase="finalization" if finalization_reason else "investigation",
                finalization_reason=finalization_reason, assignment_stage=assignment_stage, recovery=replay_reason,
                offered_tool_names=tuple(tool["name"] for tool in offered_tools),
            )
            steps.append(step)
            prior = steps[-2] if pending is not None else replayed_observation
            recovering = False
            try:
                raw = self._model.next_action(
                    question, offered_tools, pending, remaining_seconds=deadline - model_start,
                )
            except BudgetExceeded as exc:
                code = str(exc)
                step.error = {"code": code}
                prepare_replay = getattr(self._model, "prepare_request_replay", None)
                recovering = code == "investigation_token_headroom" and callable(prepare_replay) and prepare_replay()
                if not recovering:
                    budget_stop = code
                    stop = "deadline" if code == "deadline" else "budget_exhausted"
                    break
                finalization_reason = code
            except Exception as exc:
                timed_out = self._clock() >= deadline
                allocation = _recursive.budget.active_allocation if _recursive is not None else None
                prepare_recovery = getattr(self._model, "prepare_truncation_recovery", None)
                recovering = (
                    not timed_out and iteration < self._limits.max_iterations
                    and allocation is not None and allocation.finalization_left > 0
                    and callable(prepare_recovery) and prepare_recovery()
                )
                if not recovering:
                    step.error = {"code": "model_error", "message": type(exc).__name__}
                    stop = "deadline" if timed_out else "model_error"
                    break
                step.error = {"code": "truncated_response"}
            finally:
                step.model_seconds = self._clock() - model_start
                # Provider clients can report attempts and decline to credit an
                # observation when the remote endpoint did not acknowledge it.
                step.model_attempts = getattr(self._model, "last_attempts", ())
                step.provider_tool_requests = getattr(self._model, "last_tool_requests", ())
                step.provider_tool_request_count = getattr(self._model, "last_tool_request_count", 0)
                if prior is not None and step.model_attempts:
                    for observation in prior.batch_operations or (prior,):
                        observation.observation_attempted = True
                if prior is not None and getattr(self._model, "last_observation_delivered", True):
                    for observation in prior.batch_operations or (prior,):
                        self._deliver(observation, coverage)
                        if _recursive is not None and not observation.delivery_error:
                            _recursive.record_delivery(observation)
                    prior = None

            if recovering:
                # The provider reuses the valid request history, including any
                # tool results already attached to it. Credit those results once.
                replayed_observation = prior
                pending = None
                replay_reason = step.error["code"]
                if finalization_reason is None and step.assignment_stage != "assignment":
                    finalization_reason = "truncated_response"
                continue
            replayed_observation = None
            replay_reason = None

            try:
                candidate = json.loads(json.dumps(raw, allow_nan=False))
                step.requested_action = trace_data(candidate)
                if self._clock() >= deadline:
                    stop = "deadline"
                    break
                action = _ACTION.validate_python(candidate)
            except (TypeError, ValueError, RecursionError) as exc:
                message = validation_message(exc) if isinstance(exc, ValidationError) else "Expected a JSON action."
                pending = self._feedback(step, "invalid_action", message)
                if self._clock() >= deadline:
                    stop = "deadline"
                    break
                continue
            if self._clock() >= deadline:
                stop = "deadline"
                break

            if isinstance(action, (_FinishAction, _ReferenceFinishAction)):
                if step.assignment_stage == "assignment":
                    pending = self._feedback(step, "initial_assignment_required", "Attempt each whole group first.")
                    continue
                if isinstance(action, _ReferenceFinishAction):
                    if _recursive is None or not _recursive.reference_synthesis:
                        pending = self._feedback(step, "invalid_action", "Reference synthesis requires a comparative root.")
                        continue
                    try:
                        citations = _recursive.finish_references(action.synthesis, coverage.report())
                    except ValueError as exc:
                        pending = self._feedback(step, "invalid_findings", str(exc))
                        continue
                    answer = _recursive.render_answer("")
                    stop = "finished"
                    break
                visible = {ref.chunk_id for ref in coverage.report().visible_chunks}
                if not visible.issuperset(action.cited_chunk_ids):
                    pending = self._feedback(step, "unseen_citation", "Cite only chunks supplied to the model.")
                    continue
                if _recursive is not None:
                    failure = _recursive.finish(action, coverage.report())
                    if failure is not None:
                        pending = self._feedback(step, "invalid_findings", failure)
                        continue
                elif action.findings or action.used_finding_ids or action.analysis_status is not None or action.comparisons:
                    pending = self._feedback(step, "invalid_action", "Recursive fields require a research session.")
                    continue
                answer = _recursive.render_answer(action.answer) if _recursive is not None else action.answer
                citations = tuple(action.cited_chunk_ids)
                stop = "finished"
                break

            if isinstance(action, _ToolBatch):
                if len({call.tool_use_id for call in action.calls}) != len(action.calls):
                    step.error = {"code": "invalid_tool_use"}
                    stop = "model_error"
                    break
                pending, calls = self._execute_batch(action, step, calls, tool_limit, deadline, coverage, _recursive)
                if pending is None:
                    stop = "batch_observation_limit"
                    break
                if self._clock() >= deadline:
                    stop = "deadline"
                    break
                continue

            step.tool_name = trace_data(action.name)
            if step.provider_tool_requests:
                step.tool_use_id = step.provider_tool_requests[0]["tool_use_id"]
            if step.phase == "finalization":
                step.outcome = "rejected"
                pending = self._feedback(step, "finalization_only", "Call finish_research using delivered evidence.")
                continue
            if calls >= tool_limit:
                step.outcome = "skipped"
                self._feedback(step, "tool_call_limit", "Tool call budget exhausted.")
                stop = "tool_call_limit"
                break
            calls += 1
            pending = self._execute_tool(step, action, deadline, coverage, _recursive,
                                         initial_assignment=step.assignment_stage == "assignment")
            if self._clock() >= deadline:
                stop = "deadline"
                break

        report = coverage.report()
        metadata = _recursive.trace_fields(report) if _recursive is not None else {}
        if _recursive is not None and _recursive.unassigned_groups:
            metadata["initial_assignment_stop"] = finalization_reason or budget_stop or stop
        return ExecutionTrace(question, tuple(steps), calls, self._clock() - start,
                              stop, report, answer, citations, budget_stop=budget_stop, **metadata)

    def _execute_tool(
        self, operation: ToolTrace, action: _ToolAction, deadline: float,
        coverage: CoverageTracker, recursive: RecursiveSession | None, *, initial_assignment: bool = False,
        batch_share: int | None = None,
    ) -> str:
        started = self._clock()
        if initial_assignment and action.name != "delegate_research":
            operation.outcome = "rejected"
            return self._feedback(operation, "initial_assignment_required", "Delegate whole company groups first.")
        if recursive is not None and action.name == "delegate_research":
            result, operation.delegation = recursive.delegate(action.arguments, deadline, initial_assignment=initial_assignment,
                                                              batch_share=batch_share)
        else:
            try:
                if recursive is not None:
                    recursive.budget.start_corpus_attempt()
            except BudgetExceeded as exc:
                operation.outcome = "skipped"
                return self._feedback(operation, str(exc), "Shared corpus work budget exhausted.")
            if recursive is not None and action.name == "list_candidates":
                result = recursive.list_candidates(action.arguments)
            else:
                result = self._dispatcher.dispatch(action.name, action.arguments,
                                                   delivered_chunks=recursive.direct_ids if recursive else None)
        operation.tool_seconds = self._clock() - started
        delegation = operation.delegation
        operation.outcome = ("rejected" if result.arguments is None or
                             (delegation is not None and delegation.status == "rejected") else "executed")
        operation.record(result)
        coverage.record_search(result.search_discovered)
        return result.content

    def _execute_batch(
        self, batch: _ToolBatch, step: StepTrace, calls: int, tool_limit: int, deadline: float,
        coverage: CoverageTracker, recursive: RecursiveSession | None,
    ) -> tuple[str | None, int]:
        step.batch_operations = tuple(ToolTrace(
            tool_use_id=trace_data(call.tool_use_id),
            tool_name=trace_data(call.name) if isinstance(call.name, str) else None,
            requested_action=trace_data(call.model_dump()),
        ) for call in batch.calls)
        if len(batch.calls) > MAX_TOOL_BATCH:
            return self._reject_oversized_batch(batch, step), calls
        cap = self._limits.max_response_bytes
        overflow = {"ok": False, "error": {"code": "batch_observation_limit"}}
        results = [{"tool_use_id": call.tool_use_id, "observation": overflow} for call in batch.calls]

        def encode() -> str:
            return observation_json({"tool_results": results})

        # Reserve a bounded error slot for every ID before performing any work.
        step.envelope_bytes = len(encode().encode("utf-8"))
        if step.envelope_bytes > cap:
            step.error = {"code": "batch_observation_limit"}
            for operation in step.batch_operations:
                operation.outcome = "skipped"
                self._feedback(operation, "batch_observation_limit", "The result envelope cannot fit.")
            return None, calls

        mixed = any(isinstance(call.name, str) and call.name.strip() == "finish_research" for call in batch.calls)
        for index, (call, operation) in enumerate(zip(batch.calls, step.batch_operations)):
            if step.phase == "finalization":
                operation.outcome = "rejected"
                self._feedback(operation, "finalization_only", "Call finish_research using delivered evidence.")
            elif mixed:
                operation.outcome = "rejected"
                self._feedback(operation, "mixed_finish_batch", "Call finish_research on its own.")
            else:
                share = (cap - len(encode().encode("utf-8"))) // (len(batch.calls) - index)
                calls = self._execute_batch_call(call, operation, calls, tool_limit, deadline, coverage, recursive,
                                                 provider_turn=bool(step.model_attempts),
                                                 initial_assignment=step.assignment_stage == "assignment", batch_share=share)
            results[index]["observation"] = operation.observation
            if len(encode().encode("utf-8")) > cap:
                if operation.error:
                    results[index]["observation"] = {"ok": False, "error": {"code": operation.error["code"]}}
                if not operation.error or len(encode().encode("utf-8")) > cap:
                    results[index]["observation"] = overflow
                    operation.delivery_error = "batch_observation_limit"
        content = encode()
        step.envelope_bytes = len(content.encode("utf-8"))
        return content, calls

    def _reject_oversized_batch(self, batch: _ToolBatch, step: StepTrace) -> str | None:
        error = {"code": "tool_batch_limit", "max_operations": MAX_TOOL_BATCH}
        if step.phase == "finalization":
            error["finish_only"] = True
            instruction = "Call finish_research on its own."
        else:
            instruction = f"Request at most {MAX_TOOL_BATCH} operations per decision. Call finish_research on its own."
        step.error = {**error, "requested_operations": len(batch.calls)}

        content = None
        for feedback in ({**error, "message": instruction}, error):
            observation = {"ok": False, "error": feedback}
            envelope = observation_json({"tool_results": [
                {"tool_use_id": call.tool_use_id, "observation": observation} for call in batch.calls
            ]})
            if len(envelope.encode("utf-8")) <= self._limits.max_response_bytes:
                content = envelope
                break

        if content is None:
            step.error = {"code": "batch_observation_limit", "requested_operations": len(batch.calls)}
        else:
            step.envelope_bytes = len(content.encode("utf-8"))
        result = DispatchResult(None, observation_json(observation))
        for operation in step.batch_operations:
            operation.outcome = "rejected"
            operation.record(result)
            if content is None:
                operation.delivery_error = "batch_observation_limit"
        return content

    def _execute_batch_call(
        self, call: _BatchCall, operation: ToolTrace, calls: int, tool_limit: int, deadline: float,
        coverage: CoverageTracker, recursive: RecursiveSession | None, *, provider_turn: bool, initial_assignment: bool,
        batch_share: int,
    ) -> int:
        try:
            if self._clock() >= deadline:
                raise BudgetExceeded("deadline")
            if recursive is not None:
                if provider_turn and recursive.budget.active_allocation is None:
                    recursive.budget.check_provider_attempt()
                else:
                    recursive.budget.check_work()
            if calls >= tool_limit:
                raise BudgetExceeded("tool_call_limit")
        except BudgetExceeded as exc:
            operation.outcome = "skipped"
            self._feedback(operation, str(exc), "Tool request skipped because its budget is exhausted.")
            return calls
        try:
            action = _ToolAction.model_validate({"type": "tool", "name": call.name, "arguments": call.arguments})
        except ValidationError as exc:
            operation.outcome = "rejected"
            self._feedback(operation, "invalid_arguments", validation_message(exc))
        else:
            operation.tool_name = trace_data(action.name)
            self._execute_tool(operation, action, deadline, coverage, recursive, initial_assignment=initial_assignment,
                               batch_share=batch_share)
        return calls + 1

    def _feedback(self, step: ToolTrace, code: str, message: str) -> str:
        content = error_observation(code, message, self._limits.max_response_bytes)
        step.record(DispatchResult(None, content))
        return content

    @staticmethod
    def _deliver(step: ToolTrace, coverage: CoverageTracker) -> None:
        step.observation_sent = True
        if not step.delivery_error and step.validated_arguments is not None and step.observation["ok"]:
            coverage.record_delivery(step.tool_name, step.validated_arguments,
                                     step.observation["result"], step.evidence)
