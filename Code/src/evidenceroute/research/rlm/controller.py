"""Bounded model-directed corpus inspection with an offline scripted client."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import monotonic
from typing import Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from evidenceroute.research.rlm.coverage import CoverageReport, CoverageTracker, ResearchScope
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.tools import (
    MIN_RESPONSE_BYTES,
    CorpusToolDispatcher,
    DispatchResult,
    EvidenceRef,
    Identifier,
    StrictInput,
    error_observation,
    tool_schemas,
    validation_message,
)


class _ToolAction(StrictInput):
    type: Literal["tool"]
    name: Identifier
    arguments: dict[str, JsonValue]


class _FinishAction(StrictInput):
    type: Literal["finish"]
    answer: str = Field(min_length=1, max_length=12_000)
    cited_chunk_ids: list[Identifier] = Field(default_factory=list, max_length=50)


_ACTION = TypeAdapter(Annotated[_ToolAction | _FinishAction, Field(discriminator="type")])


def action_schema() -> dict:
    """JSON Schema for the root model's tool and finish decisions."""
    return _ACTION.json_schema()


def finish_input_schema() -> dict:
    """Describe a finish payload whose action type is supplied by the adapter."""
    schema = _FinishAction.model_json_schema()
    del schema["properties"]["type"]
    schema["required"].remove("type")
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


StopReason = Literal["finished", "iteration_limit", "tool_call_limit", "deadline", "model_error"]


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


@dataclass
class StepTrace:
    iteration: int
    model_seconds: float = 0
    tool_seconds: float | None = None
    requested_action: JsonValue = None
    tool_name: str | None = None
    validated_arguments: dict | None = None
    observation: dict | None = None
    observation_sent: bool = False
    evidence: tuple[EvidenceRef, ...] = ()
    search_discovered: tuple[EvidenceRef, ...] = ()
    error: dict | None = None
    model_attempts: tuple[ModelAttempt, ...] = ()

    def record(self, result: DispatchResult) -> None:
        self.validated_arguments = result.arguments
        self.observation = json.loads(result.content)
        self.error = self.observation.get("error")
        self.evidence = result.evidence
        self.search_discovered = result.search_discovered


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

    @property
    def model_attempts(self) -> tuple[ModelAttempt, ...]:
        return tuple(attempt for step in self.steps for attempt in step.model_attempts)


class RootController:
    """Run a bounded sequence of structured decisions and corpus observations."""

    def __init__(
        self, corpus: CorpusEnvironment, model: RootModelClient, *,
        limits: RootLimits | None = None, clock: Callable[[], float] = monotonic,
    ):
        self._corpus = corpus
        self._model = model
        self._limits = limits if limits is not None else RootLimits()
        self._clock = clock
        self._dispatcher = CorpusToolDispatcher(corpus, max_response_bytes=self._limits.max_response_bytes)

    def run(self, question: str, *, scope: ResearchScope | None = None) -> ExecutionTrace:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Research question must be a non-blank string.")
        start = self._clock()
        deadline = start + self._limits.deadline_seconds
        scope = scope if scope is not None else ResearchScope()
        scope.validate(self._corpus)
        coverage = CoverageTracker(scope)
        steps: list[StepTrace] = []
        calls = 0
        pending: str | None = None
        answer = None
        citations: tuple[str, ...] = ()
        stop: StopReason = "iteration_limit"

        for iteration in range(1, self._limits.max_iterations + 1):
            model_start = self._clock()
            if model_start >= deadline:
                stop = "deadline"
                break
            step = StepTrace(iteration)
            steps.append(step)
            try:
                raw = self._model.next_action(
                    question, tool_schemas(), pending, remaining_seconds=deadline - model_start,
                )
            except Exception as exc:
                step.error = {"code": "model_error", "message": type(exc).__name__}
                stop = "deadline" if self._clock() >= deadline else "model_error"
                break
            finally:
                step.model_seconds = self._clock() - model_start
                # Provider clients can report attempts and decline to credit an
                # observation when the remote endpoint did not acknowledge it.
                step.model_attempts = getattr(self._model, "last_attempts", ())
                if pending is not None and getattr(self._model, "last_observation_delivered", True):
                    self._deliver(steps[-2], coverage)

            try:
                step.requested_action = json.loads(json.dumps(raw, allow_nan=False))
                if self._clock() >= deadline:
                    stop = "deadline"
                    break
                action = _ACTION.validate_python(step.requested_action)
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

            if isinstance(action, _FinishAction):
                visible = {ref.chunk_id for ref in coverage.report().visible_chunks}
                if not visible.issuperset(action.cited_chunk_ids):
                    pending = self._feedback(step, "unseen_citation", "Cite only chunks supplied to the model.")
                    continue
                answer = action.answer
                citations = tuple(action.cited_chunk_ids)
                stop = "finished"
                break

            step.tool_name = action.name
            if calls >= self._limits.max_tool_calls:
                self._feedback(step, "tool_call_limit", "Tool call budget exhausted.")
                stop = "tool_call_limit"
                break
            calls += 1
            tool_start = self._clock()
            result = self._dispatcher.dispatch(action.name, action.arguments)
            step.tool_seconds = self._clock() - tool_start
            step.record(result)
            coverage.record_search(result.search_discovered)
            pending = result.content
            if self._clock() >= deadline:
                stop = "deadline"
                break

        return ExecutionTrace(question, tuple(steps), calls, self._clock() - start,
                              stop, coverage.report(), answer, citations)

    def _feedback(self, step: StepTrace, code: str, message: str) -> str:
        content = error_observation(code, message, self._limits.max_response_bytes)
        step.record(DispatchResult(None, content))
        return content

    @staticmethod
    def _deliver(step: StepTrace, coverage: CoverageTracker) -> None:
        step.observation_sent = True
        if step.validated_arguments is not None and step.observation["ok"]:
            coverage.record_delivery(step.tool_name, step.validated_arguments,
                                     step.observation["result"], step.evidence)
