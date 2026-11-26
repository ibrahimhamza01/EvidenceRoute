"""Admission checks and measured usage for one sequential research tree."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from evidenceroute.research.rlm.controller import ModelAttempt


class SharedLimits(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    max_depth: int = Field(default=2, ge=0, le=32)
    max_delegations: int = Field(default=8, ge=0)
    max_provider_attempts: int = Field(default=24, ge=0)
    max_corpus_attempts: int = Field(default=32, ge=0)
    max_partitions: int = Field(default=16, ge=0)
    max_reported_tokens: int = Field(default=60_000, ge=1)
    deadline_seconds: float = Field(default=90.0, gt=0)


class BudgetExceeded(RuntimeError):
    """A stable, model-facing admission failure code."""


class FinalizationPolicy(BaseModel):
    """Reserve provider allowances and configure the protected finish phase."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)

    child_provider_attempts: int = Field(default=4, ge=1)
    finalization_attempts: int = Field(default=1, ge=1)
    finalization_max_output_tokens: int = Field(default=2048, ge=1, le=128_000)
    finalization_tool_choice: Literal["auto", "required"] = "auto"
    root_token_headroom: int = Field(default=0, ge=0)
    child_token_headroom: int = Field(default=0, ge=0)
    finalization_timeout_seconds: float | None = Field(default=None, gt=0)
    finalization_timeout_retries: int = Field(default=0, ge=0, le=2)

    @model_validator(mode="after")
    def validate_timeout_retries(self):
        if self.finalization_timeout_retries >= self.finalization_attempts:
            raise ValueError("Timeout retries must fit the reserved finalization attempts.")
        return self


@dataclass
class ProviderAllocation:
    investigation_left: int
    finalization_left: int
    finalizing: bool = False


@dataclass(frozen=True)
class BudgetUsage:
    model_sessions: int
    model_decisions: int
    delegated_invocations: int
    provider_attempts: int
    corpus_attempts: int
    partition_assignments: int
    reported_tokens: int
    attempts_with_missing_usage: int
    remaining_seconds: float
    investigation_provider_attempts: int
    finalization_provider_attempts: int
    released_child_provider_slots: int


class SharedBudget:
    def __init__(self, limits: SharedLimits, clock: Callable[[], float] = monotonic, *,
                 finalization: FinalizationPolicy | None = None, deadline: float | None = None):
        self.limits = limits
        self.clock = clock
        self.deadline = clock() + limits.deadline_seconds
        if deadline is not None:
            self.deadline = min(self.deadline, deadline)
        self.model_sessions = self.model_decisions = self.delegated_invocations = 0
        self.provider_attempts = self.corpus_attempts = self.partition_assignments = 0
        self.reported_tokens = self.attempts_with_missing_usage = 0
        self.investigation_provider_attempts = self.finalization_provider_attempts = 0
        self.released_child_provider_slots = 0
        self.initial_delegations_left = self.initial_partitions_left = 0
        self.finalization = finalization
        self.active_allocation: ProviderAllocation | None = None
        self._root_allocation: ProviderAllocation | None = None
        self._active_children = 0
        if finalization is not None:
            reserve = finalization.finalization_attempts
            # Reserve even unadmitted children so an early branch cannot spend
            # a later sibling's share before that sibling is created.
            child_slots = limits.max_delegations * finalization.child_provider_attempts
            root_slots = limits.max_provider_attempts - child_slots
            if root_slots <= reserve or (limits.max_delegations and finalization.child_provider_attempts <= reserve):
                raise ValueError("Provider allocation must leave investigation and finalization capacity per invocation.")
            if finalization.root_token_headroom + limits.max_delegations * finalization.child_token_headroom >= limits.max_reported_tokens:
                raise ValueError("Token headroom must leave room for investigation within the shared token limit.")
            self._root_allocation = ProviderAllocation(root_slots - reserve, reserve)

    @contextmanager
    def invocation_allocation(self, *, root: bool) -> Iterator[None]:
        if self.finalization is None:
            yield
            return
        parent = self.active_allocation
        allocation = self._root_allocation
        if not root:
            reserve = self.finalization.finalization_attempts
            allocation = ProviderAllocation(self.finalization.child_provider_attempts - reserve, reserve)
            self._active_children += 1
        assert allocation is not None
        self.active_allocation = allocation
        try:
            yield
        finally:
            self.active_allocation = parent
            if root:
                self._root_allocation = None
            else:
                self._active_children -= 1
                unused = allocation.investigation_left + allocation.finalization_left
                assert self._root_allocation is not None
                self._root_allocation.investigation_left += unused
                self.released_child_provider_slots += unused

    def check_work(self) -> None:
        if self.clock() >= self.deadline:
            raise BudgetExceeded("deadline")
        if self.reported_tokens >= self.limits.max_reported_tokens:
            raise BudgetExceeded("reported_token_limit")

    def start_decision(self) -> None:
        self.check_work()
        self.model_decisions += 1

    def start_provider_attempt(self, *, estimated_tokens: int | None = None) -> None:
        self.check_provider_attempt(estimated_tokens=estimated_tokens)
        self.provider_attempts += 1
        allocation = self.active_allocation
        if allocation is not None and allocation.finalizing:
            allocation.finalization_left -= 1
            self.finalization_provider_attempts += 1
        else:
            if allocation is not None:
                allocation.investigation_left -= 1
            self.investigation_provider_attempts += 1

    def check_provider_attempt(self, *, estimated_tokens: int | None = None) -> None:
        self.check_work()
        if self.provider_attempts >= self.limits.max_provider_attempts:
            raise BudgetExceeded("provider_attempt_limit")
        allocation = self.active_allocation
        if allocation is not None:
            if allocation.finalizing and not allocation.finalization_left:
                raise BudgetExceeded("finalization_attempt_limit")
            if not allocation.finalizing and not allocation.investigation_left:
                raise BudgetExceeded("investigation_attempt_limit")
        policy = self.finalization
        if allocation is not None and policy and estimated_tokens is not None and (
            policy.root_token_headroom or policy.child_token_headroom
        ):
            finalizing_root = allocation.finalizing and allocation is self._root_allocation
            finalizing_child = allocation.finalizing and not finalizing_root
            held = 0 if finalizing_root else policy.root_token_headroom + policy.child_token_headroom * (
                self.initial_delegations_left + self._active_children - int(finalizing_child)
            )
            if self.reported_tokens + estimated_tokens + held > self.limits.max_reported_tokens:
                raise BudgetExceeded("finalization_token_headroom" if allocation.finalizing else "investigation_token_headroom")

    def record_provider_usage(self, attempt: ModelAttempt) -> None:
        if attempt.input_tokens is None or attempt.output_tokens is None:
            self.attempts_with_missing_usage += 1
        counters = (
            attempt.input_tokens, attempt.output_tokens,
            attempt.cache_read_input_tokens, attempt.cache_creation_input_tokens,
        )
        self.reported_tokens += sum(v for v in counters if v is not None)

    def start_corpus_attempt(self) -> None:
        self.check_work()
        if self.corpus_attempts >= self.limits.max_corpus_attempts:
            raise BudgetExceeded("corpus_attempt_limit")
        self.corpus_attempts += 1

    def reserve_child(self, depth: int, partitions: int, *, initial_assignment: bool = False) -> int:
        self.check_child(depth, partitions, initial_assignment=initial_assignment)
        if initial_assignment:
            self.initial_delegations_left -= 1
            self.initial_partitions_left -= partitions
        self.delegated_invocations += 1
        self.partition_assignments += partitions
        return self.delegated_invocations

    def check_child(self, depth: int, partitions: int, *, initial_assignment: bool = False) -> None:
        self.check_work()
        if depth > self.limits.max_depth:
            raise BudgetExceeded("depth_limit")
        held_delegations = 0 if initial_assignment else self.initial_delegations_left
        held_partitions = 0 if initial_assignment else self.initial_partitions_left
        if self.delegated_invocations + held_delegations >= self.limits.max_delegations:
            raise BudgetExceeded("delegation_limit")
        if self.partition_assignments + held_partitions + partitions > self.limits.max_partitions:
            raise BudgetExceeded("partition_limit")

    def snapshot(self) -> BudgetUsage:
        return BudgetUsage(
            self.model_sessions, self.model_decisions, self.delegated_invocations,
            self.provider_attempts, self.corpus_attempts, self.partition_assignments,
            self.reported_tokens, self.attempts_with_missing_usage,
            max(0.0, self.deadline - self.clock()),
            self.investigation_provider_attempts, self.finalization_provider_attempts,
            self.released_child_provider_slots,
        )
