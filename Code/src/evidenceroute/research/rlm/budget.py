"""Admission checks and measured usage for one sequential research tree."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

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


class SharedBudget:
    def __init__(self, limits: SharedLimits, clock: Callable[[], float] = monotonic):
        self.limits = limits
        self.clock = clock
        self.deadline = clock() + limits.deadline_seconds
        self.model_sessions = self.model_decisions = self.delegated_invocations = 0
        self.provider_attempts = self.corpus_attempts = self.partition_assignments = 0
        self.reported_tokens = self.attempts_with_missing_usage = 0

    def check_work(self) -> None:
        if self.clock() >= self.deadline:
            raise BudgetExceeded("deadline")
        if self.reported_tokens >= self.limits.max_reported_tokens:
            raise BudgetExceeded("reported_token_limit")

    def start_decision(self) -> None:
        self.check_work()
        self.model_decisions += 1

    def start_provider_attempt(self) -> None:
        self.check_provider_attempt()
        self.provider_attempts += 1

    def check_provider_attempt(self) -> None:
        self.check_work()
        if self.provider_attempts >= self.limits.max_provider_attempts:
            raise BudgetExceeded("provider_attempt_limit")

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

    def reserve_child(self, depth: int, partitions: int) -> int:
        self.check_child(depth, partitions)
        self.delegated_invocations += 1
        self.partition_assignments += partitions
        return self.delegated_invocations

    def check_child(self, depth: int, partitions: int) -> None:
        self.check_work()
        if depth > self.limits.max_depth:
            raise BudgetExceeded("depth_limit")
        if self.delegated_invocations >= self.limits.max_delegations:
            raise BudgetExceeded("delegation_limit")
        if self.partition_assignments + partitions > self.limits.max_partitions:
            raise BudgetExceeded("partition_limit")

    def snapshot(self) -> BudgetUsage:
        return BudgetUsage(
            self.model_sessions, self.model_decisions, self.delegated_invocations,
            self.provider_attempts, self.corpus_attempts, self.partition_assignments,
            self.reported_tokens, self.attempts_with_missing_usage,
            max(0.0, self.deadline - self.clock()),
        )
