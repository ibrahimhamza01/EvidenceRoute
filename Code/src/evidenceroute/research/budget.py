"""Execution budgets for SEC research workflows."""

import time
from dataclasses import dataclass, field

from evidenceroute.research.llm_client import (
    LLMClient,
    LLMClientError,
    LLMResult,
)


class BudgetExceededError(LLMClientError):
    """Raised when a research execution budget is exhausted."""


@dataclass(frozen=True)
class ExecutionLimits:
    """Configurable limits for a research request."""

    max_llm_calls: int = 8
    max_total_tokens: int = 32000
    max_seconds: float = 180.0

    def __post_init__(self):
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")

        if self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")

        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass
class ExecutionBudget:
    """Track resource consumption within one research request."""

    limits: ExecutionLimits = field(
        default_factory=ExecutionLimits
    )

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_complete: bool = True

    started_at: float = field(
        default_factory=time.monotonic
    )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def check(self) -> None:
        if self.llm_calls >= self.limits.max_llm_calls:
            raise BudgetExceededError(
                "Maximum LLM calls reached."
            )

        if self.total_tokens >= self.limits.max_total_tokens:
            raise BudgetExceededError(
                "Token budget exhausted."
            )

        if self.elapsed_seconds >= self.limits.max_seconds:
            raise BudgetExceededError(
                "Execution time budget exhausted."
            )

    def reserve_call(self) -> None:
        self.check()
        self.llm_calls += 1

    def record_result(self, result: LLMResult) -> None:
        if (
            type(result.input_tokens) is int
            and result.input_tokens >= 0
        ):
            self.input_tokens += result.input_tokens
        else:
            self.usage_complete = False

        if (
            type(result.output_tokens) is int
            and result.output_tokens >= 0
        ):
            self.output_tokens += result.output_tokens
        else:
            self.usage_complete = False

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_complete": self.usage_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "max_llm_calls": self.limits.max_llm_calls,
            "max_total_tokens": self.limits.max_total_tokens,
            "max_seconds": self.limits.max_seconds,
        }


class BudgetedLLMClient:
    """Wrap an LLM client with shared execution limits."""

    def __init__(
        self,
        client: LLMClient,
        budget: ExecutionBudget,
    ):
        self.client = client
        self.budget = budget

    def generate(self, prompt: str) -> LLMResult:
        self.budget.reserve_call()

        try:
            result = self.client.generate(prompt)
        except Exception:
            self.budget.usage_complete = False
            raise

        self.budget.record_result(result)

        return result
