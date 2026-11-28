"""Tests for research execution budgets."""

from unittest.mock import Mock

import pytest

from evidenceroute.research.budget import (
    BudgetExceededError,
    BudgetedLLMClient,
    ExecutionBudget,
    ExecutionLimits,
)
from evidenceroute.research.llm_client import (
    LLMResult,
    LLMClientError,
)


def make_result():
    return LLMResult(
        text="Test response",
        model_name="mock",
        latency_seconds=0.1,
        input_tokens=120,
        output_tokens=80,
    )


def test_budget_records_token_usage():
    budget = ExecutionBudget()
    client = Mock()
    client.generate.return_value = make_result()

    wrapped = BudgetedLLMClient(client, budget)

    wrapped.generate("Test question")

    assert budget.llm_calls == 1
    assert budget.input_tokens == 120
    assert budget.output_tokens == 80
    assert budget.total_tokens == 200
    assert budget.usage_complete is True


def test_budget_blocks_excess_calls():
    budget = ExecutionBudget(
        ExecutionLimits(max_llm_calls=1)
    )

    client = Mock()
    client.generate.return_value = make_result()

    wrapped = BudgetedLLMClient(client, budget)

    wrapped.generate("First question")

    with pytest.raises(BudgetExceededError):
        wrapped.generate("Second question")

    assert client.generate.call_count == 1
    assert budget.llm_calls == 1


def test_budget_blocks_after_token_exhaustion():
    budget = ExecutionBudget(
        ExecutionLimits(max_total_tokens=200)
    )

    client = Mock()
    client.generate.return_value = make_result()

    wrapped = BudgetedLLMClient(client, budget)

    wrapped.generate("First question")

    with pytest.raises(BudgetExceededError):
        wrapped.generate("Second question")

    assert budget.total_tokens == 200
    assert client.generate.call_count == 1


def test_budget_blocks_after_timeout():
    budget = ExecutionBudget(
        ExecutionLimits(max_seconds=1)
    )

    budget.started_at -= 10

    client = Mock()
    wrapped = BudgetedLLMClient(client, budget)

    with pytest.raises(BudgetExceededError):
        wrapped.generate("Test question")

    client.generate.assert_not_called()
    assert budget.llm_calls == 0


def test_failed_call_consumes_call_budget():
    budget = ExecutionBudget()
    client = Mock()

    client.generate.side_effect = LLMClientError(
        "API failure"
    )

    wrapped = BudgetedLLMClient(client, budget)

    with pytest.raises(LLMClientError):
        wrapped.generate("Test question")

    assert budget.llm_calls == 1
    assert budget.usage_complete is False


def test_missing_usage_is_reported():
    budget = ExecutionBudget()
    client = Mock()

    client.generate.return_value = LLMResult(
        text="Test response",
        model_name="mock",
        latency_seconds=0.1,
    )

    wrapped = BudgetedLLMClient(client, budget)

    wrapped.generate("Test question")

    assert budget.llm_calls == 1
    assert budget.usage_complete is False


@pytest.mark.parametrize("kwargs", [
    {"max_llm_calls": 0},
    {"max_total_tokens": 0},
    {"max_seconds": 0},
])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        ExecutionLimits(**kwargs)


def test_standard_rag_budget_exhaustion(monkeypatch):
    from evidenceroute.research.pipeline import run_standard_rag
    from evidenceroute.research.standard_rag import (
        BUDGET_ERROR_MESSAGE,
    )

    monkeypatch.setattr(
        "evidenceroute.research.standard_rag.build_rag_prompt",
        lambda *args: "Test prompt",
    )

    budget = ExecutionBudget(
        ExecutionLimits(max_llm_calls=1)
    )

    budget.llm_calls = 1

    client = Mock()
    wrapped = BudgetedLLMClient(client, budget)

    retriever = Mock()
    retriever.search.return_value = [Mock()]

    collected = []

    response = run_standard_rag(
        "What risks are reported?",
        retriever,
        wrapped,
        metrics_callback=collected.append,
    )

    assert response.answer == BUDGET_ERROR_MESSAGE
    assert response.citations == []
    assert collected[0].status == "budget_exhausted"
    assert collected[0].llm_calls == 0

    client.generate.assert_not_called()


def test_budget_snapshot():
    budget = ExecutionBudget(
        ExecutionLimits(
            max_llm_calls=3,
            max_total_tokens=1000,
            max_seconds=60,
        )
    )

    budget.reserve_call()
    budget.record_result(make_result())

    snapshot = budget.snapshot()

    assert snapshot["llm_calls"] == 1
    assert snapshot["total_tokens"] == 200
    assert snapshot["usage_complete"] is True
    assert snapshot["max_llm_calls"] == 3
    assert snapshot["max_total_tokens"] == 1000
    assert snapshot["max_seconds"] == 60
