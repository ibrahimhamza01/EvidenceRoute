"""Tests for Agentic RAG planning and research."""

from unittest.mock import Mock

import pytest

from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.budget import BudgetExceededError
from evidenceroute.research.agentic.planner import (
    create_research_plan,
    parse_plan,
)
from evidenceroute.research.agentic.tools import (
    SECTools,
    SearchRequest,
)
from evidenceroute.research.agentic.graph import (
    build_agent_graph,
)


PLAN = """
{
    "tasks": [
        {
            "query": "NVIDIA export controls",
            "tickers": ["NVDA"],
            "fiscal_years": [2024]
        },
        {
            "query": "AMD export controls",
            "tickers": ["AMD"],
            "fiscal_years": [2024]
        }
    ]
}
"""


def mock_llm(text=PLAN):
    llm = Mock()

    llm.generate.return_value = LLMResult(
        text=text,
        model_name="mock",
        latency_seconds=0.1,
    )

    return llm


def test_parse_plan():
    plan = parse_plan(PLAN)

    assert len(plan.tasks) == 2
    assert plan.tasks[0].tickers == ["NVDA"]


def test_parse_markdown_json():
    plan = parse_plan("```json\n" + PLAN + "\n```")

    assert len(plan.tasks) == 2


def test_planner():
    plan = create_research_plan(
        "Compare NVIDIA and AMD",
        mock_llm(),
    )

    assert len(plan.tasks) == 2


def test_planner_fallback():
    plan = create_research_plan(
        "NVIDIA risks",
        mock_llm("Invalid JSON"),
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].query == "NVIDIA risks"


def test_planner_budget_exhaustion():
    llm = Mock()
    llm.generate.side_effect = BudgetExceededError(
        "Budget exhausted"
    )

    with pytest.raises(BudgetExceededError):
        create_research_plan("NVIDIA risks", llm)


def test_search_request_filters():
    request = SearchRequest(
        query="Export controls",
        tickers=["nvda"],
        fiscal_years=[2024],
    )

    filters = request.build_filters()

    assert filters.tickers == ("NVDA",)
    assert filters.fiscal_years == (2024,)


def test_search_tool():
    retriever = Mock()
    retriever.search.return_value = []

    tools = SECTools([], retriever)

    results = tools.search_chunks(
        SearchRequest(query="Export controls")
    )

    assert results == []
    retriever.search.assert_called_once()


def test_graph_executes_research_tasks():
    tools = Mock()
    tools.search_chunks.return_value = []

    graph = build_agent_graph(tools, mock_llm())

    result = graph.invoke({
        "question": "Compare NVIDIA and AMD"
    })

    assert result["completed_tasks"] == [0, 1]
    assert result["status"] == "no_evidence"
    assert tools.search_chunks.call_count == 2


def test_graph_deduplicates_evidence():
    evidence = Mock()
    evidence.chunk.chunk_id = "chunk-1"

    tools = Mock()
    tools.search_chunks.return_value = [evidence]

    graph = build_agent_graph(tools, mock_llm())

    result = graph.invoke({
        "question": "Compare NVIDIA and AMD"
    })

    assert len(result["evidence"]) == 1
    assert result["status"] == "researched"


def test_graph_handles_tool_failure():
    tools = Mock()
    tools.search_chunks.side_effect = [
        RuntimeError("Unavailable"),
        [],
    ]

    graph = build_agent_graph(tools, mock_llm())

    result = graph.invoke({
        "question": "Compare NVIDIA and AMD"
    })

    assert len(result["errors"]) == 1
    assert result["completed_tasks"] == [1]


def test_graph_stops_on_budget_exhaustion():
    llm = Mock()
    llm.generate.side_effect = BudgetExceededError(
        "Budget exhausted"
    )

    tools = Mock()

    graph = build_agent_graph(tools, llm)

    result = graph.invoke({
        "question": "NVIDIA risks"
    })

    assert result["status"] == "budget_exhausted"
    tools.search_chunks.assert_not_called()
