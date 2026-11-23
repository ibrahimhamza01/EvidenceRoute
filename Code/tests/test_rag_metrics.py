"""Tests for Standard RAG performance metrics."""

from unittest.mock import Mock

from evidenceroute.research.metrics import RAGMetrics
from evidenceroute.research.pipeline import run_standard_rag


def test_metrics_dataclass():
    metrics = RAGMetrics(
        retrieval_seconds=0.1,
        generation_seconds=0.2,
        total_seconds=0.3,
    )

    assert metrics.retrieval_seconds == 0.1
    assert metrics.generation_seconds == 0.2
    assert metrics.total_seconds == 0.3


def test_pipeline_reports_metrics():
    retriever = Mock()
    retriever.search.return_value = []

    llm = Mock()
    collected = []

    response = run_standard_rag(
        "What risks does NVIDIA face?",
        retriever,
        llm,
        metrics_callback=collected.append,
    )

    assert response.strategy == "rag"
    assert len(collected) == 1

    metrics = collected[0]

    assert isinstance(metrics, RAGMetrics)
    assert metrics.retrieval_seconds >= 0
    assert metrics.generation_seconds >= 0
    assert metrics.total_seconds >= metrics.retrieval_seconds
    llm.generate.assert_not_called()
