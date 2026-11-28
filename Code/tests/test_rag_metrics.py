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


def test_metrics_record_retrieval_failure():
    retriever = Mock()
    retriever.search.side_effect = RuntimeError("Index unavailable")

    llm = Mock()
    collected = []

    response = run_standard_rag(
        "What risks does NVIDIA report?",
        retriever,
        llm,
        metrics_callback=collected.append,
    )

    assert response.citations == []
    assert len(collected) == 1
    assert collected[0].status == "retrieval_error"
    assert collected[0].evidence_count == 0
    assert collected[0].citation_count == 0
    assert collected[0].generation_seconds == 0
    llm.generate.assert_not_called()


def test_metrics_record_no_evidence():
    retriever = Mock()
    retriever.search.return_value = []

    collected = []

    response = run_standard_rag(
        "What risks are reported?",
        retriever,
        Mock(),
        metrics_callback=collected.append,
    )

    assert response.citations == []
    assert collected[0].status == "no_evidence"
    assert collected[0].evidence_count == 0


def test_metrics_record_missing_retrieval_results():
    retriever = Mock()
    retriever.search.return_value = None

    collected = []

    run_standard_rag(
        "What risks are reported?",
        retriever,
        Mock(),
        metrics_callback=collected.append,
    )

    assert collected[0].status == "no_evidence"
    assert collected[0].evidence_count == 0


def test_metrics_record_llm_token_usage(monkeypatch):
    from evidenceroute.research.llm_client import LLMResult
    from evidenceroute.schemas import ResearchResponse

    retriever = Mock()
    retriever.search.return_value = [Mock()]
    collected = []

    def fake_generate(**kwargs):
        kwargs["llm_result_callback"](
            LLMResult(
                text="Test answer",
                model_name="claude-sonnet-4-6",
                latency_seconds=0.25,
                input_tokens=120,
                output_tokens=80,
            )
        )

        return ResearchResponse(
            question=kwargs["question"],
            answer="Test answer",
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    monkeypatch.setattr(
        "evidenceroute.research.pipeline.generate_rag_answer",
        fake_generate,
    )

    run_standard_rag(
        "What risks does NVIDIA face?",
        retriever,
        Mock(),
        metrics_callback=collected.append,
    )

    metrics = collected[0]

    assert metrics.model_name == "claude-sonnet-4-6"
    assert metrics.input_tokens == 120
    assert metrics.output_tokens == 80
    assert metrics.total_tokens == 200
    assert metrics.llm_calls == 1
    assert metrics.llm_latency_seconds == 0.25


def test_metrics_no_evidence_has_zero_llm_calls():
    retriever = Mock()
    retriever.search.return_value = []

    llm = Mock()
    collected = []

    run_standard_rag(
        "What risks are reported?",
        retriever,
        llm,
        metrics_callback=collected.append,
    )

    metrics = collected[0]

    assert metrics.llm_calls == 0
    assert metrics.input_tokens is None
    assert metrics.output_tokens is None
    assert metrics.total_tokens is None
    assert metrics.model_name is None

    llm.generate.assert_not_called()


def test_metrics_missing_token_usage(monkeypatch):
    from evidenceroute.research.llm_client import LLMResult
    from evidenceroute.schemas import ResearchResponse

    retriever = Mock()
    retriever.search.return_value = [Mock()]
    collected = []

    def fake_generate(**kwargs):
        kwargs["llm_result_callback"](
            LLMResult(
                text="Test answer",
                model_name="mock",
                latency_seconds=0.1,
                input_tokens=100,
                output_tokens=None,
            )
        )

        return ResearchResponse(
            question=kwargs["question"],
            answer="Test answer",
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    monkeypatch.setattr(
        "evidenceroute.research.pipeline.generate_rag_answer",
        fake_generate,
    )

    run_standard_rag(
        "What risks are reported?",
        retriever,
        Mock(),
        metrics_callback=collected.append,
    )

    metrics = collected[0]

    assert metrics.input_tokens == 100
    assert metrics.output_tokens is None
    assert metrics.total_tokens is None
    assert metrics.llm_calls == 1


def test_metrics_record_generation_failure(monkeypatch):
    from evidenceroute.research.standard_rag import (
        SERVICE_ERROR_MESSAGE,
    )
    from evidenceroute.schemas import ResearchResponse

    retriever = Mock()
    retriever.search.return_value = [Mock()]
    collected = []

    def fake_generate(**kwargs):
        return ResearchResponse(
            question=kwargs["question"],
            answer=SERVICE_ERROR_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    monkeypatch.setattr(
        "evidenceroute.research.pipeline.generate_rag_answer",
        fake_generate,
    )

    run_standard_rag(
        "What risks are reported?",
        retriever,
        Mock(),
        metrics_callback=collected.append,
    )

    metrics = collected[0]

    assert metrics.status == "generation_error"
    assert metrics.llm_calls == 1
    assert metrics.total_tokens is None
    assert metrics.citation_count == 0
