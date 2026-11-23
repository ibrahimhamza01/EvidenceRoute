"""Tests for retrieval-to-answer Standard RAG integration."""

from unittest.mock import Mock

import pytest

from evidenceroute.research.pipeline import run_standard_rag
from evidenceroute.research.standard_rag import ABSTENTION_MESSAGE
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import Chunk, RetrievalResult


def sample_evidence():
    chunk = Chunk(
        chunk_id="nvda-2024-risk-0",
        document_id="nvda-2024",
        company_name="NVIDIA",
        ticker="NVDA",
        fiscal_year=2024,
        section_id="item_1a",
        section_name="Risk Factors",
        chunk_index=0,
        text="NVIDIA faces supply chain risks.",
        char_start=0,
        char_end=32,
        source_url="https://www.sec.gov/example",
    )
    return RetrievalResult(chunk=chunk, score=0.95)


def test_pipeline_retrieves_and_generates_answer():
    retriever = Mock()
    retriever.search.return_value = [sample_evidence()]

    llm = Mock()
    llm.generate.return_value.text = (
        "NVIDIA faces supply chain risks [chunk:nvda-2024-risk-0]."
    )

    response = run_standard_rag(
        "What risks does NVIDIA face?",
        retriever,
        llm,
        top_k=5,
    )

    assert response.strategy == "rag"
    assert len(response.citations) == 1
    assert response.citations[0].ticker == "NVDA"

    retriever.search.assert_called_once_with(
        query="What risks does NVIDIA face?",
        top_k=5,
        filters=None,
    )
    llm.generate.assert_called_once()


def test_pipeline_abstains_without_evidence():
    retriever = Mock()
    retriever.search.return_value = []
    llm = Mock()

    response = run_standard_rag(
        "What is NVIDIA's revenue?",
        retriever,
        llm,
    )

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    llm.generate.assert_not_called()


def test_pipeline_forwards_metadata_filters():
    retriever = Mock()
    retriever.search.return_value = []
    llm = Mock()

    filters = ChunkFilter(
        tickers=["NVDA"],
        fiscal_years=[2024],
    )

    run_standard_rag(
        "What risks did NVIDIA report?",
        retriever,
        llm,
        filters=filters,
    )

    assert retriever.search.call_args.kwargs["filters"] == filters


@pytest.mark.parametrize("question", ["", "   "])
def test_pipeline_rejects_blank_questions(question):
    retriever = Mock()
    llm = Mock()

    with pytest.raises(ValueError):
        run_standard_rag(question, retriever, llm)

    retriever.search.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_pipeline_rejects_invalid_top_k(top_k):
    retriever = Mock()
    llm = Mock()

    with pytest.raises(ValueError):
        run_standard_rag(
            "What are NVIDIA's risks?",
            retriever,
            llm,
            top_k=top_k,
        )

    retriever.search.assert_not_called()
