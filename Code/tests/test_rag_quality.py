"""Tests for Standard RAG quality metrics."""

import pytest

from evidenceroute.evaluation.rag_quality import (
    recall_at_k,
    reciprocal_rank_at_k,
    citation_id_precision,
)
from evidenceroute.schemas import Chunk, RetrievalResult


def make_result(chunk_id, score):
    text = "Supply chain risks are discussed."

    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="test-nvidia-2024",
        company_name="NVIDIA",
        ticker="NVDA",
        fiscal_year=2024,
        section_id="item-1a",
        section_name="Risk Factors",
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        source_url="https://www.sec.gov/",
    )

    return RetrievalResult(chunk=chunk, score=score)


@pytest.fixture
def ranked_results():
    return [
        make_result("chunk-a", 0.95),
        make_result("chunk-b", 0.85),
        make_result("chunk-c", 0.75),
    ]


def test_recall_at_k(ranked_results):
    relevant = {"chunk-b", "chunk-c"}

    assert recall_at_k(ranked_results, relevant, 1) == 0
    assert recall_at_k(ranked_results, relevant, 2) == 0.5
    assert recall_at_k(ranked_results, relevant, 3) == 1.0


def test_reciprocal_rank_at_k(ranked_results):
    relevant = {"chunk-c"}

    assert reciprocal_rank_at_k(ranked_results, relevant, 2) == 0
    assert reciprocal_rank_at_k(ranked_results, relevant, 3) == pytest.approx(1 / 3)


def test_citation_precision_all_valid(ranked_results):
    answer = "Risks are reported [chunk:chunk-a] [chunk:chunk-b]."

    assert citation_id_precision(answer, ranked_results) == 1.0


def test_citation_precision_partially_valid(ranked_results):
    answer = "Risks are reported [chunk:chunk-a] [chunk:fake-id]."

    assert citation_id_precision(answer, ranked_results) == 0.5


def test_citation_precision_no_citations(ranked_results):
    assert citation_id_precision("No citations.", ranked_results) == 0.0


def test_duplicate_citations_count_once(ranked_results):
    answer = "First [chunk:chunk-a]. Again [chunk:chunk-a]."

    assert citation_id_precision(answer, ranked_results) == 1.0


def test_invalid_k(ranked_results):
    with pytest.raises(ValueError):
        recall_at_k(ranked_results, {"chunk-a"}, 0)

    with pytest.raises(ValueError):
        reciprocal_rank_at_k(ranked_results, {"chunk-a"}, 0)
