"""Integration tests for Standard RAG using real SEC filing chunks."""

from pathlib import Path

import pytest

from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.pipeline import run_standard_rag
from evidenceroute.retrieval.bm25 import BM25Retriever
from evidenceroute.retrieval.corpus import load_chunks_parquet
from evidenceroute.retrieval.filters import ChunkFilter


CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/processed/full-v1/chunks.parquet"
)


class MockCitingLLM:
    """Generate a citation using the retrieved chunk ID."""

    def generate(self, prompt: str) -> LLMResult:
        import re

        matches = re.findall(r"^Chunk ID:\s*(\S+)", prompt, re.MULTILINE)

        if not matches:
            raise AssertionError("No chunk citation markers found in prompt")

        return LLMResult(
            text=f"The SEC filing provides relevant evidence "
                 f"[chunk:{matches[0]}].",
            model_name="mock-llm",
            latency_seconds=0.001,
            input_tokens=100,
            output_tokens=20,
        )


@pytest.fixture(scope="module")
def real_retriever():
    if not CORPUS_PATH.exists():
        pytest.skip("Full SEC corpus not available locally")

    corpus = load_chunks_parquet(CORPUS_PATH)

    assert len(corpus) == 24766

    return BM25Retriever(corpus)


def test_real_corpus_retrieval(real_retriever):
    results = real_retriever.search(
        "supply chain semiconductor manufacturing risks",
        top_k=5,
    )

    assert results
    assert len(results) <= 5

    for result in results:
        assert result.chunk.chunk_id
        assert result.chunk.text
        assert result.chunk.source_url


def test_real_corpus_standard_rag(real_retriever):
    response = run_standard_rag(
        question="What semiconductor supply chain risks are reported?",
        retriever=real_retriever,
        llm_client=MockCitingLLM(),
        top_k=5,
    )

    assert response.strategy == "rag"
    assert response.citations
    assert response.citations[0].source_url
    assert "[chunk:" in response.answer


def test_real_corpus_metadata_filter(real_retriever):
    filters = ChunkFilter(tickers=["NVDA"])

    results = real_retriever.search(
        "supply chain risks",
        top_k=5,
        filters=filters,
    )

    assert results
    assert all(result.chunk.ticker == "NVDA" for result in results)
