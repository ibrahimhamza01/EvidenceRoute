
"""Offline RAG regression tests using verified SEC filing excerpts."""

import pytest

from evidenceroute.schemas import Chunk, RetrievalResult
from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.standard_rag import (
    ABSTENTION_MESSAGE,
    generate_rag_answer,
)


# Verified excerpts from gold_evidence.csv:
# NVIDIA: q0001
# AMD: q0011
#
# These are SEC-sourced excerpts, not full processed chunks.

SEC_CASES = [
    {
        "ticker": "NVDA",
        "company": "NVIDIA",
        "year": 2024,
        "document_id": "0001045810_0001045810-24-000029",
        "chunk_id": "8b7deb3e2b688baf905dd233",
        "quote": (
            "As of the end of fiscal year 2024, we had "
            "approximately 29,600 employees in 36 countries"
        ),
        "question": (
            "How many employees did NVIDIA report at the end "
            "of fiscal year 2024, and in how many countries?"
        ),
        "answer": (
            "NVIDIA had approximately 29,600 employees "
            "in 36 countries."
        ),
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/1045810/"
            "000104581024000029/nvda-20240128.htm"
        ),
        "char_start": 40259,
        "char_end": 40347,
    },
    {
        "ticker": "AMD",
        "company": "AMD",
        "year": 2023,
        "document_id": "0000002488_0000002488-24-000012",
        "chunk_id": "213b1ac2a12c4e975213c8fd",
        "quote": (
            "As of December 30, 2023, we had approximately "
            "26,000 employees in our global workforce."
        ),
        "question": (
            "Roughly how many employees did AMD have "
            "as of December 30, 2023?"
        ),
        "answer": (
            "AMD had approximately 26,000 employees."
        ),
        "source_url": (
            "https://www.sec.gov/Archives/edgar/data/2488/"
            "000000248824000012/amd-20231230.htm"
        ),
        "char_start": 54135,
        "char_end": 54222,
    },
]


class MockLLMClient:
    """Simulate LLM responses without making API calls."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1

        return LLMResult(
            text=self.answer,
            model_name="mock-model",
            latency_seconds=0.01,
            input_tokens=50,
            output_tokens=20,
        )


def make_evidence(case):
    """Convert a verified SEC excerpt into a RetrievalResult."""

    chunk = Chunk(
        chunk_id=case["chunk_id"],
        document_id=case["document_id"],
        company_name=case["company"],
        ticker=case["ticker"],
        fiscal_year=case["year"],
        section_id="item_1",
        section_name="Business",
        chunk_index=0,
        text=case["quote"],
        char_start=case["char_start"],
        char_end=case["char_end"],
        source_url=case["source_url"],
    )

    return [
        RetrievalResult(
            chunk=chunk,
            score=0.95,
        )
    ]


# TEST 1 and 2: NVIDIA and AMD valid answers
@pytest.mark.parametrize(
    "case",
    SEC_CASES,
    ids=["nvidia-fy2024", "amd-fy2023"],
)
def test_verified_sec_excerpt_generates_cited_answer(case):

    answer = (
        f'{case["answer"]} '
        f'[chunk:{case["chunk_id"]}]'
    )

    client = MockLLMClient(answer)

    response = generate_rag_answer(
        case["question"],
        make_evidence(case),
        client,
    )

    assert response.strategy == "rag"
    assert response.answer == answer
    assert len(response.citations) == 1

    assert response.citations[0].chunk_id == case["chunk_id"]
    assert response.citations[0].source_url == case["source_url"]

    assert response.verification_status == "not_checked"
    assert client.calls == 1


# TEST 3 and 4: Fabricated citations
@pytest.mark.parametrize(
    "case",
    SEC_CASES,
    ids=["nvidia-fy2024", "amd-fy2023"],
)
def test_fabricated_citation_is_rejected(case):

    client = MockLLMClient(
        "An unsupported statement [chunk:fake-source]."
    )

    response = generate_rag_answer(
        case["question"],
        make_evidence(case),
        client,
    )

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert response.verification_status == "unsupported"


# TEST 5 and 6: Missing SEC evidence
@pytest.mark.parametrize(
    "case",
    SEC_CASES,
    ids=["nvidia-fy2024", "amd-fy2023"],
)
def test_missing_evidence_abstains(case):

    client = MockLLMClient("An invented answer.")

    response = generate_rag_answer(
        case["question"],
        [],
        client,
    )

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert client.calls == 0


# TEST 7: Valid citation does not guarantee factual accuracy
def test_valid_citation_does_not_imply_factual_verification():

    case = SEC_CASES[0]

    client = MockLLMClient(
        "NVIDIA had 100 million employees "
        f'[chunk:{case["chunk_id"]}].'
    )

    response = generate_rag_answer(
        case["question"],
        make_evidence(case),
        client,
    )

    # Citation ID is valid, but the claim is false.
    # Our current system checks citation provenance,
    # not factual support.

    assert response.verification_status == "not_checked"
    assert len(response.citations) == 1
