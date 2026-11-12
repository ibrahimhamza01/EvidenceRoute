
import pytest

from evidenceroute.research.llm_client import (
    LLMClientError,
    LLMResult,
)
from evidenceroute.research.standard_rag import (
    ABSTENTION_MESSAGE,
    SERVICE_ERROR_MESSAGE,
    generate_rag_answer,
)
from evidenceroute.schemas import Chunk, RetrievalResult


def make_evidence():
    chunk = Chunk(
        chunk_id="test-risk-001",
        document_id="test-nvidia-2024",
        company_name="NVIDIA",
        ticker="NVDA",
        fiscal_year=2024,
        section_id="item-1a",
        section_name="Risk Factors",
        chunk_index=0,
        text="Example SEC evidence passage.",
        char_start=0,
        char_end=29,
        source_url="https://www.sec.gov/",
    )
    return [RetrievalResult(chunk=chunk, score=0.95)]


class MockClient:
    def __init__(self, answer=None, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1

        if self.error:
            raise self.error

        return LLMResult(
            text=self.answer or "",
            model_name="mock-model",
            latency_seconds=0.01,
        )


def test_api_failure_returns_safe_response():
    client = MockClient(
        error=LLMClientError("API unavailable")
    )

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert result.answer == SERVICE_ERROR_MESSAGE
    assert result.citations == []


def test_timeout_returns_safe_response():
    client = MockClient(
        error=TimeoutError("Request timed out")
    )

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert result.answer == SERVICE_ERROR_MESSAGE
    assert result.citations == []


def test_fabricated_citation_is_rejected():
    client = MockClient(
        answer="Invented risk [chunk:fake-999]."
    )

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []
    assert result.verification_status == "unsupported"


def test_uncited_answer_abstains():
    client = MockClient(
        answer="The company reported a risk."
    )

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []


def test_empty_evidence_skips_llm():
    client = MockClient(answer="Invented answer")

    result = generate_rag_answer(
        "What risks were reported?",
        [],
        client,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert client.calls == 0


def test_empty_question_raises_error():
    client = MockClient(answer="Answer")

    with pytest.raises(ValueError):
        generate_rag_answer(
            " ",
            make_evidence(),
            client,
        )
