
from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.standard_rag import (
    ABSTENTION_MESSAGE,
    generate_rag_answer,
)
from evidenceroute.schemas import (
    Chunk,
    ResearchResponse,
    RetrievalResult,
)


class MockLLMClient:
    """Mock client that returns a predetermined response."""

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


def test_returns_research_response():
    client = MockLLMClient(
        "The supplied evidence describes a risk."
    )

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert isinstance(result, ResearchResponse)
    assert result.strategy == "rag"
    assert result.question == "What risks were reported?"
    assert result.answer == (
        "The supplied evidence describes a risk."
    )


def test_empty_evidence_abstains():
    client = MockLLMClient("This should never be returned.")

    result = generate_rag_answer(
        "What risks were reported?",
        [],
        client,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []
    assert client.calls == 0


def test_empty_llm_response_abstains():
    client = MockLLMClient("   ")

    result = generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert result.answer == ABSTENTION_MESSAGE
    assert result.citations == []


def test_prompt_reaches_llm():
    class InspectingClient(MockLLMClient):
        def generate(self, prompt):
            assert "What risks were reported?" in prompt
            assert "test-risk-001" in prompt
            return super().generate(prompt)

    client = InspectingClient("A grounded answer.")

    generate_rag_answer(
        "What risks were reported?",
        make_evidence(),
        client,
    )

    assert client.calls == 1
