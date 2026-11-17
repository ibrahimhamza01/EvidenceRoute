"""Exercise general RAG citation handling with actual chunks from every canonical filing."""
import pytest

from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.standard_rag import generate_rag_answer, ABSTENTION_MESSAGE
from evidenceroute.schemas import RetrievalResult


class MockLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return LLMResult(text=self.text, model_name="mock", latency_seconds=0.0,
                         input_tokens=0, output_tokens=0)


def test_citations_preserve_provenance_for_all_75_filings(full_corpus):
    filings = {}
    for chunk in full_corpus:
        filings.setdefault(chunk.document_id, chunk)
    assert len(filings) == 75
    for chunk in filings.values():
        question = "Quote the opening sentence of this section."
        client = MockLLM(f"{chunk.text.splitlines()[0]} [chunk:{chunk.chunk_id}]")
        response = generate_rag_answer(question, [RetrievalResult(chunk=chunk, score=1.0)], client)
        assert response.citations[0].chunk_id == chunk.chunk_id
        assert response.citations[0].source_url == chunk.source_url
        assert response.verification_status == "not_checked"
        assert client.calls == 1
        invalid = generate_rag_answer(question, [RetrievalResult(chunk=chunk, score=1.0)],
                                      MockLLM("Unsupported [chunk:unknown]"))
        assert invalid.answer == ABSTENTION_MESSAGE
        assert invalid.citations == []


@pytest.fixture(params=[2022, 2023, 2024], ids=lambda year: f"fy{year}")
def filing_evidence(request, full_corpus):
    """Use a real canonical chunk for each fiscal year, without benchmark answers."""
    chunk = next(c for c in full_corpus if c.fiscal_year == request.param)
    return [RetrievalResult(chunk=chunk, score=1.0)]


def test_verified_sec_excerpt_generates_cited_answer(filing_evidence):
    chunk = filing_evidence[0].chunk
    answer = f"{chunk.text.splitlines()[0]} [chunk:{chunk.chunk_id}]"
    client = MockLLM(answer)
    question = "Quote the opening line of the supplied section."

    response = generate_rag_answer(question, filing_evidence, client)

    assert response.strategy == "rag"
    assert response.question == question
    assert response.answer == answer
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.citations[0].source_url == chunk.source_url
    assert response.verification_status == "not_checked"
    assert client.calls == 1


def test_fabricated_citation_is_rejected(filing_evidence):
    client = MockLLM("An unsupported statement [chunk:fake-source].")

    response = generate_rag_answer("What does this section report?", filing_evidence, client)

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert response.verification_status == "unsupported"
    assert client.calls == 1


@pytest.mark.parametrize("question", [
    "What does the annual report say about revenue?",
    "What risks does the annual report describe?",
], ids=["financial-question", "risk-question"])
def test_missing_evidence_abstains(question):
    # This behavior must be tested even when canonical artifacts are unavailable.
    client = MockLLM("An invented answer.")

    response = generate_rag_answer(question, [], client)

    assert response.answer == ABSTENTION_MESSAGE
    assert response.citations == []
    assert client.calls == 0


def test_valid_citation_does_not_imply_factual_verification(filing_evidence):
    chunk = filing_evidence[0].chunk
    # A deliberately false claim with an authentic supplied chunk ID must never
    # be promoted to factually verified merely because citation lookup succeeds.
    answer = f"This filing is for fiscal year {chunk.fiscal_year + 100}. [chunk:{chunk.chunk_id}]"
    client = MockLLM(answer)

    response = generate_rag_answer("Which fiscal year is this filing for?", filing_evidence, client)

    assert response.answer == answer
    assert response.verification_status == "not_checked"
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.citations[0].source_url == chunk.source_url
    assert client.calls == 1
