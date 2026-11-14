
import pytest

from evidenceroute.research.prompts import build_rag_prompt
from evidenceroute.schemas import Chunk, RetrievalResult


def sample_evidence():
    chunk = Chunk(
        chunk_id="test-chunk-001",
        document_id="test-document-001",
        company_name="NVIDIA",
        ticker="NVDA",
        fiscal_year=2024,
        section_id="item-1a",
        section_name="Risk Factors",
        chunk_index=0,
        text="Example evidence passage for testing.",
        char_start=0,
        char_end=37,
        source_url="https://www.sec.gov/"
    )

    return [RetrievalResult(chunk=chunk, score=0.95)]


def test_question_in_prompt():
    prompt = build_rag_prompt(
        "What risks were reported?",
        sample_evidence()
    )
    assert "What risks were reported?" in prompt


def test_evidence_in_prompt():
    prompt = build_rag_prompt(
        "What risks were reported?",
        sample_evidence()
    )
    assert "Example evidence passage" in prompt
    assert "test-chunk-001" in prompt


def test_empty_evidence():
    prompt = build_rag_prompt(
        "What risks were reported?",
        []
    )
    assert "No evidence supplied." in prompt


def test_empty_question():
    with pytest.raises(ValueError):
        build_rag_prompt("   ", [])
