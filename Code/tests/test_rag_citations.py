
from evidenceroute.research.citations import (
    extract_citation_ids,
    validate_citations,
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


def test_extract_citation_ids():
    answer = (
        "Risk described [chunk:test-risk-001]. "
        "Another claim [chunk:test-risk-002]."
    )

    assert extract_citation_ids(answer) == [
        "test-risk-001",
        "test-risk-002",
    ]


def test_duplicate_citations():
    answer = (
        "First [chunk:test-risk-001]. "
        "Second [chunk:test-risk-001]."
    )

    assert extract_citation_ids(answer) == ["test-risk-001"]


def test_valid_citation():
    answer = "Risk described [chunk:test-risk-001]."

    valid, invalid = validate_citations(
        answer,
        make_evidence(),
    )

    assert len(valid) == 1
    assert valid[0].chunk_id == "test-risk-001"
    assert invalid == []


def test_fabricated_citation():
    answer = "Invented claim [chunk:fake-999]."

    valid, invalid = validate_citations(
        answer,
        make_evidence(),
    )

    assert valid == []
    assert invalid == ["fake-999"]


def test_no_citations():
    valid, invalid = validate_citations(
        "An answer without citations.",
        make_evidence(),
    )

    assert valid == []
    assert invalid == []
