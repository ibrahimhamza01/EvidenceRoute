"""Regression coverage for topic-organized SEC Form 10-K filings."""

from pathlib import Path
import re

import pytest

from evidenceroute.ingestion.downloader import (
    load_manifest,
    relative_filing_path,
)
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.ingestion.processor import chunk_sections
from evidenceroute.ingestion.topical_parser import (
    extract_topical_sections,
    is_topical_form10k,
)


MANIFEST = Path("data/filings_manifest.csv")
REQUIRED = {"item_1", "item_1a", "item_7", "item_8", "item_15"}


def test_topical_detector_requires_cross_reference_index():
    blocks = [
        "Management's Discussion and Analysis",
        "Financial Statements and Supplemental Details",
    ]
    assert not is_topical_form10k(blocks)

    blocks.append("Form 10-K Cross-Reference Index")
    assert is_topical_form10k(blocks)


def test_topical_parser_rejects_ambiguous_headings():
    blocks = [
        "Fundamentals of Our Business",
        "Management's Discussion and Analysis",
        "Risk Factors",
        "Risk Factors",
        "Financial Statements and Supplemental Details",
        "Exhibits",
        "Form 10-K Cross-Reference Index",
    ]

    with pytest.raises(ValueError, match="riskfactors"):
        extract_topical_sections(
            blocks,
            "test_document",
            "https://www.sec.gov/",
            include_item_1c=True,
        )


@pytest.mark.parametrize(
    "year,has_cybersecurity",
    [(2022, False), (2023, True), (2024, True)],
)
def test_intel_real_filing_topical_extraction(year, has_cybersecurity):
    if not MANIFEST.exists():
        pytest.skip("Full manifest not available")

    filing = next(
        f for f in load_manifest(MANIFEST)
        if f.ticker == "INTC" and f.fiscal_year == year
    )

    path = Path("data/raw") / relative_filing_path(filing)
    if not path.is_file():
        pytest.skip("Original Intel SEC filing unavailable")

    sections = extract_sections(
        path,
        filing.document_id,
        filing.source_url,
        include_item_1c=True,
        resolve_financial_references=True,
    )

    by_id = {s.section_id: s for s in sections}

    assert len(by_id) == len(sections)
    assert REQUIRED <= set(by_id)
    assert ("item_1c" in by_id) == has_cybersecurity
    assert len(sections) == (6 if has_cybersecurity else 5)

    for item in REQUIRED:
        assert len(by_id[item].text) > 10000

    financial = re.sub(
        r"[^a-z0-9]", "", by_id["item_8"].text.lower()
    )
    assert "consolidatedbalancesheets" in financial
    assert "consolidatedstatementsofcashflows" in financial
    assert (
        "consolidatedstatementsofincome" in financial
        or "consolidatedstatementsofoperations" in financial
    )

    chunks = chunk_sections(sections, filing)
    assert len(chunks) > 100
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    for chunk in chunks:
        section = by_id[chunk.section_id]
        assert (
            section.text[chunk.char_start:chunk.char_end]
            == chunk.text
        )
