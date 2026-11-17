"""Regression tests for SEC filings with cross-referenced financial sections."""

from pathlib import Path
import re

import pytest

from evidenceroute.ingestion.downloader import (
    load_manifest,
    relative_filing_path,
)
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.ingestion.processor import chunk_sections


MANIFEST = Path("data/full_filings_manifest.csv")


@pytest.mark.parametrize(
    "ticker,income_heading,cash_heading",
    [
        (
            "XOM",
            "consolidatedstatementofincome",
            "consolidatedstatementofcashflows",
        ),
        (
            "DE",
            "statementsofconsolidatedincome",
            "statementsofconsolidatedcashflows",
        ),
    ],
)
def test_referenced_financial_sections(
    ticker, income_heading, cash_heading
):
    if not MANIFEST.exists():
        pytest.skip("Full manifest not available")

    filing = next(
        f for f in load_manifest(MANIFEST)
        if f.ticker == ticker and f.fiscal_year == 2024
    )

    path = Path("data/raw") / relative_filing_path(filing)
    if not path.exists():
        pytest.skip("Original SEC HTML not available locally")

    sections = extract_sections(
        path,
        filing.document_id,
        filing.source_url,
        include_item_1c=True,
        resolve_financial_references=True,
    )

    by_id = {s.section_id: s for s in sections}

    assert len(sections) == 6
    assert "item_1c" in by_id
    assert len(by_id["item_7"].text) > 10000
    assert len(by_id["item_8"].text) > 10000

    financial_text = re.sub(
        r"[^a-z0-9]", "", by_id["item_8"].text.lower()
    )

    assert income_heading in financial_text
    assert "consolidatedbalancesheet" in financial_text
    assert cash_heading in financial_text
    assert "notestoconsolidatedfinancialstatements" in financial_text

    chunks = chunk_sections(sections, filing)
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    for chunk in chunks:
        section = by_id[chunk.section_id]
        assert (
            section.text[chunk.char_start:chunk.char_end]
            == chunk.text
        )
