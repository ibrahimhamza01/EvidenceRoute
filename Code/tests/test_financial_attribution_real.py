"""Regression tests for financial-statement cross-references in SEC filings."""

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


@pytest.mark.parametrize("ticker", ["NVDA", "JPM", "CVX"])
@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_real_filing_financial_attribution(ticker, year):
    if not MANIFEST.exists():
        pytest.skip("Full SEC manifest unavailable")

    filing = next(
        (
            f for f in load_manifest(MANIFEST)
            if f.ticker == ticker and f.fiscal_year == year
        ),
        None,
    )
    if filing is None:
        pytest.fail(f"Missing manifest entry: {ticker} FY{year}")

    path = Path("data/raw") / relative_filing_path(filing)
    if not path.is_file():
        pytest.skip("Original SEC HTML unavailable")

    sections = extract_sections(
        path,
        filing.document_id,
        filing.source_url,
        include_item_1c=True,
        resolve_financial_references=True,
    )

    by_id = {s.section_id: s for s in sections}

    assert len(by_id) == len(sections)
    assert {"item_7", "item_8", "item_15"} <= set(by_id)

    item7 = by_id["item_7"].text
    item8 = by_id["item_8"].text
    item15 = by_id["item_15"].text

    assert len(item7) > 10000
    assert len(item8) > 10000
    assert 100 < len(item15) < 50000

    # Confirm that the recovered Item 8 has actual statements,
    # not just a short SEC cross-reference.
    normalized = re.sub(r"[^a-z0-9]", "", item8.casefold())

    assert "consolidatedbalancesheet" in normalized
    assert "cashflows" in normalized
    assert any(
        name in normalized
        for name in (
            "consolidatedstatementofincome",
            "consolidatedstatementsofincome",
            "consolidatedstatementsofoperations",
        )
    )

    assert sum(
        " | " in line for line in item8.splitlines()
    ) > 100

    # A substantial financial-statement sample should not also
    # be duplicated within the exhibit section.
    balance_heading = re.search(
        r"consolidated balance sheets?",
        item8,
        re.I,
    )
    assert balance_heading is not None
    sample = item8[
        balance_heading.start():balance_heading.start() + 400
    ]
    assert sample not in item15

    chunks = chunk_sections(sections, filing)

    assert len(chunks) > 50
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    for chunk in chunks:
        section = by_id[chunk.section_id]
        assert (
            section.text[chunk.char_start:chunk.char_end]
            == chunk.text
        )


@pytest.mark.parametrize("year", [2023, 2024])
def test_nvidia_pilot_mode_retains_reference_only_item8(year):
    if not MANIFEST.exists():
        pytest.skip("Full SEC manifest unavailable")

    filing = next(
        f for f in load_manifest(MANIFEST)
        if f.ticker == "NVDA" and f.fiscal_year == year
    )
    path = Path("data/raw") / relative_filing_path(filing)

    if not path.is_file():
        pytest.skip("Original SEC HTML unavailable")

    sections = extract_sections(
        path,
        filing.document_id,
        filing.source_url,
    )

    by_id = {s.section_id: s for s in sections}

    # Pilot mode must not silently adopt the full-corpus
    # financial reattribution behavior.
    assert len(by_id["item_8"].text) < 2000
