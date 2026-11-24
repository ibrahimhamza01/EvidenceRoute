"""Exercise the local corpus interface with a self-contained SEC-shaped snapshot."""

import csv

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evidenceroute.ingestion.processor import chunk_sections, write_parquet
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.retrieval import ChunkFilter, CorpusError
from evidenceroute.schemas import Filing, Section


@pytest.fixture
def snapshot(tmp_path):
    manifest = tmp_path / "filings_manifest.csv"
    filings = []
    sections = []
    chunks = []
    for ticker, cik, year in (
        ("BETA", "0000000002", 2024),
        ("ACME", "0000000001", 2024),
        ("ACME", "0000000001", 2023),
    ):
        accession = f"{cik}-{year % 100:02d}-000001"
        document_id = f"{cik}_{accession}"
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/report.htm"
        filing = Filing(
            document_id=document_id, company_name=f"{ticker} Corp", ticker=ticker, cik=cik,
            accession_number=accession, form_type="10-K", fiscal_year=year,
            report_period_end=f"{year}-09-30", filing_date=f"{year}-11-01",
            primary_document="report.htm", source_url=url,
        )
        items = [
            Section(document_id=document_id, section_id=section_id, section_name=name,
                    text=text, source_url=url)
            for section_id, name, text in (
                ("item_1", "Business", "Supply chain overview."),
                ("item_1a", "Risk Factors", "Supply chain disruptions. " * 200),
            )
        ]
        filings.append(filing)
        sections.extend(items)
        chunks.extend(chunk_sections(items, filing))

    rows = [filing.model_dump(mode="json", exclude_none=True) for filing in filings]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_parquet(sections, list(reversed(chunks)), tmp_path)
    return manifest, tmp_path, filings[1], sections[3], chunks


@pytest.fixture
def environment(snapshot):
    manifest, directory, *_ = snapshot
    return CorpusEnvironment(manifest_path=manifest, processed_dir=directory)


def test_inventory_and_filing_filters(environment, snapshot):
    _, _, expected, *_ = snapshot
    first = environment.list_companies(limit=1)
    assert (first.total, first.next_offset) == (2, 1)
    assert (first.items[0].ticker, first.items[0].fiscal_years) == ("ACME", (2023, 2024))
    second = environment.list_companies(offset=first.next_offset, limit=1)
    assert (second.items[0].ticker, second.items[0].fiscal_years) == ("BETA", (2024,))
    assert second.next_offset is None
    assert environment.list_filings(ticker=" acme ", fiscal_year=2024).items == (expected,)
    assert environment.list_filings(fiscal_year=2024).total == 2
    assert environment.list_filings(ticker="MISSING").total == 0
    assert environment.list_filings(offset=20).items == ()
    assert environment.get_filing(expected.document_id) == expected
    with pytest.raises(KeyError):
        environment.get_filing("unknown")


def test_section_windows_and_original_chunk_pages(environment, snapshot):
    _, _, filing, risk, chunks = snapshot
    document_id = filing.document_id
    first = environment.list_sections(document_id, limit=1)
    assert (first.items[0].section_id, first.total, first.next_offset) == ("item_1", 2, 1)
    window = environment.get_section(document_id, risk.section_id)
    tail = environment.get_section(document_id, risk.section_id, offset=window.next_offset)
    assert len(window.text) == 4_000
    assert window.text + tail.text == risk.text
    assert tail.next_offset is None
    assert (window.metadata.ticker, window.metadata.fiscal_year) == ("ACME", 2024)
    assert window.metadata.source_url == filing.source_url
    assert window.metadata.char_count == len(risk.text)
    expected = tuple(
        chunk for chunk in chunks
        if (chunk.document_id, chunk.section_id) == (document_id, risk.section_id)
    )
    page = environment.read_chunks(document_id, risk.section_id, limit=1)
    tail_page = environment.read_chunks(document_id, risk.section_id, offset=page.next_offset)
    assert page.total == window.metadata.chunk_count == len(expected)
    assert page.items + tail_page.items == expected
    assert tail_page.next_offset is None
    assert environment.read_chunks(document_id).items[1:] == expected
    with pytest.raises(KeyError):
        environment.read_chunks(document_id, "missing")


def test_keyword_search_is_scoped_and_deterministically_paginated(environment, snapshot):
    _, _, filing, risk, _ = snapshot
    filters = ChunkFilter(tickers="acme", fiscal_years=2024, section_ids=risk.section_id)
    page = environment.search_chunks("SUPPLY chain", filters=filters, limit=1)
    rest = environment.search_chunks("supply chain", filters=filters, offset=page.next_offset)
    hits = page.items + rest.items
    assert len(hits) == page.total == 4
    assert rest.next_offset is None
    assert all(hit.chunk.document_id == filing.document_id for hit in hits)
    assert all(hit.chunk.section_id == risk.section_id for hit in hits)
    assert all(hit.chunk.source_url == filing.source_url for hit in hits)
    assert [(-hit.score, hit.chunk.chunk_id) for hit in hits] == sorted(
        (-hit.score, hit.chunk.chunk_id) for hit in hits
    )
    assert environment.search_chunks("supply chain", filters=filters).items == hits
    assert environment.search_chunks("   ").items == ()
    assert environment.search_chunks("nonmatchingtoken", filters=filters).total == 0


def test_returned_records_cannot_mutate_the_snapshot(environment, snapshot):
    _, _, filing, risk, _ = snapshot
    environment.list_filings(ticker="ACME", fiscal_year=2024).items[0].ticker = "EDIT"
    assert environment.get_filing(filing.document_id).ticker == "ACME"
    filters = ChunkFilter(document_ids=filing.document_id, section_ids=risk.section_id)
    hit = environment.search_chunks("supply chain", filters=filters, limit=1).items[0]
    original = hit.chunk.text
    hit.chunk.text = "Edited"
    assert environment.search_chunks("supply chain", filters=filters, limit=1).items[0].chunk.text == original
    page = environment.read_chunks(filing.document_id, risk.section_id, limit=1)
    original = page.items[0].text
    page.items[0].text = "Edited"
    assert environment.read_chunks(filing.document_id, risk.section_id, limit=1).items[0].text == original


def test_get_chunk_returns_an_original_defensive_copy(environment, snapshot):
    *_, chunks = snapshot
    expected = chunks[0]
    chunk = environment.get_chunk(expected.chunk_id)
    assert chunk == expected
    chunk.text = "Edited"
    assert environment.get_chunk(expected.chunk_id) == expected
    with pytest.raises(KeyError, match="Unknown chunk ID"):
        environment.get_chunk("missing")


def test_response_limits_are_enforced(environment, snapshot):
    _, _, filing, risk, _ = snapshot
    with pytest.raises(ValueError, match="limit"):
        environment.list_filings(limit=26)
    with pytest.raises(ValueError, match="offset"):
        environment.list_companies(offset=-1)
    with pytest.raises(ValueError, match="limit"):
        environment.get_section(filing.document_id, risk.section_id, limit=4_001)
    with pytest.raises(ValueError, match="limit"):
        environment.read_chunks(filing.document_id, limit=11)
    with pytest.raises(ValueError, match="limit"):
        environment.search_chunks("risk", limit=11)
    with pytest.raises(ValueError, match="query"):
        environment.search_chunks("r" * 1_001)


@pytest.mark.parametrize("field,value,match", [
    ("source_url", "https://example.com/foreign.htm", "metadata conflicts"),
    ("text", "altered", "text conflicts"),
])
def test_inconsistent_processed_chunk_fails_at_load(snapshot, field, value, match):
    manifest, directory, *_ = snapshot
    path = directory / "chunks.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0][field] = value
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(CorpusError, match=match):
        CorpusEnvironment(manifest_path=manifest, processed_dir=directory)
