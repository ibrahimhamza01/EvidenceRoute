from datetime import date

import pyarrow.parquet as pq
import pytest

from evidenceroute.ingestion.parser import ParsingError, detect_item_boundaries
from evidenceroute.ingestion.processor import chunk_sections, write_parquet
from evidenceroute.schemas import Filing, Section


def sample_filing():
    return Filing(document_id="doc", company_name="Example", ticker="EX", cik="0000000001",
                  accession_number="0000000001-24-000001", form_type="10-K", fiscal_year=2024,
                  report_period_end=date(2024, 12, 31), filing_date=date(2025, 2, 1),
                  primary_document="example.htm", source_url="https://www.sec.gov/example")


def test_chunk_offsets_and_determinism():
    filing = sample_filing()
    section = Section(document_id="doc", section_id="item_1a", section_name="Risk Factors",
                      text="Risk paragraph one.\n" * 120, source_url=filing.source_url)
    a = chunk_sections([section], filing, max_chars=150, overlap_chars=30)
    b = chunk_sections([section], filing, max_chars=150, overlap_chars=30)
    assert a == b
    assert len(a) > 1
    assert a[0].char_start == 0
    assert a[-1].char_end == len(section.text)
    assert all(section.text[c.char_start:c.char_end] == c.text for c in a)
    assert len({c.chunk_id for c in a}) == len(a)


def test_reject_bad_chunk_parameters():
    with pytest.raises(ValueError):
        chunk_sections([], sample_filing(), max_chars=100, overlap_chars=100)


def test_parquet_round_trip(tmp_path):
    filing = sample_filing()
    sections = [Section(document_id="doc", section_id="item_1", section_name="Business",
                        text="A business description " * 20, source_url=filing.source_url)]
    chunks = chunk_sections(sections, filing)
    write_parquet(sections, chunks, tmp_path)
    assert pq.read_table(tmp_path / "sections.parquet").num_rows == 1
    assert pq.read_table(tmp_path / "chunks.parquet").num_rows == len(chunks)


def test_reject_incomplete_heading_sequence():
    with pytest.raises(ParsingError):
        detect_item_boundaries(["ITEM 1. BUSINESS", "ITEM 1A. RISK FACTORS"])
