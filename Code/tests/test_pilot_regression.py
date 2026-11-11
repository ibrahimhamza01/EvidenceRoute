
"""Regression tests against the four real SEC 10-K pilot filings."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from evidenceroute.ingestion.downloader import (
    load_manifest,
    relative_filing_path,
)
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.ingestion.processor import chunk_sections

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw"
MANIFEST = ROOT / "data/filings_manifest.csv"
OUTPUT = ROOT / "data/processed/pilot-v1"

EXPECTED_ITEMS = {"item_1", "item_1a", "item_7", "item_8", "item_15"}

# Representative financial values from the original pilot filings.
# These catch missing tables or accidental truncation.
EXPECTED_FINANCIAL_ROWS = {
    ("NVDA", 2023): ("item_15", "3,389", "41,182", "4,368"),
    ("NVDA", 2024): ("item_15", "7,280", "65,728", "29,760"),
    ("AMD", 2023): ("item_8", "3,933", "67,885", "854"),
    ("AMD", 2024): ("item_8", "3,787", "69,226", "1,641"),
}


@pytest.fixture(scope="module")
def pilot_data():
    if not MANIFEST.is_file():
        pytest.skip("Pilot filing manifest is unavailable")

    filings = list(load_manifest(MANIFEST))

    if len(filings) != 4:
        pytest.skip("The four-filing pilot manifest is unavailable")

    if any(
        not (RAW_DIR / relative_filing_path(f)).is_file()
        for f in filings
    ):
        pytest.skip("Pilot raw SEC filings are unavailable")

    return filings


@pytest.mark.parametrize(
    "ticker,year",
    [
        ("NVDA", 2023),
        ("NVDA", 2024),
        ("AMD", 2023),
        ("AMD", 2024),
    ],
)
def test_real_filing_sections_and_financial_tables(
    pilot_data, ticker, year
):
    filing = next(
        f for f in pilot_data
        if f.ticker == ticker and f.fiscal_year == year
    )

    path = RAW_DIR / relative_filing_path(filing)

    sections = extract_sections(
        path,
        filing.document_id,
        filing.source_url,
    )

    by_item = {s.section_id: s for s in sections}

    assert set(by_item) == EXPECTED_ITEMS
    assert len(sections) == 5

    for section in sections:
        assert section.text.strip()
        assert section.document_id == filing.document_id
        assert section.source_url == filing.source_url

    item, cash, assets, income = EXPECTED_FINANCIAL_ROWS[
        (ticker, year)
    ]

    text = by_item[item].text

    assert "Cash and cash equivalents" in text
    assert "Total assets" in text
    assert "Net income" in text

    # Check the financial values occur in the correct statement section.
    assert cash in text
    assert assets in text
    assert income in text

    # Check the balance-sheet rows preserve their reported values.
    cash_line = next(
        line for line in text.splitlines()
        if line.strip().startswith("Cash and cash equivalents")
    )
    assets_line = next(
        line for line in text.splitlines()
        if line.strip().startswith("Total assets")
    )
    income_line = next(
        line for line in text.splitlines()
        if line.strip().startswith("Net income |")
    )

    assert cash in cash_line
    assert assets in assets_line
    assert income in income_line


def test_real_filing_chunk_offsets_and_determinism(pilot_data):
    for filing in pilot_data:
        path = RAW_DIR / relative_filing_path(filing)

        sections = extract_sections(
            path, filing.document_id, filing.source_url
        )

        first = chunk_sections(sections, filing)
        second = chunk_sections(sections, filing)

        assert first == second

        lookup = {s.section_id: s.text for s in sections}

        for chunk in first:
            assert (
                lookup[chunk.section_id][
                    chunk.char_start:chunk.char_end
                ]
                == chunk.text
            )


def test_exported_parquet_integrity():
    sections_path = OUTPUT / "sections.parquet"
    chunks_path = OUTPUT / "chunks.parquet"

    if not sections_path.is_file() or not chunks_path.is_file():
        pytest.skip("Generated pilot Parquet files are unavailable")

    sections = pq.read_table(sections_path).to_pylist()
    chunks = pq.read_table(chunks_path).to_pylist()

    assert len(sections) == 20
    assert len(chunks) == 994

    lookup = {
        (s["document_id"], s["section_id"]): s["text"]
        for s in sections
    }

    assert len(lookup) == 20
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)

    for chunk in chunks:
        key = (chunk["document_id"], chunk["section_id"])
        text = lookup[key]

        assert (
            text[chunk["char_start"]:chunk["char_end"]]
            == chunk["text"]
        )

def test_processor_generates_dataset_manifest(pilot_data, tmp_path):
    import hashlib
    import json

    from evidenceroute.ingestion.processor import write_parquet
    from evidenceroute.ingestion.downloader import load_manifest

    # Import the existing processing entry point.
    import importlib.util

    script = ROOT / "Code/scripts/process_filings.py"
    spec = importlib.util.spec_from_file_location("process_filings", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "pilot-v1"

    metadata = module.process(
        manifest=MANIFEST,
        raw_dir=RAW_DIR,
        output_dir=output,
        dataset="pilot",
        max_chars=1800,
        overlap_chars=200,
    )

    manifest_path = output / "dataset_manifest.json"
    assert manifest_path.is_file()
    assert (output / "sections.parquet").is_file()
    assert (output / "chunks.parquet").is_file()

    saved = json.loads(manifest_path.read_text())

    assert saved == metadata
    assert saved["dataset_name"] == "pilot-v1"
    assert saved["dataset_type"] == "pilot"
    assert saved["filing_count"] == 4
    assert saved["section_count"] == 20
    assert saved["chunk_count"] == 994

    for filename in ("sections.parquet", "chunks.parquet"):
        path = output / filename
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assert saved["files"][filename]["sha256"] == actual_hash

    assert len(saved["inputs"]) == 4


def test_s3_publish_is_idempotent_and_rejects_conflicts(
    pilot_data, tmp_path, monkeypatch
):
    import importlib.util
    import io
    import sys
    import types

    script = ROOT / "Code/scripts/process_filings.py"
    spec = importlib.util.spec_from_file_location(
        "process_filings_s3_test", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "pilot-v1"
    module.process(
        manifest=MANIFEST,
        raw_dir=RAW_DIR,
        output_dir=output,
        dataset="pilot",
        max_chars=1800,
        overlap_chars=200,
    )

    objects = {}

    class FakeClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}

    class FakeStreamingBody(io.BytesIO):
        def iter_chunks(self, chunk_size=1024 * 1024):
            while chunk := self.read(chunk_size):
                yield chunk

    class FakeS3:
        def get_object(self, *, Bucket, Key):
            key = (Bucket, Key)
            if key not in objects:
                raise FakeClientError("NoSuchKey")
            return {"Body": FakeStreamingBody(objects[key])}

        def put_object(
            self, *, Bucket, Key, Body, IfNoneMatch, Metadata
        ):
            assert IfNoneMatch == "*"
            key = (Bucket, Key)
            if key in objects:
                raise FakeClientError("PreconditionFailed")
            objects[key] = Body.read()

    fake_boto3 = types.SimpleNamespace(
        client=lambda service: FakeS3()
    )
    fake_exceptions = types.SimpleNamespace(
        ClientError=FakeClientError
    )

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(
        sys.modules, "botocore",
        types.ModuleType("botocore")
    )
    monkeypatch.setitem(
        sys.modules, "botocore.exceptions",
        fake_exceptions
    )

    kwargs = {
        "output_dir": output,
        "dataset": "pilot",
        "bucket": "test-bucket",
        "prefix": "EvidenceRoute",
    }

    first = module.publish_to_s3(**kwargs)
    assert len(first) == 3
    assert len(objects) == 3

    second = module.publish_to_s3(**kwargs)
    assert first == second
    assert len(objects) == 3

    key = (
        "test-bucket",
        "EvidenceRoute/processed/pilot-v1/chunks.parquet",
    )
    objects[key] = b"corrupted"

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        module.publish_to_s3(**kwargs)
