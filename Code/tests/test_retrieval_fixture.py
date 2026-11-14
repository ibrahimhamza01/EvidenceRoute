"""The development chunks are genuine, schema-valid passages from the four pilot filings."""
import csv
import importlib.util
from pathlib import Path

import pytest

from retrieval_testdata import FIXTURE_PATH
from evidenceroute.retrieval import ChunkCorpus, CorpusError
from evidenceroute.schemas import Chunk

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/filings_manifest.csv"
BUILD_SCRIPT = ROOT / "Code/scripts/build_retrieval_dev_chunks.py"


def load_manifest_rows():
    with MANIFEST.open(encoding="utf-8") as stream:
        return {row["document_id"]: row for row in csv.DictReader(stream)}


def test_fixture_is_valid_and_matches_filing_metadata(dev_corpus):
    manifest = load_manifest_rows()
    assert len(dev_corpus) == 32
    assert {c.document_id for c in dev_corpus} == set(manifest)
    for chunk in dev_corpus:
        Chunk.model_validate(chunk.model_dump())
        assert chunk.chunk_id.startswith("dev-")
        assert chunk.char_end - chunk.char_start == len(chunk.text)
        row = manifest[chunk.document_id]
        assert chunk.ticker == row["ticker"]
        assert chunk.fiscal_year == int(row["fiscal_year"])
        assert chunk.company_name == row["company_name"]
        assert chunk.source_url == row["source_url"]
        assert chunk.section_id in {"item_1", "item_1a", "item_7", "item_8", "item_15"}


def test_fixture_file_is_canonically_ordered():
    lines = FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
    records = [Chunk.model_validate_json(line) for line in lines]
    keys = [(c.document_id, c.section_id, c.char_start) for c in records]
    assert keys == sorted(keys)


def test_corpus_rejects_conflicting_duplicate_ids(dev_corpus):
    first = dev_corpus[0]
    clone = first.model_copy(update={"text": first.text + " altered"})
    with pytest.raises(CorpusError):
        ChunkCorpus([first, clone])


def test_corpus_collapses_exact_duplicates_and_is_order_independent(dev_corpus):
    chunks = list(dev_corpus)
    doubled = ChunkCorpus(chunks + list(reversed(chunks)))
    assert len(doubled) == len(dev_corpus)
    assert doubled.fingerprint() == dev_corpus.fingerprint()


def test_fixture_rebuilds_from_raw_filings():
    """Re-derive every passage from the original HTML when the filings are downloaded."""
    pytest.importorskip("bs4")
    raw = ROOT / "data/raw/sec-filings"
    if not raw.is_dir() or len(list(raw.glob("*/*/*.htm"))) < 4:
        pytest.skip("Pilot raw SEC filings are unavailable")
    spec = importlib.util.spec_from_file_location("build_retrieval_dev_chunks", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = module.serialize(module.build_chunks(MANIFEST, ROOT / "data/raw"))
    assert rebuilt == FIXTURE_PATH.read_text(encoding="utf-8")
