"""Validate loading, schema, metadata and duplicate handling for all canonical chunks."""
import pytest
from evidenceroute.retrieval import ChunkCorpus, CorpusError
from evidenceroute.ingestion.downloader import load_canonical_manifest
from evidenceroute.schemas import Chunk


def test_all_chunks_match_canonical_manifest(full_corpus):
    filings = {f.document_id: f for f in load_canonical_manifest()}
    assert {c.document_id for c in full_corpus} == set(filings)
    for chunk in full_corpus:
        Chunk.model_validate(chunk.model_dump())
        filing = filings[chunk.document_id]
        assert (chunk.ticker, chunk.fiscal_year, chunk.company_name, chunk.source_url) == (
            filing.ticker, filing.fiscal_year, filing.company_name, filing.source_url)
        assert chunk.char_end - chunk.char_start == len(chunk.text)


def test_corpus_rejects_conflicting_duplicate_ids(full_corpus):
    first = next(iter(full_corpus))
    with pytest.raises(CorpusError):
        ChunkCorpus([first, first.model_copy(update={"text": first.text + " altered"})])


def test_corpus_collapses_duplicates_independent_of_order(full_corpus):
    chunks = list(full_corpus)
    doubled = ChunkCorpus(chunks + list(reversed(chunks)))
    assert len(doubled) == 24766
    assert doubled.fingerprint() == full_corpus.fingerprint()
