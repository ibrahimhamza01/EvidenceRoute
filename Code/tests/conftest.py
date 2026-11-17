from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def full_corpus():
    from evidenceroute.retrieval import load_chunks
    path = Path(__file__).resolve().parents[2] / "data/processed/full-v1/chunks.parquet"
    if not path.is_file():
        pytest.skip("Canonical processed corpus unavailable")
    corpus = load_chunks(path)
    assert len(corpus) == 24766
    assert len({c.document_id for c in corpus}) == 75
    assert len({c.ticker for c in corpus}) == 25
    return corpus
