import pytest

from retrieval_testdata import FIXTURE_PATH


@pytest.fixture(scope="session")
def dev_corpus():
    from evidenceroute.retrieval import load_chunks_jsonl

    return load_chunks_jsonl(FIXTURE_PATH)


@pytest.fixture(scope="session")
def chunk_by_anchor(dev_corpus):
    """Find the unique fixture chunk for (document_id, text prefix)."""

    def find(document_id: str, prefix: str):
        hits = [c for c in dev_corpus if c.document_id == document_id and c.text.startswith(prefix)]
        assert len(hits) == 1, (document_id, prefix, len(hits))
        return hits[0]

    return find
