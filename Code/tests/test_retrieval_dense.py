"""Dense retrieval with Qdrant (embedded mode).

Most tests use the deterministic ``HashingEmbedder`` so they run offline and fast.
``test_sentence_transformer_*`` exercises a real SentenceTransformer model and is
skipped when the library or the model weights are unavailable.
"""
import random

import numpy as np
import pytest

from evidenceroute.retrieval import ChunkCorpus, ChunkFilter
from evidenceroute.retrieval.dense import (
    DenseIndexError,
    DenseRetriever,
    HashingEmbedder,
    point_id,
    to_qdrant_filter,
)
from evidenceroute.schemas import RetrievalResult


@pytest.fixture(scope="module")
def dense(full_corpus):
    retriever = DenseRetriever(full_corpus, HashingEmbedder())
    retriever.build()
    yield retriever
    retriever.close()


def signature(results):
    return [(r.chunk.chunk_id, round(r.score, 6)) for r in results]


def test_build_indexes_every_chunk_with_stable_ids(dense, full_corpus):
    assert dense.client.count(dense.collection_name, exact=True).count == len(full_corpus)
    assert dense.is_current()
    assert point_id("dev-abc") == point_id("dev-abc") != point_id("dev-abd")


def test_results_validate_and_preserve_original_chunks(dense, full_corpus):
    results = dense.search("export controls on advanced AI chips to China", top_k=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        RetrievalResult.model_validate(r.model_dump())
        assert r.chunk == full_corpus.get(r.chunk.chunk_id)


@pytest.mark.parametrize("filters,check", [
    (ChunkFilter(tickers="nvda"), lambda c: c.ticker == "NVDA"),
    (ChunkFilter(fiscal_years=2024), lambda c: c.fiscal_year == 2024),
    (ChunkFilter(tickers="JPM"), lambda c: c.ticker == "JPM"),
    (ChunkFilter(section_ids="item_7"), lambda c: c.section_id == "item_7"),
    (ChunkFilter(tickers="AMD", fiscal_years=2024, section_ids=["item_1", "item_1a"]),
     lambda c: c.ticker == "AMD" and c.fiscal_year == 2024 and c.section_id in {"item_1", "item_1a"}),
])
def test_filters_are_applied_inside_qdrant(dense, full_corpus, filters, check):
    results = dense.search("revenue growth and supply risks", top_k=50, filters=filters)
    expected = {c.chunk_id for c in full_corpus if check(c)}
    assert len(results) == min(50, len(expected))
    assert {r.chunk.chunk_id for r in results} <= expected


def test_qdrant_filter_translation():
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter(ChunkFilter()) is None
    f = to_qdrant_filter(ChunkFilter(tickers="AMD", fiscal_years=[2024, 2023]))
    assert {c.key: c.match.any for c in f.must} == {"ticker": ["AMD"], "fiscal_year": [2023, 2024]}


@pytest.mark.parametrize("query,filters", [
    ("", None),
    ("  ", None),
    ("export controls", ChunkFilter(tickers="UNKNOWN")),
    ("export controls", ChunkFilter(fiscal_years=2021)),
])
def test_empty_results(dense, query, filters):
    assert dense.search(query, top_k=5, filters=filters) == []


def test_empty_corpus_returns_no_results():
    retriever = DenseRetriever([], HashingEmbedder())
    retriever.build()
    assert retriever.search("export controls", top_k=5) == []


@pytest.mark.parametrize("bad", [0, -3, False])
def test_invalid_top_k_raises(dense, bad):
    with pytest.raises(ValueError):
        dense.search("export", top_k=bad)


def test_search_before_build_raises(full_corpus):
    with pytest.raises(DenseIndexError):
        DenseRetriever(full_corpus, HashingEmbedder()).search("export", top_k=3)


def test_rebuild_is_reproducible_and_input_order_independent(full_corpus):
    shuffled = list(full_corpus)
    random.Random(11).shuffle(shuffled)
    query = "cybersecurity attacks and data breaches"
    one = DenseRetriever(full_corpus, HashingEmbedder())
    one.build()
    two = DenseRetriever(shuffled, HashingEmbedder())
    two.build()
    assert one.manifest() == two.manifest()
    assert signature(one.search(query, top_k=10)) == signature(two.search(query, top_k=10))


def test_persistent_index_reopens_and_detects_staleness(tmp_path, full_corpus):
    path = tmp_path / "qdrant"
    query = "employees in our global workforce"
    first = DenseRetriever(full_corpus, HashingEmbedder(), index_path=path)
    assert first.ensure_index() is True
    assert first.ensure_index() is False
    expected = signature(first.search(query, top_k=5))
    first.close()

    reopened = DenseRetriever(full_corpus, HashingEmbedder(), index_path=path)
    assert reopened.is_current()
    assert signature(reopened.search(query, top_k=5)) == expected
    reopened.close()

    smaller = ChunkCorpus(list(full_corpus)[:-1])
    stale = DenseRetriever(smaller, HashingEmbedder(), index_path=path)
    assert not stale.is_current()
    assert stale.ensure_index() is True
    assert stale.client.count(stale.collection_name, exact=True).count == len(smaller)
    stale.close()

    other_model = DenseRetriever(smaller, HashingEmbedder(dimension=64), index_path=path)
    assert not other_model.is_current()
    other_model.close()


class _ConstantEmbedder(HashingEmbedder):
    """Every text gets the same vector, so every score ties."""

    @property
    def model_id(self):
        return "constant"

    def _vector(self, text):
        vec = np.zeros(self.dimension, dtype=np.float32)
        vec[0] = 1.0
        return vec


def test_full_ties_are_ordered_by_chunk_id(full_corpus):
    retriever = DenseRetriever(full_corpus, _ConstantEmbedder(dimension=8))
    retriever.build()
    results = retriever.search("anything", top_k=5)
    assert [r.chunk.chunk_id for r in results] == [c.chunk_id for c in full_corpus][:5]


# Real SentenceTransformer model ------------------------------------------------

@pytest.fixture(scope="module")
def st_dense(full_corpus):
    pytest.importorskip("sentence_transformers")
    from evidenceroute.retrieval.dense import DEFAULT_MODEL, SentenceTransformerEmbedder

    try:
        embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    except (AttributeError, TypeError, DenseIndexError):
        raise  # adapter bugs must fail the test, not skip it
    except Exception as exc:  # weights not cached and the Hub is unreachable
        pytest.skip(f"SentenceTransformer model {DEFAULT_MODEL} unavailable: {exc!r}")
    retriever = DenseRetriever(full_corpus, embedder)
    retriever.build()
    yield retriever
    retriever.close()


def test_sentence_transformer_search_preserves_canonical_provenance(st_dense, full_corpus):
    results = st_dense.search("financial results and business risks", top_k=5)
    assert len(results) == 5
    assert all(r.chunk == full_corpus.get(r.chunk.chunk_id) for r in results)
