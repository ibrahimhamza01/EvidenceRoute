"""Dense retrieval with Qdrant (embedded mode).

Most tests use the deterministic ``HashingEmbedder`` so they run offline and fast.
``test_sentence_transformer_*`` exercises a real SentenceTransformer model and is
skipped when the library or the model weights are unavailable.
"""
import random

import numpy as np
import pytest

from retrieval_testdata import AMD_2023, AMD_2024, NVDA_2024
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
def dense(dev_corpus):
    retriever = DenseRetriever(dev_corpus, HashingEmbedder())
    retriever.build()
    yield retriever
    retriever.close()


def signature(results):
    return [(r.chunk.chunk_id, round(r.score, 6)) for r in results]


def test_build_indexes_every_chunk_with_stable_ids(dense, dev_corpus):
    assert dense.client.count(dense.collection_name, exact=True).count == len(dev_corpus)
    assert dense.is_current()
    assert point_id("dev-abc") == point_id("dev-abc") != point_id("dev-abd")


def test_results_validate_and_preserve_original_chunks(dense, dev_corpus):
    results = dense.search("export controls on advanced AI chips to China", top_k=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    for r in results:
        RetrievalResult.model_validate(r.model_dump())
        assert r.chunk == dev_corpus.get(r.chunk.chunk_id)


def test_similar_passage_ranks_first(dense, chunk_by_anchor):
    query = "Silo AI lab in Finland trains large language models on AMD Instinct accelerators"
    top = dense.search(query, top_k=1)[0].chunk
    assert top == chunk_by_anchor(AMD_2024, "During the year, we completed the acquisition of Silo AI")


@pytest.mark.parametrize("filters,check", [
    (ChunkFilter(tickers="nvda"), lambda c: c.ticker == "NVDA"),
    (ChunkFilter(fiscal_years=2024), lambda c: c.fiscal_year == 2024),
    (ChunkFilter(document_ids=AMD_2023), lambda c: c.document_id == AMD_2023),
    (ChunkFilter(section_ids="item_7"), lambda c: c.section_id == "item_7"),
    (ChunkFilter(tickers="AMD", fiscal_years=2024, section_ids=["item_1", "item_1a"]),
     lambda c: c.ticker == "AMD" and c.fiscal_year == 2024 and c.section_id in {"item_1", "item_1a"}),
])
def test_filters_are_applied_inside_qdrant(dense, dev_corpus, filters, check):
    results = dense.search("revenue growth and supply risks", top_k=50, filters=filters)
    expected = {c.chunk_id for c in dev_corpus if check(c)}
    assert {r.chunk.chunk_id for r in results} == expected


def test_qdrant_filter_translation():
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter(ChunkFilter()) is None
    f = to_qdrant_filter(ChunkFilter(tickers="AMD", fiscal_years=[2024, 2023]))
    assert {c.key: c.match.any for c in f.must} == {"ticker": ["AMD"], "fiscal_year": [2023, 2024]}


@pytest.mark.parametrize("query,filters", [
    ("", None),
    ("  ", None),
    ("export controls", ChunkFilter(tickers="MSFT")),
    ("export controls", ChunkFilter(fiscal_years=2022)),
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


def test_search_before_build_raises(dev_corpus):
    with pytest.raises(DenseIndexError):
        DenseRetriever(dev_corpus, HashingEmbedder()).search("export", top_k=3)


def test_identical_texts_are_both_kept_and_tie_broken_by_chunk_id(dense, chunk_by_anchor):
    a = chunk_by_anchor(AMD_2023, "We utilize third-party wafer foundries")
    b = chunk_by_anchor(AMD_2024, "We utilize third-party wafer foundries")
    results = dense.search(a.text, top_k=2)
    assert [r.chunk.chunk_id for r in results] == sorted([a.chunk_id, b.chunk_id])
    assert results[0].score == pytest.approx(results[1].score)


def test_rebuild_is_reproducible_and_input_order_independent(dev_corpus):
    shuffled = list(dev_corpus)
    random.Random(11).shuffle(shuffled)
    query = "cybersecurity attacks and data breaches"
    one = DenseRetriever(dev_corpus, HashingEmbedder())
    one.build()
    two = DenseRetriever(shuffled, HashingEmbedder())
    two.build()
    assert one.manifest() == two.manifest()
    assert signature(one.search(query, top_k=10)) == signature(two.search(query, top_k=10))


def test_persistent_index_reopens_and_detects_staleness(tmp_path, dev_corpus):
    path = tmp_path / "qdrant"
    query = "employees in our global workforce"
    first = DenseRetriever(dev_corpus, HashingEmbedder(), index_path=path)
    assert first.ensure_index() is True
    assert first.ensure_index() is False
    expected = signature(first.search(query, top_k=5))
    first.close()

    reopened = DenseRetriever(dev_corpus, HashingEmbedder(), index_path=path)
    assert reopened.is_current()
    assert signature(reopened.search(query, top_k=5)) == expected
    reopened.close()

    smaller = ChunkCorpus(list(dev_corpus)[:-1])
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


def test_full_ties_are_ordered_by_chunk_id(dev_corpus):
    retriever = DenseRetriever(dev_corpus, _ConstantEmbedder(dimension=8))
    retriever.build()
    results = retriever.search("anything", top_k=5)
    assert [r.chunk.chunk_id for r in results] == [c.chunk_id for c in dev_corpus][:5]


# Real SentenceTransformer model ------------------------------------------------

@pytest.fixture(scope="module")
def st_dense(dev_corpus):
    pytest.importorskip("sentence_transformers")
    from evidenceroute.retrieval.dense import DEFAULT_MODEL, SentenceTransformerEmbedder

    try:
        embedder = SentenceTransformerEmbedder(DEFAULT_MODEL)
    except (AttributeError, TypeError, DenseIndexError):
        raise  # adapter bugs must fail the test, not skip it
    except Exception as exc:  # weights not cached and the Hub is unreachable
        pytest.skip(f"SentenceTransformer model {DEFAULT_MODEL} unavailable: {exc!r}")
    retriever = DenseRetriever(dev_corpus, embedder)
    retriever.build()
    yield retriever
    retriever.close()


def test_sentence_transformer_matches_paraphrased_supply_question(st_dense, chunk_by_anchor):
    results = st_dense.search("reliance on outside contract manufacturers to fabricate chips",
                              top_k=2, filters=ChunkFilter(tickers="AMD", fiscal_years=2024))
    expected = chunk_by_anchor(AMD_2024, "We utilize third-party wafer foundries")
    assert expected in [r.chunk for r in results]


def test_sentence_transformer_matches_paraphrased_headcount_question(st_dense, chunk_by_anchor):
    results = st_dense.search("How many people does the company employ?",
                              top_k=1, filters=ChunkFilter(document_ids=NVDA_2024))
    assert results[0].chunk == chunk_by_anchor(NVDA_2024, "We believe that our employees")
