"""Cross-encoder reranking.

Most tests use a deterministic stub scorer so they run offline without model
downloads. ``test_real_cross_encoder_*`` loads a pretrained cross-encoder and is
skipped when sentence-transformers or the weights are unavailable.
"""
import sys

import numpy as np
import pytest

from evidenceroute.retrieval import ChunkFilter
from evidenceroute.retrieval.dense import HashingEmbedder
from evidenceroute.retrieval.factory import build_hybrid_retriever
from evidenceroute.retrieval.rerank import (
    RerankConfig,
    RerankedHit,
    RerankerError,
    RerankerUnavailableError,
    RerankingRetriever,
    SentenceTransformerCrossEncoder,
    build_reranking_retriever,
)
from evidenceroute.retrieval.tokenization import tokenize
from evidenceroute.schemas import RetrievalResult


class OverlapScorer:
    """Scores a passage by how many distinct query terms it contains."""

    model_id = "stub:overlap"

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        terms = set(tokenize(query))
        return np.array([len(terms & set(tokenize(p))) for p in passages], dtype=np.float32)


class FixedScorer:
    model_id = "stub:fixed"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)

    def score(self, query, passages):
        return self.scores


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=10, filters=None):
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


@pytest.fixture(scope="module")
def hybrid(full_corpus):
    return build_hybrid_retriever(full_corpus, HashingEmbedder())


@pytest.fixture(scope="module")
def candidates(full_corpus):
    return [RetrievalResult(chunk=c, score=1.0 / (i + 1)) for i, c in enumerate(list(full_corpus)[:20])]


def ids(results):
    return [r.chunk.chunk_id for r in results]


# Ordering and configuration ------------------------------------------------------

def test_reranks_candidates_by_cross_encoder_score(candidates):
    scores = np.arange(len(candidates), dtype=np.float32)  # last candidate scores highest
    retriever = RerankingRetriever(StubRetriever(candidates), FixedScorer(scores),
                                   RerankConfig(candidate_k=20))
    results = retriever.search("anything", top_k=5)
    assert ids(results) == ids(candidates[::-1][:5])
    assert [r.score for r in results] == [19.0, 18.0, 17.0, 16.0, 15.0]


def test_candidate_depth_and_final_top_k_are_configurable(candidates):
    base = StubRetriever(candidates)
    scorer = OverlapScorer()
    retriever = RerankingRetriever(base, scorer, RerankConfig(candidate_k=12))
    assert len(retriever.search("risk", top_k=3)) == 3
    assert base.calls[-1][1] == 12
    assert len(scorer.calls[-1][1]) == 12
    retriever.search("risk", top_k=15)  # top_k above candidate_k widens the pool
    assert base.calls[-1][1] == 15


def test_ties_keep_first_stage_order_then_chunk_id(candidates):
    retriever = RerankingRetriever(StubRetriever(candidates),
                                   FixedScorer(np.zeros(len(candidates))),
                                   RerankConfig(candidate_k=20))
    assert ids(retriever.search("q", top_k=20)) == ids(candidates)


def test_details_record_first_stage_rank(candidates):
    scores = np.zeros(len(candidates))
    scores[7] = 5.0
    retriever = RerankingRetriever(StubRetriever(candidates), FixedScorer(scores),
                                   RerankConfig(candidate_k=20))
    top = retriever.search_with_details("q", top_k=1)[0]
    assert isinstance(top, RerankedHit)
    assert (top.first_stage_rank, top.score) == (8, 5.0)
    assert top.first_stage_score == candidates[7].score


def test_disabled_returns_first_stage_unchanged_without_scoring(hybrid):
    filters = ChunkFilter(tickers="NVDA", fiscal_years=2024)
    retriever = RerankingRetriever(hybrid, None, RerankConfig(enabled=False))
    assert retriever.search("export controls", 10, filters) == hybrid.search("export controls", 10, filters)


def test_enabled_without_scorer_is_rejected(candidates):
    with pytest.raises(ValueError, match="no cross-encoder scorer"):
        RerankingRetriever(StubRetriever(candidates), None, RerankConfig())


def test_config_validation():
    with pytest.raises(ValueError):
        RerankConfig(candidate_k=0)
    with pytest.raises(ValueError):
        RerankConfig(unknown=True)


@pytest.mark.parametrize("top_k", [0, -1, True, 2.5])
def test_rejects_invalid_top_k(candidates, top_k):
    retriever = RerankingRetriever(StubRetriever(candidates), OverlapScorer())
    with pytest.raises(ValueError):
        retriever.search("q", top_k=top_k)


def test_blank_query_and_no_candidates_return_empty(candidates):
    scorer = OverlapScorer()
    assert RerankingRetriever(StubRetriever(candidates), scorer).search("   ") == []
    assert RerankingRetriever(StubRetriever([]), scorer).search("risk") == []
    assert scorer.calls == []


@pytest.mark.parametrize("scores,match", [
    (np.zeros(3), "scores for 20 candidates"),
    (np.full(20, np.nan), "non-finite"),
])
def test_unusable_scores_raise(candidates, scores, match):
    retriever = RerankingRetriever(StubRetriever(candidates), FixedScorer(scores),
                                   RerankConfig(candidate_k=20))
    with pytest.raises(RerankerError, match=match):
        retriever.search("q")


# Failure reporting ----------------------------------------------------------------

def test_missing_dependency_reports_the_cause(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails
    with pytest.raises(RerankerUnavailableError) as info:
        SentenceTransformerCrossEncoder("some/model")
    message = str(info.value)
    assert "some/model" in message and "sentence_transformers" in message
    assert "RerankConfig(enabled=False)" in message
    assert isinstance(info.value.__cause__, ImportError)


def test_disabled_config_never_loads_a_model(monkeypatch, candidates):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    retriever = build_reranking_retriever(StubRetriever(candidates), RerankConfig(enabled=False))
    assert retriever.scorer is None
    assert ids(retriever.search("q", top_k=4)) == ids(candidates[:4])


# Full corpus ----------------------------------------------------------------------

def test_full_corpus_rerank_preserves_chunks_and_filters(hybrid, full_corpus):
    filters = ChunkFilter(tickers=["AMD", "NVDA"], fiscal_years=[2023, 2024], section_ids="item_1a")
    retriever = RerankingRetriever(hybrid, OverlapScorer(), RerankConfig(candidate_k=50))
    results = retriever.search("export restrictions on data center GPUs", top_k=10, filters=filters)
    assert len(results) == 10
    candidate_ids = set(ids(hybrid.search("export restrictions on data center GPUs", 50, filters)))
    for r in results:
        RetrievalResult.model_validate(r.model_dump())
        assert r.chunk == full_corpus.get(r.chunk.chunk_id)  # original text, IDs and source URL
        assert r.chunk.chunk_id in candidate_ids
        assert filters.matches(r.chunk)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_full_corpus_rerank_is_reproducible(hybrid):
    retriever = RerankingRetriever(hybrid, OverlapScorer())
    first = retriever.search("supply chain disruption semiconductor", top_k=10)
    assert first == retriever.search("supply chain disruption semiconductor", top_k=10)


# Real pretrained cross-encoder (skips without model weights) -----------------------

@pytest.fixture(scope="module")
def real_scorer():
    pytest.importorskip("sentence_transformers")
    try:
        return SentenceTransformerCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except RerankerUnavailableError as exc:
        pytest.skip(f"Cross-encoder weights unavailable: {exc}")


def test_real_cross_encoder_prefers_the_relevant_passage(real_scorer):
    scores = real_scorer.score("Which company relies on third-party foundries to make its chips?", [
        "We rely on third-party foundries such as TSMC to manufacture our semiconductor wafers.",
        "Our board of directors approved a quarterly cash dividend.",
    ])
    assert scores.shape == (2,) and scores[0] > scores[1]


def test_real_cross_encoder_reranks_full_corpus_hybrid(real_scorer, hybrid, full_corpus):
    retriever = RerankingRetriever(hybrid, real_scorer, RerankConfig(candidate_k=20))
    results = retriever.search("reliance on outside chip manufacturers", top_k=5)
    assert len(results) == 5
    assert all(r.chunk == full_corpus.get(r.chunk.chunk_id) for r in results)
    assert results == retriever.search("reliance on outside chip manufacturers", top_k=5)
