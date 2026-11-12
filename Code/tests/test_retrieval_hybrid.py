import pytest

from retrieval_testdata import AMD_2024, NVDA_2023, NVDA_2024
from evidenceroute.retrieval import (
    BM25Retriever,
    ChunkFilter,
    HybridConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from evidenceroute.retrieval.dense import DenseRetriever, HashingEmbedder
from evidenceroute.retrieval.factory import build_hybrid_retriever
from evidenceroute.schemas import RetrievalResult


def r(chunk, score=1.0):
    return RetrievalResult(chunk=chunk, score=score)


@pytest.fixture(scope="module")
def chunks(dev_corpus):
    return list(dev_corpus)


@pytest.fixture(scope="module")
def hybrid(dev_corpus):
    return build_hybrid_retriever(dev_corpus, HashingEmbedder())


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=10, filters=None):
        self.calls.append((query, top_k, filters))
        return self.results[:top_k]


# Reciprocal Rank Fusion ----------------------------------------------------------

def test_rrf_scores_follow_the_formula(chunks):
    a, b, c = chunks[:3]
    hits = reciprocal_rank_fusion({"bm25": [r(a), r(b)], "dense": [r(b), r(c)]}, rrf_k=60)
    scores = {h.chunk.chunk_id: h.score for h in hits}
    assert scores[b.chunk_id] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[a.chunk_id] == pytest.approx(1 / 61)
    assert scores[c.chunk_id] == pytest.approx(1 / 62)
    assert hits[0].chunk == b
    assert dict(hits[0].ranks) == {"bm25": 2, "dense": 1}


def test_rrf_removes_duplicates_within_and_across_rankings(chunks):
    a, b = chunks[:2]
    hits = reciprocal_rank_fusion({"bm25": [r(a), r(a), r(b)], "dense": [r(b), r(a)]})
    assert [h.chunk.chunk_id for h in hits].count(a.chunk_id) == 1
    assert len(hits) == 2
    # The repeated entry does not consume a rank: b is still rank 2 in bm25.
    by_id = {h.chunk.chunk_id: h for h in hits}
    assert dict(by_id[b.chunk_id].ranks) == {"bm25": 2, "dense": 1}


def test_rrf_tie_break_uses_best_rank_then_chunk_id(chunks):
    a, b, c, d = sorted(chunks[:4], key=lambda x: x.chunk_id)
    # a has ranks (1, 4) and d has (4, 1): same fused score and best rank, so chunk_id decides.
    # b and c tie the same way one level below.
    hits = reciprocal_rank_fusion({"x": [r(a), r(b), r(c), r(d)], "y": [r(d), r(c), r(b), r(a)]}, 60)
    assert hits[0].score == pytest.approx(hits[1].score)
    assert [h.chunk.chunk_id for h in hits] == [a.chunk_id, d.chunk_id, b.chunk_id, c.chunk_id]
    # Exact ties on score and best rank fall back to chunk_id.
    hits = reciprocal_rank_fusion({"x": [r(b)], "y": [r(a)]})
    assert [h.chunk.chunk_id for h in hits] == [a.chunk_id, b.chunk_id]


def test_rrf_handles_empty_and_one_sided_rankings(chunks):
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []
    hits = reciprocal_rank_fusion({"bm25": [], "dense": [r(chunks[0])]})
    assert [h.chunk for h in hits] == [chunks[0]]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({}, rrf_k=0)


def test_hybrid_dedupes_and_requests_enough_candidates(chunks):
    a, b, c = chunks[:3]
    left = StubRetriever([r(a), r(b), r(a)])
    right = StubRetriever([r(c), r(a), r(b)])
    hybrid = HybridRetriever({"bm25": left, "dense": right}, HybridConfig(candidate_k=5))
    results = hybrid.search("anything", top_k=10)
    assert len({x.chunk.chunk_id for x in results}) == len(results) == 3
    assert results[0].chunk == a
    assert left.calls[0][1] == 10 and right.calls[0][1] == 10  # max(candidate_k, top_k)


def test_hybrid_passes_filters_to_every_retriever(chunks):
    left, right = StubRetriever([]), StubRetriever([])
    flt = ChunkFilter(tickers="AMD")
    HybridRetriever({"bm25": left, "dense": right}).search("q", top_k=3, filters=flt)
    assert left.calls[0][2] is flt and right.calls[0][2] is flt


# End-to-end over the verified passages -------------------------------------------

def test_hybrid_results_validate_and_preserve_chunks(hybrid, dev_corpus):
    results = hybrid.search("export controls on GPUs shipped to China", top_k=8)
    assert len(results) == 8
    assert len({x.chunk.chunk_id for x in results}) == 8
    scores = [x.score for x in results]
    assert scores == sorted(scores, reverse=True)
    for x in results:
        RetrievalResult.model_validate(x.model_dump())
        original = dev_corpus.get(x.chunk.chunk_id)
        assert (x.chunk.chunk_id, x.chunk.text, x.chunk.source_url) == \
            (original.chunk_id, original.text, original.source_url)


def test_hybrid_combines_both_signals(hybrid, chunk_by_anchor):
    hits = hybrid.search_with_details("Silo AI acquisition", top_k=3)
    top = hits[0]
    assert top.chunk == chunk_by_anchor(AMD_2024, "During the year, we completed the acquisition of Silo AI")
    assert set(top.ranks) == {"bm25", "dense"}


def test_same_text_in_different_filings_is_not_deduplicated(hybrid, chunk_by_anchor):
    fy23 = chunk_by_anchor(NVDA_2023, "We derive our revenue from product sales")
    fy24 = chunk_by_anchor(NVDA_2024, "We derive our revenue from product sales")
    assert fy23.text == fy24.text
    results = hybrid.search("identification of the performance obligations in the contract", top_k=2)
    assert {x.chunk.chunk_id for x in results} == {fy23.chunk_id, fy24.chunk_id}


@pytest.mark.parametrize("filters,check", [
    (ChunkFilter(tickers="NVDA"), lambda c: c.ticker == "NVDA"),
    (ChunkFilter(fiscal_years=2023), lambda c: c.fiscal_year == 2023),
    (ChunkFilter(document_ids=AMD_2024), lambda c: c.document_id == AMD_2024),
    (ChunkFilter(section_ids="item_1a"), lambda c: c.section_id == "item_1a"),
])
def test_hybrid_filters(hybrid, filters, check):
    results = hybrid.search("supply chain export risk", top_k=20, filters=filters)
    assert results and all(check(x.chunk) for x in results)


@pytest.mark.parametrize("query,filters", [
    ("", None),
    ("\t", None),
    ("export controls", ChunkFilter(tickers="GOOG")),
])
def test_hybrid_empty_results(hybrid, query, filters):
    assert hybrid.search(query, top_k=5, filters=filters) == []


@pytest.mark.parametrize("bad", [0, -1, None])
def test_hybrid_invalid_top_k(hybrid, bad):
    with pytest.raises(ValueError):
        hybrid.search("export", top_k=bad)


def test_hybrid_is_reproducible_across_rebuilds(dev_corpus):
    query = "data center revenue growth driven by GPUs"
    one = build_hybrid_retriever(dev_corpus, HashingEmbedder(), rebuild=True).search(query, top_k=10)
    two = build_hybrid_retriever(list(reversed(list(dev_corpus))), HashingEmbedder(),
                                 rebuild=True).search(query, top_k=10)
    assert [(x.chunk.chunk_id, x.score) for x in one] == [(x.chunk.chunk_id, x.score) for x in two]


def test_hybrid_works_with_explicit_components(dev_corpus):
    dense = DenseRetriever(dev_corpus, HashingEmbedder())
    dense.build()
    hybrid = HybridRetriever({"bm25": BM25Retriever(dev_corpus), "dense": dense})
    assert hybrid.search("employees", top_k=3)
