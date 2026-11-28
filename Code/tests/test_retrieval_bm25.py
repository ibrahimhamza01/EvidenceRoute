import random

import pytest

from evidenceroute.retrieval import BM25Config, BM25Retriever, ChunkFilter
from evidenceroute.retrieval.tokenization import tokenize
from evidenceroute.schemas import Chunk, RetrievalResult


@pytest.fixture(scope="module")
def bm25(full_corpus):
    return BM25Retriever(full_corpus)


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_tokenizer_keeps_identifiers():
    assert tokenize("NVIDIA’s H100 and MI300X; ECCN 3A090, Item 1A.") == [
        "nvidia", "h100", "mi300x", "eccn", "3a090", "item", "1a"]


@pytest.mark.parametrize("filters,check", [
    (ChunkFilter(tickers="amd"), lambda c: c.ticker == "AMD"),
    (ChunkFilter(fiscal_years=2023), lambda c: c.fiscal_year == 2023),
    (ChunkFilter(tickers="MSFT"), lambda c: c.ticker == "MSFT"),
    (ChunkFilter(section_ids="ITEM_1A"), lambda c: c.section_id == "item_1a"),
    (ChunkFilter(tickers=["NVDA"], fiscal_years=[2024], section_ids=["item_1a", "item_7"]),
     lambda c: c.ticker == "NVDA" and c.fiscal_year == 2024 and c.section_id in {"item_1a", "item_7"}),
])
def test_filters_restrict_results(bm25, full_corpus, filters, check):
    query = "export controls revenue data center employees foundries security"
    results = bm25.search(query, top_k=30000, filters=filters)
    assert results and all(check(r.chunk) for r in results)
    unfiltered = [r for r in bm25.search(query, top_k=30000) if check(r.chunk)]
    # Filtering never changes a chunk's score or relative order (corpus-level IDF).
    assert [(r.chunk.chunk_id, r.score) for r in results] == \
        [(r.chunk.chunk_id, r.score) for r in unfiltered]


@pytest.mark.parametrize("query,filters", [
    ("", None),
    ("   \n", None),
    ("the of and with", None),                      # stopwords only
    ("zyzzyva quokka", None),                        # no matching terms
    ("export controls", ChunkFilter(tickers="UNKNOWN")),  # filter matches nothing
])
def test_empty_results(bm25, query, filters):
    assert bm25.search(query, top_k=5, filters=filters) == []


def test_empty_corpus_returns_no_results():
    assert BM25Retriever([]).search("export controls", top_k=5) == []


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_invalid_top_k_raises(bm25, bad):
    with pytest.raises(ValueError):
        bm25.search("export", top_k=bad)


def test_empty_filter_list_is_rejected():
    with pytest.raises(ValueError):
        ChunkFilter(tickers=[])


def test_top_k_limits_and_never_pads(bm25):
    assert len(bm25.search("export", top_k=2)) == 2
    all_hits = bm25.search("Silo", top_k=100)
    assert all_hits and all("silo" in r.chunk.text.casefold() for r in all_hits)


def test_results_validate_and_preserve_original_chunks(bm25, full_corpus):
    for result in bm25.search("revenue data center GPUs", top_k=10):
        RetrievalResult.model_validate(result.model_dump())
        original = full_corpus.get(result.chunk.chunk_id)
        assert result.chunk == original
        assert result.chunk.text == original.text
        assert result.chunk.source_url == original.source_url
        assert result.score > 0


def test_index_is_reproducible_regardless_of_input_order(full_corpus):
    shuffled = list(full_corpus)
    random.Random(7).shuffle(shuffled)
    a = BM25Retriever(full_corpus).search("export restrictions China GPUs", top_k=10)
    b = BM25Retriever(shuffled).search("export restrictions China GPUs", top_k=10)
    assert [(r.chunk.chunk_id, r.score) for r in a] == [(r.chunk.chunk_id, r.score) for r in b]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id="d", company_name="C", ticker="T", fiscal_year=2024,
                 section_id="item_1a", section_name="Risk Factors", chunk_index=0, text=text,
                 char_start=0, char_end=len(text), source_url="https://www.sec.gov/x")


def test_term_in_every_document_still_scores_positive():
    chunks = [_chunk("a", "supply risk"), _chunk("b", "supply chain"), _chunk("c", "supply demand")]
    results = BM25Retriever(chunks, BM25Config()).search("supply", top_k=3)
    assert len(results) == 3 and all(r.score > 0 for r in results)


def test_duplicate_text_tie_break_and_no_padding():
    chunks = [_chunk("b", "unique supply phrase"), _chunk("a", "unique supply phrase")]
    results = BM25Retriever(chunks).search("unique", top_k=10)
    assert ids(results) == ["a", "b"]
    assert results[0].score == results[1].score


def test_full_corpus_search_reaches_every_canonical_filing(bm25, full_corpus):
    for document_id in {c.document_id for c in full_corpus}:
        results = bm25.search("revenue financial risk", top_k=3, filters=ChunkFilter(document_ids=document_id))
        assert results
        assert all(r.chunk.document_id == document_id for r in results)


@pytest.mark.parametrize("filters", [
    ChunkFilter(tickers="nvda"),
    ChunkFilter(fiscal_years=[2022, 2024]),
    ChunkFilter(section_ids="ITEM_1A"),
    ChunkFilter(tickers=["AMD", "INTC"], fiscal_years=2023, section_ids=["item_1", "item_7"]),
    ChunkFilter(tickers="NVDA", section_ids="item_1c", fiscal_years=2022),
    ChunkFilter(tickers="NOPE"),
])
def test_precomputed_filters_match_per_chunk_filtering(bm25, full_corpus, filters):
    expected = frozenset(i for i, c in enumerate(full_corpus) if filters.matches(c))
    assert bm25.allowed_positions(filters) == expected
    query = "risk competition revenue supply export"
    unfiltered = bm25.scores(query)
    assert bm25.scores(query, filters) == {i: s for i, s in unfiltered.items() if i in expected}
