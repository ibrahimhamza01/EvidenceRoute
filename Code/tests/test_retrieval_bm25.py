import random

import pytest

from retrieval_testdata import AMD_2023, AMD_2024, NVDA_2023, NVDA_2024
from evidenceroute.retrieval import BM25Config, BM25Retriever, ChunkFilter
from evidenceroute.retrieval.tokenization import tokenize
from evidenceroute.schemas import Chunk, RetrievalResult


@pytest.fixture(scope="module")
def bm25(dev_corpus):
    return BM25Retriever(dev_corpus)


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_tokenizer_keeps_identifiers():
    assert tokenize("NVIDIA’s H100 and MI300X; ECCN 3A090, Item 1A.") == [
        "nvidia", "h100", "mi300x", "eccn", "3a090", "item", "1a"]


@pytest.mark.parametrize("query,document_id,prefix", [
    ("Silo AI acquisition Finland", AMD_2024, "During the year, we completed the acquisition of Silo AI"),
    ("ECCN 3A090 4A090", AMD_2023, "In October 2023, BIS issued new requirements"),
    ("Data Center revenue $15.01 billion", NVDA_2023, "Data Center revenue for fiscal year 2023 was $15.01 billion"),
    ("Transformer Engine Hopper", NVDA_2023, "In fiscal year 2023, we introduced the Hopper architecture"),
    ("29,600 employees in 36 countries", NVDA_2024, "We believe that our employees are our greatest assets"),
])
def test_exact_terms_rank_the_source_passage_first(bm25, chunk_by_anchor, query, document_id, prefix):
    results = bm25.search(query, top_k=3)
    assert results[0].chunk == chunk_by_anchor(document_id, prefix)


def test_product_name_query_returns_only_passages_containing_it(bm25):
    results = bm25.search("H100", top_k=10)
    assert len(results) == 4
    assert all("H100" in r.chunk.text for r in results)
    assert {r.chunk.ticker for r in results} == {"NVDA"}


def test_identical_text_ties_are_broken_by_chunk_id(bm25, chunk_by_anchor):
    a = chunk_by_anchor(AMD_2023, "We utilize third-party wafer foundries")
    b = chunk_by_anchor(AMD_2024, "We utilize third-party wafer foundries")
    assert a.text == b.text and a.chunk_id != b.chunk_id
    results = bm25.search("TSMC", top_k=2)
    assert results[0].score == results[1].score
    assert ids(results) == sorted([a.chunk_id, b.chunk_id])


@pytest.mark.parametrize("filters,check", [
    (ChunkFilter(tickers="amd"), lambda c: c.ticker == "AMD"),
    (ChunkFilter(fiscal_years=2023), lambda c: c.fiscal_year == 2023),
    (ChunkFilter(document_ids=NVDA_2024), lambda c: c.document_id == NVDA_2024),
    (ChunkFilter(section_ids="ITEM_1A"), lambda c: c.section_id == "item_1a"),
    (ChunkFilter(tickers=["NVDA"], fiscal_years=[2024], section_ids=["item_1a", "item_7"]),
     lambda c: c.ticker == "NVDA" and c.fiscal_year == 2024 and c.section_id in {"item_1a", "item_7"}),
])
def test_filters_restrict_results(bm25, dev_corpus, filters, check):
    query = "export controls revenue data center employees foundries security"
    results = bm25.search(query, top_k=50, filters=filters)
    assert results and all(check(r.chunk) for r in results)
    unfiltered = [r for r in bm25.search(query, top_k=50) if check(r.chunk)]
    # Filtering never changes a chunk's score or relative order (corpus-level IDF).
    assert [(r.chunk.chunk_id, r.score) for r in results] == \
        [(r.chunk.chunk_id, r.score) for r in unfiltered]


def test_filter_can_change_the_top_hit(bm25, chunk_by_anchor):
    top_all = bm25.search("export controls", top_k=1)[0].chunk
    top_amd = bm25.search("export controls", top_k=1, filters=ChunkFilter(tickers="AMD"))[0].chunk
    assert top_amd.ticker == "AMD" and top_amd != top_all


@pytest.mark.parametrize("query,filters", [
    ("", None),
    ("   \n", None),
    ("the of and with", None),                      # stopwords only
    ("zyzzyva quokka", None),                        # no matching terms
    ("export controls", ChunkFilter(tickers="MSFT")),  # filter matches nothing
    ("H100", ChunkFilter(tickers="AMD")),            # term absent from filtered subset
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
    assert len(all_hits) == 1


def test_results_validate_and_preserve_original_chunks(bm25, dev_corpus):
    for result in bm25.search("revenue data center GPUs", top_k=10):
        RetrievalResult.model_validate(result.model_dump())
        original = dev_corpus.get(result.chunk.chunk_id)
        assert result.chunk == original
        assert result.chunk.text == original.text
        assert result.chunk.source_url == original.source_url
        assert result.score > 0


def test_index_is_reproducible_regardless_of_input_order(dev_corpus):
    shuffled = list(dev_corpus)
    random.Random(7).shuffle(shuffled)
    a = BM25Retriever(dev_corpus).search("export restrictions China GPUs", top_k=10)
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
