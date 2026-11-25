# Hybrid retrieval and reranking (Issues #4 and #17)

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

BM25 lexical search, dense semantic search (SentenceTransformer embeddings stored in Qdrant), Reciprocal Rank Fusion, and optional cross-encoder reranking over validated `Chunk` records. The retrieval core consumes `Chunk` objects only and does not import the HTML parser. The production loader, demos, benchmark and evaluation read all 24,766 chunks from `data/processed/full-v1/chunks.parquet`, covering the canonical 75 filings.

```
Question -> BM25 + dense retrieval -> RRF -> cross-encoder reranker (optional) -> final top-k evidence
```

| File | Purpose |
| --- | --- |
| `Code/src/evidenceroute/retrieval/corpus.py` | `ChunkCorpus`: validation, duplicate handling, `chunk_id` ordering, fingerprint; JSONL and Parquet loaders |
| `Code/src/evidenceroute/retrieval/filters.py` | `ChunkFilter` by ticker, fiscal year, document ID and section ID; `filter_values` shared normalization |
| `Code/src/evidenceroute/retrieval/bm25.py` | `BM25Retriever` (Okapi BM25, non-negative IDF, precomputed filter index) |
| `Code/src/evidenceroute/retrieval/dense.py` | `DenseRetriever`, `SentenceTransformerEmbedder`, Qdrant index build and staleness checks |
| `Code/src/evidenceroute/retrieval/hybrid.py` | `reciprocal_rank_fusion`, `HybridRetriever` |
| `Code/src/evidenceroute/retrieval/rerank.py` | `RerankingRetriever`, `RerankConfig`, `SentenceTransformerCrossEncoder`, `build_reranking_retriever` |
| `Code/src/evidenceroute/retrieval/factory.py` | `build_hybrid_retriever` wiring BM25, dense and RRF from one corpus |
| `Code/scripts/demo_retrieval.py` | Command-line search demo |
| `Code/scripts/benchmark_retrieval.py` | Indexing time, index size, memory and query latency for every mode |
| `Code/scripts/retrieval_queries.json` | 24 fixed engineering queries (no gold evidence) |
| `Code/scripts/evaluate_retrieval.py` | Recall@K, MRR@10 and full coverage against supplied gold evidence |

## Usage

```python
from evidenceroute.retrieval import ChunkFilter, RerankConfig, load_chunks
from evidenceroute.retrieval.dense import SentenceTransformerEmbedder
from evidenceroute.retrieval.factory import build_hybrid_retriever
from evidenceroute.retrieval.rerank import build_reranking_retriever

chunks = load_chunks("data/processed/full-v1/chunks.parquet")
hybrid = build_hybrid_retriever(chunks, SentenceTransformerEmbedder(), index_path="data/indexes/full-v1")
retriever = build_reranking_retriever(hybrid, RerankConfig(candidate_k=50))
results = retriever.search("export restrictions on AI GPUs", top_k=10,
                           filters=ChunkFilter(tickers="NVDA", fiscal_years=2024, section_ids="item_1a"))
# -> list[RetrievalResult]; each result.chunk preserves the original text, IDs and source URL
```

All retrievers expose `search(query, top_k, filters) -> list[RetrievalResult]`, so a reranked retriever can replace a hybrid one without other changes. `HybridRetriever.search_with_details` returns each chunk's BM25 and dense ranks; `RerankingRetriever.search_with_details` returns each chunk's first-stage rank and score.

## Cross-encoder reranking

A cross-encoder reads the question and a passage together and scores their relevance. It is more precise than comparing separately computed embeddings, but it must run for every candidate of every query, so it only reorders the hybrid candidates.

| `RerankConfig` field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | `False` returns the first-stage ranking unchanged and loads no model |
| `candidate_k` | `50` | Candidates requested from the first stage (at least `top_k`) |
| `model_name` | `BAAI/bge-reranker-base` | Pretrained cross-encoder used by `build_reranking_retriever` |

- **Final top-k** is the `top_k` argument of `search` (10 in the benchmark and evaluation).
- **Ordering** is deterministic: cross-encoder score descending, then first-stage rank, then `chunk_id`.
- **Metadata** is preserved: results carry the original `Chunk` objects, and filters are applied by the first stage.
- **Failures are explicit.** When reranking is enabled and the model cannot be loaded or scored, `RerankerUnavailableError` is raised with the underlying cause, for example `Cross-encoder reranker 'BAAI/bge-reranker-base' could not be loaded: OSError: ...`, followed by how to install the dependency or disable reranking. The pipeline never silently falls back to an unreranked ranking; the fallback is choosing `RerankConfig(enabled=False)`. Unusable scores (wrong length, non-finite) raise `RerankerError`.
- **Model inputs** are truncated at 512 tokens. Chunks are at most 1,800 characters.

**Why `bge-reranker-base` is the default.** In the gold-evidence evaluation below it was the only reranker that improved on hybrid retrieval: MRR@10 rose from 0.372 to 0.528 without filters and from 0.551 to 0.641 with ticker/year filters. `cross-encoder/ms-marco-MiniLM-L-6-v2` left both roughly unchanged (0.404 and 0.538). The cost is latency: about 1.8 s per query for 50 candidates versus 0.33 s for MiniLM on the benchmark machine. Single-search strategies such as Standard RAG can absorb this. Strategies that search many times (Agentic RAG sub-queries, RLM sweeps) can lower `candidate_k` or disable reranking, and should measure the trade-off for their workload. An earlier qualitative look at four engineering queries suggested MiniLM was better; the gold-evidence scores contradicted it, which is why the choice rests on the scored evaluation.

## Full-corpus workflow

Run from the repository root with dependencies from `Code/requirements.txt` (`pyarrow==25.0.0` is pinned so regenerated Parquet files match the validated hashes).

```bash
# 1. Canonical chunks: download from S3 or regenerate (see ingestion.md)
PYTHONPATH=Code/src python Code/scripts/download_filings.py
PYTHONPATH=Code/src python Code/scripts/process_filings.py

# 2. Build (or reuse) the dense index and measure indexing and latency
PYTHONPATH=Code/src python Code/scripts/benchmark_retrieval.py --rebuild   # measures indexing stages
PYTHONPATH=Code/src python Code/scripts/benchmark_retrieval.py             # reuses the index

# 3. Score retrieval against gold evidence (paths are explicit)
PYTHONPATH=Code/src python Code/scripts/evaluate_retrieval.py \
  --benchmark data/benchmark/questions.csv --gold data/benchmark/gold_evidence.csv \
  --rerankers BAAI/bge-reranker-base cross-encoder/ms-marco-MiniLM-L-6-v2

# 4. Search
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "reliance on outside chip manufacturers" --show-ranks
```

`benchmark_retrieval.py` options: `--repeats`, `--warmup-rounds`, `--top-k`, `--rerankers` (pass with no values to skip), `--rerank-candidates`, `--device`, `--embedder hashing` for an offline smoke run. `evaluate_retrieval.py` options: `--split all|dev|heldout`, `--ks`, `--rerank-candidates`. Both write JSON reports under `data/benchmarks/`, which is ignored by Git.

## Index persistence, reuse and rebuilding

- The dense index is stored under `data/indexes/full-v1/<embedding model id>/` (ignored by Git). The directory name includes the model and query-prefix fingerprint, so different embedders never share an index.
- The collection metadata stores a manifest: schema version, embedding model, dimension, distance, corpus fingerprint (SHA-256 over canonical chunk records) and chunk count. `ensure_index()` reuses the index only when the stored manifest and point count match the current corpus and embedder; otherwise it rebuilds. A reused index loaded in 0.25 s on the benchmark machine.
- The corpus fingerprint depends on chunk content, not Parquet bytes, so an index stays valid across Parquet writer versions that produce identical records.
- Force a rebuild with `--rebuild` (benchmark and demo) or `DenseRetriever.build()`. Rebuilding is reproducible: chunks are inserted in `chunk_id` order with UUIDv5 point IDs derived from `chunk_id`.
- Rebuild after any change to `chunks.parquet`, the embedding model, its revision or its query/document prefixes. Reranker changes never require a rebuild because cross-encoders score at query time.

## Performance (engineering measurements)

Measured on an Apple Silicon Mac (macOS 26.6, arm64, PyTorch MPS), Python 3.13.5, sentence-transformers 6.3.0, torch 2.15.0, qdrant-client 1.20.0 (embedded mode), pyarrow 25.0.0. Embedding model `BAAI/bge-small-en-v1.5` (384 dimensions). Latency uses the 24 queries in `retrieval_queries.json`: one warm-up round, then 5 repeats per query (120 timings per mode), `top_k = 10`, hybrid `candidate_k = 50`, `rrf_k = 60`. These numbers depend on hardware and are not relevance results.

**Indexing (full rebuild of 24,766 chunks)**

| Stage | Time |
| --- | --- |
| Corpus load (Parquet to validated chunks) | 0.40 s |
| BM25 index build (in memory) | 1.39 s |
| Embedding model load | 4.40 s |
| Document embedding | 297.5 s |
| Qdrant writes and manifest | 9.3 s |
| **Dense index total** | **306.7 s** |

Index size on disk: 146.6 MB (`storage.sqlite`; vectors plus full chunk payloads). Peak process memory after indexing: about 1.5 GB. Embedding dominates (97%); Qdrant writes are small on this machine.

**Query latency (ms)**

| Mode | p50 | p95 | Unfiltered p50 | Filtered p50 |
| --- | --- | --- | --- | --- |
| BM25 | 0.66 | 10.8 | 2.6 | 0.4 |
| Dense | 114.6 | 143.0 | 23.0 | 120.0 |
| Hybrid (RRF) | 117.7 | 148.5 | 28.3 | 121.8 |
| Hybrid + bge-reranker-base @50 | 1,817.8 | 2,045.6 | | |
| Hybrid + bge-reranker-base @100 | 3,621.8 | 3,983.7 | | |
| Hybrid + MiniLM-L-6-v2 @50 | 328.6 | 418.8 | | |
| Hybrid + MiniLM-L-6-v2 @100 | 615.1 | 741.4 | | |

Reranked rows are end-to-end retrieval latency (query embedding, BM25, dense, RRF and reranking). Query embedding is about 8 to 13 ms of the dense time. Filtered and unfiltered columns use the medians of per-query p50 over the 14 filtered and 10 unfiltered queries.

**Reranking cost and effect (engineering queries)**

| Mode | Rerank-only p50 (ms) | Kept from hybrid top-10 | Promoted from beyond rank 10 | Deepest promoted rank |
| --- | --- | --- | --- | --- |
| bge-reranker-base @50 | 1,744 | 5.33 | 4.67 | 49 |
| bge-reranker-base @100 | 3,553 | 4.75 | 5.25 | 100 |
| MiniLM-L-6-v2 @50 | 265 | 5.96 | 4.04 | 50 |
| MiniLM-L-6-v2 @100 | 545 | 5.62 | 4.50 | 98 |

Reranking cost grows linearly with candidate depth. Doubling candidates to 100 doubled latency and promoted about 0.5 more passages per query, so 50 is the default. Loading `bge-reranker-base` took 33.8 s the first time (including a 1.1 GB download); MiniLM took 0.6 s.

All modes returned identical rankings across repeats, and every result satisfied its query filter.

## Retrieval quality against gold evidence

Scored with `evaluate_retrieval.py` against the reviewed full-corpus benchmark from Issue #18 (65 questions, 142 gold passages mapped to full-v1 chunk IDs; at the time of measurement it was in an open pull request and had not been merged). The 59 answerable questions with supporting gold evidence were scored (22 simple, 22 multi-source, 15 corpus-wide); the 6 unanswerable questions were skipped. Recall@K is the share of a question's evidence groups with a supporting chunk in the top K; MRR@10 uses the first supporting chunk; full coverage@10 requires every group in the top 10.

Two conditions: **none** searches with the question text only; **metadata** adds a filter on the question's annotated tickers and fiscal years, as a planner that already knows the target filings would.

**No filter**

| Mode | R@1 | R@5 | R@10 | R@20 | MRR@10 | Full cov.@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.180 | 0.449 | 0.552 | 0.643 | 0.354 | 0.441 |
| Dense | 0.146 | 0.343 | 0.454 | 0.546 | 0.298 | 0.356 |
| Hybrid | 0.188 | 0.480 | 0.527 | 0.623 | 0.372 | 0.441 |
| Hybrid + MiniLM @50 | 0.245 | 0.436 | 0.517 | 0.646 | 0.404 | 0.390 |
| **Hybrid + bge @50** | **0.324** | **0.572** | **0.630** | **0.682** | **0.528** | **0.491** |

**Ticker/year filter**

| Mode | R@1 | R@5 | R@10 | R@20 | MRR@10 | Full cov.@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.266 | 0.560 | 0.683 | 0.749 | 0.495 | 0.525 |
| Dense | 0.321 | 0.518 | 0.566 | 0.688 | 0.484 | 0.475 |
| Hybrid | 0.316 | 0.552 | 0.657 | 0.784 | 0.551 | 0.525 |
| Hybrid + MiniLM @50 | 0.324 | 0.519 | 0.640 | 0.735 | 0.538 | 0.525 |
| **Hybrid + bge @50** | **0.427** | **0.625** | **0.699** | **0.801** | **0.641** | **0.576** |

**By scope (R@10 / MRR@10)**

| Mode | Simple | Multi-source | Corpus-wide |
| --- | --- | --- | --- |
| Hybrid, no filter | 0.727 / 0.504 | 0.409 / 0.269 | 0.404 / 0.328 |
| bge @50, no filter | 0.864 / 0.659 | 0.523 / 0.499 | 0.447 / 0.380 |
| Hybrid, filter | 0.818 / 0.642 | 0.568 / 0.396 | 0.552 / 0.644 |
| bge @50, filter | 0.909 / 0.811 | 0.545 / 0.507 | 0.616 / 0.589 |

**Per-question comparison with hybrid** (better / same / worse; two-sided sign test on non-ties)

| Mode | Condition | R@10 | MRR@10 |
| --- | --- | --- | --- |
| bge @50 | none | 10 / 46 / 3 (p = 0.09) | 23 / 25 / 11 (p = 0.06) |
| bge @50 | filter | 9 / 43 / 7 (p = 0.80) | 18 / 29 / 12 (p = 0.36) |
| MiniLM @50 | none | 5 / 48 / 6 | 15 / 30 / 14 (p = 1.0) |
| MiniLM @50 | filter | 9 / 41 / 9 | 15 / 26 / 18 (p = 0.73) |

**Observations**

- `bge-reranker-base` improves ranking quality, mostly by moving gold evidence toward the top (largest gain at R@1 and MRR@10, and for simple questions). The gain is clearer without filters than with them.
- `ms-marco-MiniLM-L-6-v2` does not improve on hybrid retrieval for this corpus.
- BM25 alone has higher R@10 than hybrid in both conditions (0.552 vs 0.527, 0.683 vs 0.657), although hybrid has higher MRR@10 with filters. The weaker dense ranking appears to displace some BM25 hits during fusion. This is reported for the team; RRF was not changed in this work.
- With filters, bge lowers corpus-wide MRR@10 (0.644 to 0.589) while raising R@10.
- Corpus-wide questions can have more evidence groups than 10, so their R@10 is bounded below 1 by design. These questions are the motivation for decomposed or recursive strategies rather than a single top-k search.

**Limitations.** 59 questions is a small sample; only the unfiltered MRR@10 improvement for bge approaches conventional significance. Scores cover both dev and heldout splits; use `--split heldout` for held-out numbers. The benchmark has one embedding model and one candidate depth for reranking quality. These are retrieval metrics only, not answer quality.

## Qdrant deployment notes

- **Embedded mode** (`QdrantClient(path=...)` or `:memory:`) needs no server and is what tests, demos and the measurements above use. It is suitable for development on this corpus.
- **Filtered dense search is slow in embedded mode**: the median filtered query took 120 ms versus 23 ms unfiltered, because embedded mode ignores payload indexes and checks every point's metadata. Agentic and recursive strategies issue many filtered searches, so this matters for them.
- **Scores jitter in embedded mode**: identical calls returned scores (and vectors) differing by up to about 5e-8. Dense scores are rounded to 6 decimals before the `chunk_id` tie-break, which made repeated rankings identical in all tests (repeated top-50 rankings are covered by `test_deep_rankings_are_identical_across_repeated_searches`). Server behavior was not measured.
- Qdrant warns that embedded collections above 20,000 points are not recommended; this corpus has 24,766.
- **For deployment**, run Qdrant as a server (Docker) or use Qdrant Cloud and pass `QdrantClient(url=..., api_key=...)` to `DenseRetriever` or `build_hybrid_retriever`. `build()` creates payload indexes for `ticker`, `fiscal_year`, `document_id` and `section_id`, which servers use for filtering. Plan for about 150 MB of storage for this corpus with full payloads, about 1.5 GB of memory while embedding, and a one-time rebuild of about 5 minutes on Apple Silicon (dominated by embedding; a GPU server will differ). Server-mode latency has not been measured yet.

## Design notes

- **Filters**: `None` means no constraint; values within a field are ORed, fields are ANDed. Tickers are case-insensitive, section IDs are lower-cased (`item_1a`). `filter_values` defines the normalization once for `ChunkFilter.matches` and the BM25 filter index. BM25 intersects precomputed per-value position sets before scoring; dense search applies filters inside Qdrant.
- **BM25**: Lucene-style IDF `ln(1 + (N - df + 0.5)/(df + 0.5))`, which stays positive in small corpora. IDF is computed over the full corpus, so filtering never changes a chunk's score. Chunks with zero score are not returned.
- **RRF**: `score = Σ 1/(60 + rank)` over the BM25 and dense rankings, each fetched with `candidate_k = max(50, top_k)`. Duplicates are removed by `chunk_id`. Identical text in different filings remains as separate evidence because its chunk IDs and provenance differ.
- **Deterministic ordering**: ties are broken by `chunk_id` (BM25, dense), by best single rank and then `chunk_id` (RRF), and by first-stage rank and then `chunk_id` (reranking). Dense scores are rounded to `SCORE_DECIMALS = 6` first, and dense search widens the Qdrant fetch when a tie group crosses the top-k cutoff.
- **Reproducible indexes**: see [Index persistence, reuse and rebuilding](#index-persistence-reuse-and-rebuilding).

Query filters narrow search results within the canonical corpus; they do not create another SEC dataset. `HashingEmbedder` is a deterministic integration-test adapter, not a substitute for semantic embeddings.

## Tests

```bash
PYTHONPATH=Code/src python -m pytest Code/tests/test_retrieval_*.py -v
```

| Test file | Covers |
| --- | --- |
| `test_retrieval_corpus.py` | All canonical chunks against filing metadata, schema validation, conflicting duplicate rejection, exact duplicate collapse and order-independent fingerprinting |
| `test_retrieval_bm25.py` | Tokenization, each filter, corpus-level IDF, empty results, `top_k` validation, tie-breaking, reproducibility, searches reaching every canonical filing, and precomputed filters matching per-chunk filtering |
| `test_retrieval_dense.py` | Qdrant build over every chunk, filters, stable IDs, empty results, stale-index detection, persistence, tie-breaking, input-order independence and identical repeated top-50 rankings; real SentenceTransformer coverage skips when model weights cannot be loaded |
| `test_retrieval_hybrid.py` | RRF formula, deduplication within and across rankings, candidate depth, tie-breaking, filters, empty results, preserved chunks and rebuild reproducibility |
| `test_retrieval_rerank.py` | Reordering by cross-encoder score, configurable depth and top-k, deterministic ties, disabled mode identical to hybrid, failure messages with causes, unusable scores, full-corpus provenance and filters; real cross-encoder tests skip when weights cannot be loaded |

Integration tests require the canonical processed Parquet. Most dense and reranking tests use offline adapters (`HashingEmbedder`, stub scorers) so ordinary runs need no model downloads; semantic-model tests run only when weights are available. On the benchmark machine every retrieval test passed with real models available (no skips).
