# Hybrid retrieval (Issue #4)

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

BM25 lexical search, dense semantic search (SentenceTransformer embeddings stored in Qdrant), and Reciprocal Rank Fusion over validated `Chunk` records. The retrieval core consumes `Chunk` objects only and does not import the HTML parser. The production loader and demo read all 24,766 chunks from `data/processed/full-v1/chunks.parquet`, covering the canonical 75 filings.

| File | Purpose |
| --- | --- |
| `Code/src/evidenceroute/retrieval/corpus.py` | `ChunkCorpus`: validation, duplicate handling, `chunk_id` ordering, fingerprint; JSONL and Parquet loaders |
| `Code/src/evidenceroute/retrieval/filters.py` | `ChunkFilter` by ticker, fiscal year, document ID and section ID |
| `Code/src/evidenceroute/retrieval/bm25.py` | `BM25Retriever` (Okapi BM25, non-negative IDF) |
| `Code/src/evidenceroute/retrieval/dense.py` | `DenseRetriever`, `SentenceTransformerEmbedder`, Qdrant index build and staleness checks |
| `Code/src/evidenceroute/retrieval/hybrid.py` | `reciprocal_rank_fusion`, `HybridRetriever` |
| `Code/src/evidenceroute/retrieval/factory.py` | `build_hybrid_retriever` wiring all three from one corpus |
| `Code/scripts/demo_retrieval.py` | Command-line search demo |

## Usage

```python
from evidenceroute.retrieval import ChunkFilter, load_chunks
from evidenceroute.retrieval.dense import SentenceTransformerEmbedder
from evidenceroute.retrieval.factory import build_hybrid_retriever

chunks = load_chunks("data/processed/full-v1/chunks.parquet")
retriever = build_hybrid_retriever(chunks, SentenceTransformerEmbedder(), index_path="data/indexes/full-v1")
results = retriever.search("export restrictions on AI GPUs", top_k=5,
                           filters=ChunkFilter(tickers="NVDA", fiscal_years=2024, section_ids="item_1a"))
# -> list[RetrievalResult]; each result.chunk preserves the original text, ID and source URL
```

All three retrievers expose `search(query, top_k, filters) -> list[RetrievalResult]`. `HybridRetriever.search_with_details` also returns each chunk's BM25 and dense ranks for analysis.

## Demo

```bash
python -m pip install -r Code/requirements.txt
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "H100 export license" --mode bm25
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py \
  "reliance on outside chip manufacturers" --ticker AMD --year 2024 --show-ranks
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "data center revenue growth" --section item_7 --top-k 3
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "supply chain risks" --embedder hashing
PYTHONPATH=Code/src python Code/examples/demo_standard_rag.py
```

The first dense or hybrid run loads the embedding model, downloading it if not cached (`BAAI/bge-small-en-v1.5` by default; change with `--embedder`). `--embedder hashing` runs the Qdrant path offline with a non-semantic hashing embedder, for smoke tests only. Other options: `--document-id`, repeatable `--ticker/--year/--section`, `--json`, `--rebuild`, `--chunks`.

## Design notes

- **Filters**: `None` means no constraint; values within a field are ORed, fields are ANDed. Tickers are case-insensitive, section IDs are lower-cased (`item_1a`). BM25 applies filters before scoring; dense search applies them inside Qdrant.
- **BM25**: Lucene-style IDF `ln(1 + (N - df + 0.5)/(df + 0.5))`, which stays positive in small corpora. IDF is computed over the full corpus, so filtering never changes a chunk's score. Chunks with zero score are not returned.
- **RRF**: `score = Σ 1/(60 + rank)` over the BM25 and dense rankings, each fetched with `candidate_k = max(50, top_k)`. Duplicates are removed by `chunk_id`. Identical text in different filings remains as separate evidence because its chunk IDs and provenance differ.
- **Deterministic ordering**: ties are broken by `chunk_id` (BM25, dense) or by best single rank and then `chunk_id` (RRF). Dense search widens the Qdrant fetch when a tie group crosses the top-k cutoff.
- **Reproducible indexes**: chunks are sorted by `chunk_id`, Qdrant point IDs are UUIDv5 of `chunk_id`, and the collection metadata stores the embedding model, dimension and a SHA-256 fingerprint of the chunks. `ensure_index()` reuses a persisted index only when that manifest matches; otherwise it rebuilds.
- **Qdrant**: local development uses embedded mode (`:memory:` in tests, `data/indexes/` for the demo), so no server is needed. Pass a `QdrantClient(url=...)` to `DenseRetriever` or `build_hybrid_retriever` for a server or Qdrant Cloud. Payload indexes for the filter fields are created on servers and ignored in embedded mode. Qdrant warns about embedded collections above 20,000 points; the 24,766-chunk corpus exceeds that threshold. The implementation also supports a server client for larger deployments.

Query filters narrow search results within the canonical corpus; they do not create another SEC dataset. Generated Qdrant indexes under `data/indexes/` are ignored by Git. `HashingEmbedder` is a deterministic integration-test adapter, not a substitute for semantic embeddings.

## Tests

```bash
PYTHONPATH=Code/src python -m pytest Code/tests/test_retrieval_*.py -v
```

| Test file | Covers |
| --- | --- |
| `test_retrieval_corpus.py` | All canonical chunks against filing metadata, schema validation, conflicting duplicate rejection, exact duplicate collapse and order-independent fingerprinting |
| `test_retrieval_bm25.py` | Tokenization, each filter, corpus-level IDF, empty results, `top_k` validation, tie-breaking, reproducibility and searches reaching every canonical filing |
| `test_retrieval_dense.py` | Qdrant build over every chunk, filters, stable IDs, empty results, stale-index detection, persistence, tie-breaking and input-order independence; real SentenceTransformer coverage skips when model weights cannot be loaded |
| `test_retrieval_hybrid.py` | RRF formula, deduplication within and across rankings, candidate depth, tie-breaking, filters, empty results, preserved chunks and rebuild reproducibility |

Integration tests require the canonical processed Parquet. Most dense tests use the offline hashing embedder; semantic-model tests require SentenceTransformer weights. Index reuse depends on the recorded corpus and embedding configuration, so rerun or rebuild the index when either changes.
