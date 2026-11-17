# EvidenceRoute — Code and Pipelines

EvidenceRoute uses one canonical SEC corpus: 25 companies, FY2022–2024, 75 Form 10-K filings, 416 extracted sections and 24,766 chunks. `data/companies.csv` is the company registry and `data/filings_manifest.csv` is the filing manifest. Run commands from the repository root with Python 3.10+ (3.11+ recommended).

## Production workflow

1. Install dependencies in an activated virtual environment:

   ```bash
   python -m pip install -r Code/requirements.txt
   ```

2. Set the SEC User-Agent using your real contact address:

   ```bash
   export SEC_USER_AGENT="EvidenceRoute Academic Research your-real-email@university.edu"
   ```

3. Download all 75 filings from the canonical manifest:

   ```bash
   PYTHONPATH=Code/src python Code/scripts/download_filings.py --dry-run
   PYTHONPATH=Code/src python Code/scripts/download_filings.py
   ```

   Original HTML is stored under `data/raw/sec-filings/`. The downloader validates SEC URLs, limits requests, retries temporary failures, and records acquisition hashes in `data/raw/acquisition_results.jsonl`. Local cache reuse requires matching hashes. Optional raw-file publication uses `download_filings.py --s3` and conditional writes under the allocated `raw/sec-filings/` S3 prefix.

4. Process the corpus:

   ```bash
   PYTHONPATH=Code/src python Code/scripts/process_filings.py
   ```

   Outputs are `sections.parquet`, `chunks.parquet`, and `dataset_manifest.json` under `data/processed/full-v1/`. The parser extracts Items 1, 1A, 1C (where present), 7, 8 and 15, preserves financial tables, supports topical filings and resolves financial-statement references into Item 8. Chunks retain exact section offsets and source metadata. The generated manifest records input/output SHA-256 hashes, counts and chunking parameters (`max_chars=1800`, `overlap_chars=200`). Its existing `dataset_name` and `dataset_type` values remain `full` for reproducibility.

   To verify regeneration without replacing existing artifacts, use `--output-dir /tmp/evidenceroute-full-verification`.

5. Run the parsing audit:

   ```bash
   PYTHONPATH=Code/src python Code/scripts/audit_full_corpus.py
   ```

   The validated baseline is 75 PASS, 0 REVIEW, 0 FAIL. The audit checks boundaries, financial tables, metadata, unique IDs and exact chunk offsets. Its report defaults to `data/processed/full-v1/parsing_audit.csv`; use `--output` for a separate report.

6. Verify the three output hashes:

   ```bash
   sha256sum data/processed/full-v1/{sections.parquet,chunks.parquet,dataset_manifest.json}
   ```

   | File | Expected SHA-256 |
   | --- | --- |
   | `sections.parquet` | `4a703731cb31a4c3a7bb4f4113b7fcd0203119fc9fd5fed072b8f6d920350b7d` |
   | `chunks.parquet` | `edc98caf9600da7bebc457cbcd35f64b172027f1af0729e141b5d491b75279fc` |
   | `dataset_manifest.json` | `2f8e5f3bccd1eb679446e712668d80b8f84ddad49861b3ce0c8c09d3c5199e39` |

7. Publish after validation, using the normal AWS credential chain or EC2 IAM role:

   ```bash
   PYTHONPATH=Code/src python Code/scripts/process_filings.py --s3
   ```

   The processed destination is `s3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/full-v1/`. Publishing checks local hashes, preflights remote content, uses conditional creation, verifies uploaded bytes, and writes the manifest last. Matching objects are reused; different existing objects are rejected. No AWS keys belong in source code.

8. Download the shared processed dataset:

   ```bash
   aws s3 sync \
     's3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/full-v1/' \
     data/processed/full-v1/
   ```

   Raw HTML, processed artifacts and local indexes remain excluded from Git.

9. Run the entire test suite:

   ```bash
   PYTHONPATH=Code/src python -m pytest -q
   ```

   Real-filing regressions require original HTML. S3 integrity tests use a fake client and require no credentials. Dense model tests may skip if model weights are unavailable. `test_corpus_regression.py` regenerates all 75 filings into a temporary directory and checks the validated hashes, counts, metadata and offsets. Retrieval and RAG integration tests consume all 24,766 canonical chunks.

## Hybrid retrieval (Issue #4)

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

### Usage

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

### Demo

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

### Design notes

- **Filters**: `None` means no constraint; values within a field are ORed, fields are ANDed. Tickers are case-insensitive, section IDs are lower-cased (`item_1a`). BM25 applies filters before scoring; dense search applies them inside Qdrant.
- **BM25**: Lucene-style IDF `ln(1 + (N - df + 0.5)/(df + 0.5))`, which stays positive in small corpora. IDF is computed over the full corpus, so filtering never changes a chunk's score. Chunks with zero score are not returned.
- **RRF**: `score = Σ 1/(60 + rank)` over the BM25 and dense rankings, each fetched with `candidate_k = max(50, top_k)`. Duplicates are removed by `chunk_id`. Identical text in different filings remains as separate evidence because its chunk IDs and provenance differ.
- **Deterministic ordering**: ties are broken by `chunk_id` (BM25, dense) or by best single rank and then `chunk_id` (RRF). Dense search widens the Qdrant fetch when a tie group crosses the top-k cutoff.
- **Reproducible indexes**: chunks are sorted by `chunk_id`, Qdrant point IDs are UUIDv5 of `chunk_id`, and the collection metadata stores the embedding model, dimension and a SHA-256 fingerprint of the chunks. `ensure_index()` reuses a persisted index only when that manifest matches; otherwise it rebuilds.
- **Qdrant**: local development uses embedded mode (`:memory:` in tests, `data/indexes/` for the demo), so no server is needed. Pass a `QdrantClient(url=...)` to `DenseRetriever` or `build_hybrid_retriever` for a server or Qdrant Cloud. Payload indexes for the filter fields are created on servers and ignored in embedded mode. Qdrant warns about embedded collections above 20,000 points; the 24,766-chunk corpus exceeds that threshold. The implementation also supports a server client for larger deployments.

Query filters narrow search results within the canonical corpus; they do not create another SEC dataset. Generated Qdrant indexes under `data/indexes/` are ignored by Git. `HashingEmbedder` is a deterministic integration-test adapter, not a substitute for semantic embeddings.

### Tests

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

## Question routing baselines (Issue #6)

Classical routers predict a question's scope (`simple`, `multi_source` or `corpus_wide`) and return a `RouterResult`. The label describes the scope of evidence needed for a complete answer, not which research strategy is best or how difficult a question is. A `simple` question is answered by one filing/section; `multi_source` needs multiple named sources or sections; `corpus_wide` leaves the relevant filers open and requires searching across the collection.

### Components and model architecture

| File | Purpose |
| --- | --- |
| `Code/src/evidenceroute/routing/baselines.py` | scikit-learn pipelines, `BaselineRouter` and CSV split loader |
| `Code/scripts/make_splits.py` | Assigns train/eval groups by `template_id` within each scope, using seed 42 by default |
| `Code/scripts/train_routers.py` | Trains both classifiers, evaluates the supplied eval split and writes metrics and confusion matrices |
| `Code/tests/test_routing_baselines.py` | Offline schema, confidence-range, determinism and invalid-model tests |

Both classifiers use `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))`: words and two-word phrases become features. The `logreg` pipeline uses `LogisticRegression(max_iter=1000, random_state=seed)`. The `naive_bayes` pipeline uses `MultinomialNB(alpha=1.0)`, with add-one smoothing. `BaselineRouter.fit(questions, scopes)` trains the pipeline; `predict` and `predict_many` select the class with the highest `predict_proba` value and return its probability as `RouterResult.confidence`.

These are classical baselines. Their confidence is the classifier's probability, not a guarantee of answer correctness, and the scope prediction alone does not prove which downstream strategy will deliver the best quality for its cost.

### Training data and procedure

Supply a reviewed annotation CSV for the canonical corpus. Training reads `question`, `scope` and `split` (`train` or `eval`); split assignment additionally needs `template_id`. No validated full-corpus routing annotation set or routing results are shipped. The paths below are user-supplied annotations and result locations, not additional SEC corpora.

```bash
# Assign splits only when annotations are added or changed; this rewrites the CSV.
python Code/scripts/make_splits.py --questions /path/to/annotations.csv --eval-fraction 0.25 --seed 42

# Fit both classifiers and evaluate the explicit eval split.
PYTHONPATH=Code/src python Code/scripts/train_routers.py --questions /path/to/annotations.csv --out-dir /path/to/results --seed 42

# Test the router implementations without external data or model downloads.
PYTHONPATH=Code/src python -m pytest Code/tests/test_routing_baselines.py -q
```

`make_splits.py` sorts and shuffles templates deterministically within each scope, then assigns whole template groups to eval until the target fraction is reached. Group sizes can make the realized fraction differ from 25%. Keep a template's scope consistent and verify that no template appears in both splits, so a classifier is never evaluated on a rewording of a training question. Fit features and classifiers on the training split only; reserve eval questions for reporting.

`train_routers.py` writes `metrics.json`, `confusion_logreg.csv` and `confusion_naive_bayes.csv` to the requested output directory. The report records the annotation path, seed, train/eval counts and model configurations, plus accuracy, macro F1 and per-class precision, recall, F1 and support. Confusion-matrix rows are true labels and columns are predictions in the order `simple`, `multi_source`, `corpus_wide`.

Review wording for shortcuts such as associating every corpus-wide question with "which companies" or "in the collection". Keep scope and answerability separate: a question can have an identifiable scope while lacking supporting evidence. An absence claim must be checked against the full canonical corpus before it is used as an abstention label.

## Evaluation utilities (Issue #10)

The evaluation package supports retrieval metrics, answer/citation evaluation inputs and question-routing labels. No annotated benchmark, gold answers or evaluation results are currently shipped as validated against the full 75-filing corpus. Corpus integrity and citation-provenance tests validate software behavior; they do not establish benchmark answer correctness.

### Components and data requirements

| File | Purpose |
| --- | --- |
| `Code/src/evidenceroute/evaluation/benchmark.py` | CSV loading, label and split validation, evidence grouping and human-review coverage |
| `Code/src/evidenceroute/evaluation/gold.py` | Quote normalization, location with original section offsets, and fully containing chunk lookup |
| `Code/src/evidenceroute/evaluation/metrics.py` | Evidence-group recall, supporting-chunk precision, reciprocal rank and full coverage at K |
| `Code/scripts/validate_benchmark.py` | Validates explicitly supplied annotation files against the canonical manifest; optionally checks chunk-ID membership in processed data |
| `Code/scripts/remap_gold_evidence.py` | Recomputes locations and chunk IDs from supplied quotes against processed sections and chunks |
| `Code/tests/test_benchmark.py` | Offline tests of metrics, quote matching, containment, label validation, template leakage and evidence-group rules |

Prepare question and evidence CSVs from reviewed original SEC filings, using the canonical manifest's document IDs and URLs. These files are annotations of `data/processed/full-v1/`, not a second dataset mode. The utilities expect these fields:

| Input | Fields |
| --- | --- |
| Questions | `question_id`, `scope`, `answerable`, `eval_split`, `router_split`, `template_id`, `document_ids`, `expected_answer`, `gold_evidence_status`, `quote_verified`; retain the question text for evaluation |
| Evidence | `question_id`, `evidence_group`, `document_id`, `section_id`, `chunk_ids`, `source_url`, `quote`, `relevance`, `mapping_status`, `char_start`, `char_end`, `quote_verified` |

`scope` is `simple`, `multi_source` or `corpus_wide`; `answerable` is `true`, `false` or `pending`. Use `eval_split=dev` for development and tuning and `heldout` for final reporting. Router `eval` questions must be held out. Separate multiple document IDs or chunk IDs with semicolons. `relevance` distinguishes `supporting` evidence from `related` passages that do not answer the question. One evidence group represents one required piece of evidence; overlapping chunks containing the same quote are alternative IDs in that group.

Question evidence statuses are `complete`, `incomplete_section_not_processed`, `none`, `related_only` and `pending_annotation`. `complete` requires mapped supporting evidence and answerability; `multi_source` additionally requires at least two supporting groups. `none` and `related_only` require an unanswerable label, and related-only questions must not have supporting groups. Item 1C is extracted where present; missing evidence should be investigated against the actual filing and processed sections rather than assumed unavailable.

### Validation and remapping

```bash
# Check labels, splits, evidence consistency, manifest provenance and chunk IDs.
PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py --benchmark /path/to/questions.csv --gold /path/to/evidence.csv --processed-dir data/processed/full-v1

# Recompute evidence locations after processing; this rewrites the supplied CSV.
PYTHONPATH=Code/src python Code/scripts/remap_gold_evidence.py --gold /path/to/evidence.csv --processed-dir data/processed/full-v1

# Offline utility tests; these do not require an annotated benchmark or an LLM.
PYTHONPATH=Code/src python -m pytest Code/tests/test_benchmark.py -q
```

The validator checks duplicate question IDs, allowed labels, nonempty expected answers, manifest document membership, matching source URLs for mapped evidence, template leakage across dev/heldout, and consistency between evidence status and supporting groups. With `--processed-dir`, every mapped gold chunk ID must exist in the canonical chunks. It also reports `quote_verified` coverage by split, excluding questions whose answerability is still pending.

Quotes are the remapper's source of truth. For rows marked `mapped` or `section_not_processed`, it searches the recorded section first, then other sections of the same filing. Matching normalizes whitespace, curly quotes and table separators while retaining offsets into the original section text. It records every chunk in that section that fully contains the quote. Missing previously mapped quotes are reported with a nonzero exit, but the supplied CSV is still rewritten, so review its diff and rerun validation. Check for unmapped quotes and empty chunk-ID lists; do not infer verification merely from a successful remap.

### Retrieval metrics and interpretation

`gold_groups` includes supporting chunk IDs only. For each question with complete supporting evidence, score ranked chunk IDs at K (for example K=5 and K=10), then report overall and per-scope aggregates:

- **Recall@K (group level)**: fraction of required evidence groups with at least one chunk in the top K. Overlapping chunks from one quote cannot inflate the number of groups found. Recall without any gold groups raises `ValueError`.
- **Precision@K**: number of relevant supporting chunks in the top K divided by K; returning fewer results does not reduce the denominator.
- **Reciprocal rank@K**: `1 / rank` of the first relevant chunk within the top K, or zero if none is found. Average per-question reciprocal ranks to obtain MRR@K.
- **Full coverage@K**: whether every required group is found. Average these booleans to report the share of questions with complete retrieved evidence.

Answer evaluation also needs factual correctness, completeness, evidence support and citation validity. Track correct abstentions on unanswerable questions, false answers on unanswerable questions, and incorrect abstentions on answerable questions separately. Routing reports should include accuracy, macro F1, per-class precision/recall and confusion matrices; a correct scope label does not establish the best strategy for cost or answer quality.

### Current limitations

The validator checks structure, references and chunk-ID existence; it does not verify quote text at the recorded offsets, prove that a chunk supports an answer, establish completeness across 75 filings, or perform factual answer grading. Review quotes against original HTML, confirm their offsets and containing chunks, and assess negative facts against the entire corpus before marking annotations verified. Remapping preserves human verification flags and never establishes them automatically.

The metrics are callable utilities, not a complete live evaluation runner or an automated factual judge. Standard RAG checks citation provenance and reports factual verification as `not_checked`. The RAG demo uses a mock LLM and actual canonical chunks; it is an integration demonstration, not an evaluation result. Review data quality, template leakage and human agreement before interpreting any future full-corpus benchmark scores.
