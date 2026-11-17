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

## Retrieval, routing and evaluation

Retrieval loads `data/processed/full-v1/chunks.parquet` and supports BM25, dense embeddings with Qdrant, and Reciprocal Rank Fusion. Query filters select search results without constructing another SEC dataset.

```bash
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "financial results" --mode bm25
PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "supply chain risks" --embedder hashing
PYTHONPATH=Code/src python Code/examples/demo_standard_rag.py
```

The hashing embedder is an offline integration check; semantic search uses SentenceTransformer weights. Standard RAG validates citation provenance and reports factual verification as `not_checked`.

General routing models, train/eval splitting, benchmark validation, quote location, remapping and retrieval metrics remain available. No annotated benchmark, gold answers or routing results are currently shipped as validated against the full corpus. New annotations must be checked against the canonical corpus before evaluation.

Annotation tools require explicit inputs:

```bash
PYTHONPATH=Code/src python Code/scripts/train_routers.py --questions /path/to/annotations.csv --out-dir /path/to/results
python Code/scripts/make_splits.py --questions /path/to/annotations.csv
PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py --benchmark /path/to/questions.csv --gold /path/to/evidence.csv --processed-dir data/processed/full-v1
PYTHONPATH=Code/src python Code/scripts/remap_gold_evidence.py --gold /path/to/evidence.csv --processed-dir data/processed/full-v1
```

Benchmark validation uses question scope, answerability, evidence status and split fields without dataset-mode distinctions. Remapping locates supplied quotes and never establishes human verification or answer correctness.
