# EvidenceRoute — Quick start

One canonical corpus: **25 companies, FY2022–2024, 75 Form 10-K filings, 416 sections and 24,766 chunks**. The company registry is [`data/companies.csv`](../data/companies.csv) and the filing manifest is [`data/filings_manifest.csv`](../data/filings_manifest.csv). Processed artifacts live in `data/processed/full-v1/`.

## Setup and prerequisites

Run commands from the repository root. Use Python 3.10+ (3.11+ recommended) and an activated virtual environment. AWS CLI and normal AWS credentials or an EC2 IAM role are needed only for S3 commands. Dense semantic search needs SentenceTransformer model weights; the mock RAG demo needs no API key.

```bash
python -m pip install -r Code/requirements.txt
export PYTHONPATH=Code/src
export SEC_USER_AGENT="EvidenceRoute Academic Research your-real-email@university.edu"
```

Use your real contact address for SEC requests. For live Gemini generation, set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` as described in the [Standard RAG guide](docs/standard-rag.md#gemini-configuration). Do not put credentials in source code.

## Download, process and validate

```bash
# Plan acquisition without network requests, then download all 75 original filings.
python Code/scripts/download_filings.py --dry-run
python Code/scripts/download_filings.py

# Generate the canonical sections, chunks and dataset manifest.
python Code/scripts/process_filings.py
python Code/scripts/audit_full_corpus.py
sha256sum data/processed/full-v1/{sections.parquet,chunks.parquet,dataset_manifest.json}
```

Original HTML is stored under `data/raw/sec-filings/`. The validated audit baseline is **75 PASS, 0 REVIEW, 0 FAIL**. To check regeneration without replacing existing outputs, use `python Code/scripts/process_filings.py --output-dir /tmp/evidenceroute-full-verification`. See [ingestion](docs/ingestion.md) for parsing, financial tables, Item 1C, chunking, acquisition logs and integrity checks.

## Validated hashes

| File | Expected SHA-256 |
| --- | --- |
| `sections.parquet` | `4a703731cb31a4c3a7bb4f4113b7fcd0203119fc9fd5fed072b8f6d920350b7d` |
| `chunks.parquet` | `edc98caf9600da7bebc457cbcd35f64b172027f1af0729e141b5d491b75279fc` |
| `dataset_manifest.json` | `2f8e5f3bccd1eb679446e712668d80b8f84ddad49861b3ce0c8c09d3c5199e39` |

## S3

```bash
# Publish processed artifacts after auditing and checking hashes.
python Code/scripts/process_filings.py --s3

# Or download the shared processed corpus.
aws s3 sync \
  's3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/full-v1/' \
  data/processed/full-v1/
```

Publishing reuses matching objects and refuses to overwrite different content. Optional original-HTML publication uses `python Code/scripts/download_filings.py --s3` under the allocated `raw/sec-filings/` prefix. See [S3 integrity details](docs/ingestion.md#production-workflow). Raw HTML, processed artifacts and local indexes are excluded from Git.

## Retrieval and RAG demos

```bash
python Code/scripts/demo_retrieval.py "financial results" --mode bm25
python Code/scripts/demo_retrieval.py "supply chain risks" --embedder hashing
python Code/examples/demo_standard_rag.py
```

All demos consume canonical full-v1 chunks. The hashing embedder checks the vector-search integration offline; use the default SentenceTransformer embedder for semantic search. The RAG demo uses a mock LLM. See [retrieval](docs/retrieval.md) for semantic/hybrid examples and [Standard RAG](docs/standard-rag.md) for live Gemini integration and citation behavior.

## Tests

```bash
python -m pytest -q
```

Real-filing regressions require original HTML; corpus integration tests require full-v1 Parquet. S3 integrity tests use a fake client, and semantic-model tests may skip when weights are unavailable. Focused test commands are in the detailed guides.

## Detailed guides

- [Ingestion](docs/ingestion.md): acquisition, parsing, financial statements, Item 1C, chunking, audits, hashes and S3 integrity.
- [Hybrid retrieval — Issue #4](docs/retrieval.md): BM25, embeddings, Qdrant, RRF, filtering, reproducible indexes and tests.
- [Standard RAG — Issue #5](docs/standard-rag.md): prompts, Gemini, citations, abstention, tests and limitations.
- [Question routing — Issue #6](docs/routing.md): TF-IDF classifiers, annotation inputs, train/eval splits, training and metrics.
- [Evaluation — Issue #10](docs/evaluation.md): annotation requirements, evidence mapping, metrics, validation and limitations.

No annotated benchmark, gold answers or routing results are currently shipped as validated against the full corpus. Routing and evaluation utilities require explicit reviewed annotation inputs; see their guides before running them.
