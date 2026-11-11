# EvidenceRoute — Code and Pipelines

This patch adds a manifest-driven raw HTML downloader. It does not include or overwrite `Code/src/evidenceroute/schemas.py` or existing Git history. The two `data/*.csv` files are copies of the four-row pilot manifest previously provided.

## Prerequisites

- Python 3.10+ (recommended: 3.11+), an activated virtual environment, and working EC2 IAM credentials for the given S3 prefix.
- Your existing `Code/src/evidenceroute/schemas.py` must use `Filing` with `document_id`, `company_name`, `ticker`, `cik`, `accession_number`, `form_type`, `fiscal_year`, `report_period_end: date`, `filing_date: date`, `primary_document`, `source_url` and optional `raw_sha256`. Other models may remain unchanged. Pydantic 2 is required.
- Run commands from the **repository root**.

## Install and test

```bash
python -m pip install -r Code/requirements.txt
PYTHONPATH=Code/src python -m pytest Code/tests/test_downloader.py -q
```

## Preflight / dry run (no HTTP, S3, or file modifications)

```bash
PYTHONPATH=Code/src python Code/scripts/download_filings.py --dry-run --s3
```

## Download SEC pilot locally

Set the SEC User-Agent to your *real* contact email, not the sample placeholder. SEC requires identifiable automated requests.

```bash
export SEC_USER_AGENT="EvidenceRoute Academic Research your-real-email@university.edu"
PYTHONPATH=Code/src python Code/scripts/download_filings.py
```

## Publish to existing S3 prefix (uses EC2's AWS credential chain / IAM role)

```bash
PYTHONPATH=Code/src python Code/scripts/download_filings.py --s3
```

By default, `--s3` targets:

` s3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/raw/sec-filings/ `

No AWS access keys belong in source code. You can override using `--s3-bucket` and `--s3-prefix`, or `EVIDENCEROUTE_S3_BUCKET` / `EVIDENCEROUTE_S3_PREFIX`. To start with one pilot document, add `--ticker NVDA --year 2024`.

## Expected outputs

- `data/raw/sec-filings/<10-digit-cik>/<accession-without-hyphens>/<primary-document>.htm`: original bytes.
- `data/raw/acquisition_results.jsonl`: append-only local acquisition history with checksum, size, source URL, file path, S3 URI, status, and UTC timestamp.
- S3 objects under the assigned prefix with the `sha256` and `document-id` user metadata.

The CSVs are **inputs** and may be tracked in Git. Ignore `data/raw/` and `data/processed/` in Git. The local JSONL acquisition log is untracked, so back it up if you want cache reuse across machines, or preserve hashes in an explicit, reviewed acquisition manifest for fully cross-machine reproducibility. The original raw S3 objects are not overwritten by this tool. A changed existing S3 object fails verification.

## Safety/integrity design

- CSV validated before any network work. Canonical `https://www.sec.gov` archive URLs are required; arbitrary URL fetching is rejected.
- One request at a time, at most 2 per second; retries for network errors, HTTP 429 and 5xx, and respect for integer `Retry-After` values.
- Redirects must end on SEC domains. Suspicious small, non-HTML or blocked responses are rejected.
- Downloads stream to temporary files; SHA-256 is calculated before atomic local replacement.
- A cached file is trusted only when its checksum matches the manifest or an earlier successful local acquisition log entry.
- S3 writes use `If-None-Match: *` to avoid overwriting an existing object. Existing objects are reused only when stored SHA-256 and content length match.
- A failed or partial run exits nonzero; successful filings remain available for retry.

## What this does not do

It does **not** select filings from the SEC submissions API, parse or chunk HTML, generate Parquet, or validate fiscal-year metadata inside the HTML. The pilot is based on the reviewed frozen manifest. Those are separate stages after the downloader succeeds.

## Question routing baselines (issue #6)

Classical routers that predict a question's scope (`simple`, `multi_source` or `corpus_wide`) and return a `RouterResult`.

| File | Purpose |
| --- | --- |
| `data/questions/LABELING.md` | Labeling rules, edge cases and split method |
| `data/questions/questions.csv` | 63 pilot questions with labels, evidence and train/eval split |
| `data/questions/labeling_log.md` | Ambiguous questions and labeling decisions |
| `Code/scripts/make_splits.py` | Assigns train/eval by `template_id` (seed 42) |
| `Code/src/evidenceroute/routing/baselines.py` | TF-IDF + Logistic Regression and TF-IDF + Multinomial Naive Bayes |
| `Code/scripts/train_routers.py` | Trains both routers and saves evaluation results |
| `Code/tests/test_routing_baselines.py` | Offline tests for the routers |
| `Code/results/routing_pilot/` | Pilot metrics and confusion matrices |

Run from the repository root:

```bash
python -m pip install -r Code/requirements.txt
PYTHONPATH=Code/src python -m pytest Code/tests/test_routing_baselines.py -q
PYTHONPATH=Code/src python Code/scripts/train_routers.py
```

`make_splits.py` only needs to be rerun if questions are added or changed:

```bash
python Code/scripts/make_splits.py
```

## SEC 10-K processing (Issue #3)

The processing pipeline converts downloaded SEC Form 10-K HTML filings into structured sections and citation-ready chunks.

Run commands from the repository root with `PYTHONPATH=Code/src`.

### Processing

The pipeline reads `data/filings_manifest.csv` and the original HTML files under `data/raw/sec-filings/`.

It extracts Items 1, 1A, 7, 8, and 15, preserves financial statement tables as normalized text, and generates Parquet datasets with source metadata and exact character offsets.

NVIDIA's pilot filings contain their consolidated financial statements under Item 15. AMD's pilot filings contain them under Item 8. Both Items are retained without relocating content between sections.

### Pilot dataset

The four-filing pilot consists of NVIDIA and AMD FY2023 and FY2024.

```bash
PYTHONPATH=Code/src python Code/scripts/process_filings.py --dataset pilot
```

Outputs:

```text
data/processed/pilot-v1/
├── sections.parquet
├── chunks.parquet
└── dataset_manifest.json
```

The validated pilot contains 4 filings, 20 sections, and 994 chunks with the default chunking configuration.

`dataset_manifest.json` is generated automatically by the processing script. It records dataset identity, input SHA-256 hashes, output SHA-256 hashes, processing parameters, and record counts. It is not maintained manually or committed to Git.

### Full dataset

`pilot-v1` is a temporary validation dataset name, not a permanent production directory.

When the full filing collection is ready, run:

```bash
PYTHONPATH=Code/src python Code/scripts/process_filings.py \
  --dataset full \
  --manifest data/full_filings_manifest.csv
```

The full manifest path above is illustrative; the complete manifest must exist before running this command.

Full dataset outputs are written directly under `data/processed/`:

```text
data/processed/
├── sections.parquet
├── chunks.parquet
└── dataset_manifest.json
```

The full dataset does not use pilot-specific output directories. The dataset manifest records `dataset_type: "full"`.

### S3 storage

The project uses the existing S3 bucket and prefix:

```text
s3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/
```

Raw SEC filings are stored under `raw/sec-filings/`.

Processed pilot data is published under `processed/pilot-v1/`. Full dataset outputs are published directly under `processed/`.

Publish the pilot after validation:

```bash
PYTHONPATH=Code/src python Code/scripts/process_filings.py \
  --dataset pilot --s3
```

Publishing is opt-in. The script writes the local dataset first, verifies checksums, and publishes `sections.parquet`, `chunks.parquet`, and `dataset_manifest.json`. Existing S3 objects with different contents are not overwritten.

Team members can download the published pilot dataset using:

```bash
aws s3 sync \
  's3://dats-nlp/syedibrahim.hamza@gwu.edu/EvidenceRoute/processed/pilot-v1/' \
  data/processed/pilot-v1/
```

The original HTML and generated datasets remain excluded from Git. S3 provides shared storage; the processing script supports local regeneration.

### Tests

```bash
PYTHONPATH=Code/src python -m pytest \
  Code/tests/test_filing_processing.py \
  Code/tests/test_pilot_regression.py -v
```

`test_filing_processing.py` covers unit-level parsing, chunking, and serialization behavior.

`test_pilot_regression.py` covers the four real SEC filings, financial statement extraction, exact chunk offsets, automatic dataset manifest generation, and S3 publishing behavior using a fake S3 client.

Real-filing tests require the downloaded pilot HTML files. S3 publishing tests do not require AWS credentials.
