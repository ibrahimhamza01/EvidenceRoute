# EvidenceRoute — 10-K acquisition patch

This patch adds a manifest-driven raw HTML downloader. It does not include or overwrite `Code/src/evidenceroute/schemas.py` or existing Git history. The two `data/*.csv` files are copies of the four-row pilot manifest previously provided.

## Prerequisites

- Python 3.10+ (recommended: 3.11+), an activated virtual environment, and working EC2 IAM credentials for the given S3 prefix.
- Your existing `Code/src/evidenceroute/schemas.py` must use `Filing` with `document_id`, `company_name`, `ticker`, `cik`, `accession_number`, `form_type`, `fiscal_year`, `report_period_end: date`, `filing_date: date`, `primary_document`, `source_url` and optional `raw_sha256`. Other models may remain unchanged. Pydantic 2 is required.
- Run commands from the **repository root**.

## Install and test

```bash
python -m pip install -r Code/requirements-acquisition.txt
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
python -m pip install -r Code/requirements-routing.txt
PYTHONPATH=Code/src python -m pytest Code/tests/test_routing_baselines.py -q
PYTHONPATH=Code/src python Code/scripts/train_routers.py
```

`make_splits.py` only needs to be rerun if questions are added or changed:

```bash
python Code/scripts/make_splits.py
```
