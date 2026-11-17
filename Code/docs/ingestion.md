# SEC acquisition and ingestion

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

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

   Expected values are listed in the [quick-start checksum table](../README.md#validated-hashes). Compare all three hashes; matching record counts alone does not establish byte-for-byte reproducibility.

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

## Parsing and financial-table semantics

Sections contain normalized text rather than HTML. Financial table rows preserve their cell values with ` | ` separators. Chunk offsets address that normalized section text, not character positions in the original HTML bytes. Financial-statement cross-references are resolved into Item 8, including filings whose statements are located elsewhere in the annual report; the recovered statements are not duplicated in the exhibit section. Topic-organized filings use their cross-reference index to identify the required sections.

Item 1C is retained when present. Earlier filings without this item do not receive fabricated cybersecurity sections. The validated corpus has 41 Item 1C sections across its 75 filings. Parsing fails on missing or ambiguous required boundaries rather than silently exporting incomplete sections.

The parser and processor are covered by `Code/tests/test_filing_processing.py`, `test_financial_attribution_real.py`, `test_full_financial_references.py`, `test_topical_parser.py` and `test_corpus_regression.py`. The full-corpus regeneration test writes to a temporary directory, validates every chunk's offsets and metadata, and compares all three output hashes against the validated baseline.
