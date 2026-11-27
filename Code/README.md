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

## Recursive research (RLM)

RLM investigates bounded groups of SEC filings, retains exact excerpts, and reports coverage and uncertainty. It uses the canonical manifest and the full-v1 sections and chunks Parquet files from the setup above. Offline demos require no AWS access.

```bash
export PYTHONPATH=Code/src
python Code/examples/demo_rlm_recursive.py
python Code/examples/demo_rlm_question.py "What foreign exchange risks does Apple disclose in FY2024?"
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus
```

For live runs, use an authorized AWS profile or a temporary access key, secret, and session token via the Boto3 credential chain. Keep credentials outside the repository.

```bash
export BEDROCK_REGION=us-east-1
export BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
export CLAUDE_TIMEOUT_SECONDS=20 CLAUDE_MAX_OUTPUT_TOKENS=1024
```

Generate the overall budgets from the repository root. Each continuation and verification call authorizes a fresh allowance. The built-in `full-corpus` profile supports two companies per batch and a final singleton. Its per-tree limits are listed in the [RLM guide](docs/rlm.md#corpus-wide-acceptance-follow-up).

```bash
python - <<'PY'
import json
from pathlib import Path

for name, batches, attempts, tokens, calls, groups, seconds in [
    ("single", 1, 18, 100000, 12, 2, 180),
    ("full", 13, 234, 1300000, 156, 25, 2340),
    ("constrained", 13, 72, 400000, 48, 6, 720),
    ("continuation", 10, 180, 1000000, 120, 19, 1800),
]:
    limits = dict(max_batches=batches, max_provider_attempts=attempts, max_reported_tokens=tokens,
                  max_corpus_attempts=calls, max_delegations=groups, max_partitions=2 * groups,
                  deadline_seconds=float(seconds), max_scan_chunks=3038, max_scan_pages=328,
                  scan_deadline_seconds=10.0)
    Path(f"rlm-milestone-{name}-limits.json").write_text(json.dumps(limits, indent=2) + "\n")
verify = dict(max_provider_attempts=8, max_tokens=32000, max_output_tokens=1024, max_items=16,
              batch_size=2, max_input_bytes=16000, deadline_seconds=120.0)
Path("rlm-milestone-verify-limits.json").write_text(json.dumps(verify, indent=2) + "\n")
PY
```

The first live command is a bounded AMD comparison. The next two run and resume the full corpus. Omitting `--company` on the initial call sets the denominator to 25 companies and 50 filing sections. Exit 1 means the report is partial. Inspect the checkpoint before authorizing another pass.

```bash
python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live --trace \
  --company AMD --limits-file rlm-milestone-single-limits.json \
  --save-checkpoint rlm-amd-live-checkpoint.json > rlm-amd-live.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live --trace \
  --limits-file rlm-milestone-constrained-limits.json \
  --save-checkpoint rlm-full-live-checkpoint.json > rlm-full-live-initial.json

python Code/examples/demo_rlm_corpus_research.py --profile full-corpus --live \
  --resume rlm-full-live-checkpoint.json --limits-file rlm-milestone-continuation-limits.json \
  --verification-limits-file rlm-milestone-verify-limits.json \
  --save-checkpoint rlm-full-live-checkpoint.json > rlm-full-live-continued.json
```

Use `rlm-milestone-full-limits.json` for a single larger pass. Checkpoint saves occur at pass boundaries. Default continuation selects unresolved assignments. To investigate a specific verification gap, set `--resume`, `--company`, and `--goal` with fresh research and verification allowances. See the [small verified comparison](docs/rlm.md#live-validation-commands).

For the quality-oriented comparison, select `--profile full-corpus-quality` and use its larger explicit limits. The economical `full-corpus` profile remains available for controlled runs. Use `demo_rlm_verification.py --outstanding --checkpoint FILE --limits-file FILE --save-checkpoint FILE` for a separate verification pass over outstanding cases. Check `assignment_evidence` and `outstanding_verification` in the report. Add `--live` to authorize Bedrock calls. [Quality limits and live commands](docs/rlm.md#next-live-validation)

### Python integration

This live example creates the corpus and runner once. Reuse both across benchmark questions. Use a separate checkpoint path and harness ID for each question. Each invocation gets a fresh model client.

```python
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.checkpoint import create_checkpoint, save_checkpoint
from evidenceroute.research.rlm.corpus_research import CorpusLimits
from evidenceroute.research.rlm.interpretation import InterpretationLimits
from evidenceroute.research.rlm.natural_language import research_question
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.verification import BedrockSemanticVerifier, review_checkpoint, verification_report
from evidenceroute.research.rlm.verification_records import VerificationLimits

corpus = CorpusEnvironment()
profile = COMPARATIVE_PROFILES["full-corpus"]
runner = RecursiveResearch(
    corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
    shared_limits=profile.shared_limits, local_limits=profile.local_limits,
    finalization=profile.finalization, comparative_capture=profile.comparative_capture,
    max_root_reinvestigation_calls=profile.max_root_reinvestigation_calls,
)
limits = CorpusLimits(max_batches=1, max_provider_attempts=20, max_reported_tokens=116000,
                      max_corpus_attempts=12, max_delegations=2, max_partitions=4, deadline_seconds=210.0)
interpretation = InterpretationLimits(max_provider_attempts=2, max_tokens=16000, deadline_seconds=30.0)
result = research_question(
    "Did AMD disclose a new AI infrastructure risk in FY2024 compared with FY2023?",
    runner, limits=limits, interpretation_limits=interpretation, max_groups=profile.max_groups,
)
if result.research is None:
    print(result.interpretation.outcome, result.usage)
else:
    checkpoint = create_checkpoint(corpus, result)
    save_checkpoint("rlm-benchmark-example.json", checkpoint)
    checkpoint = review_checkpoint(
        checkpoint, corpus, BedrockSemanticVerifier.from_env(disable_retries=True),
        limits=VerificationLimits(max_provider_attempts=4, max_tokens=16000, max_items=8, deadline_seconds=60.0),
    )
    save_checkpoint("rlm-benchmark-example.json", checkpoint)
    report = verification_report(checkpoint)
    print(report["answer"], report["citations"], report["conclusion_citations"])
    print(checkpoint.merged_coverage(corpus), report["evidence_gaps"], report["items"])
    print(report["execution_status"], report["research_status"], report["combined_usage"])
```

To resume a saved full-corpus question from Python, call `load_checkpoint(path, corpus)` and pass the result to `continue_research(checkpoint, runner, limits=new_limits, max_groups=profile.max_groups)` from the checkpoint module. You can then save and run `review_checkpoint` with a separate budget. Pass `companies=("AMD",)` and a specific `goal` to reinvestigate a completed assignment.

For descriptive or legacy runs without checkpoint review, use `result.research.answer`, `citations`, `coverage`, `groups`, and the execution and research statuses. `result.usage` includes interpretation. For resumed or reviewed runs use the unified report and merged coverage. `citations` holds historical SEC references, and `conclusion_citations` holds evidence cited by verifier-supported comparative components. `items` contains review verdicts and their exact evidence pointers. Store the original comparisons and findings for audit. Finished execution can leave novelty and coverage unresolved.

```bash
AWS_EC2_METADATA_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q Code/tests/test_rlm_*.py
```

[RLM architecture and experiments](docs/rlm.md) · [GitHub issue #16](https://github.com/ibrahimhamza01/EvidenceRoute/issues/16)


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
