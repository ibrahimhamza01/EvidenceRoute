# Evaluation utilities and benchmark (Issues #10 and #18)

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

The evaluation package supports retrieval metrics, answer/citation evaluation inputs and question-routing labels. The reviewed full-corpus benchmark is in `data/benchmark/` (Issue #18, described below). Corpus integrity and citation-provenance tests validate software behavior; they do not establish benchmark answer correctness.

## Full-corpus benchmark (Issue #18)

| File | Contents |
| --- | --- |
| `data/benchmark/questions.csv` | 65 questions with scope, answerability, expected answer, companies, fiscal years, document IDs, source URLs, splits, evidence status and review fields |
| `data/benchmark/gold_evidence.csv` | 142 evidence passages with document, section and chunk IDs, character offsets, quote and relevance |

| | simple | multi_source | corpus_wide | Total |
| --- | --- | --- | --- | --- |
| Questions | 25 | 23 | 17 | 65 |
| dev / heldout | 16 / 9 | 16 / 7 | 12 / 5 | 44 / 21 |
| Unanswerable | 3 | 1 | 2 | 6 |

All 25 companies and fiscal years 2022 to 2024 appear in the gold evidence. Question types include single-filing facts, multi-company comparisons, multi-year comparisons, two-section questions within one filing, and corpus-wide investigations.

**Construction.** Questions were drafted with LLM assistance and checked against the processed corpus. Twenty-two were reused from the earlier pilot set after revalidation against full-v1 (`source = reused:<old id>`); pilot answers that changed with 25 companies were rewritten or dropped. Every quote was located by exact normalized match in the processed sections, and the build fails if any quote is not found or no chunk fully contains it. Retrieval model outputs were not used to create gold evidence.

**Labeling guidelines.**

- `simple`: one filing (one company, one fiscal year) answers the question. A prior-year comparison stated inside that filing stays `simple`.
- `multi_source`: the question names two or more filings or sections, for example two companies, two fiscal years reported in separate filings, or two sections of one filing.
- `corpus_wide`: the set of relevant filers is left open ("who", "which filers", "does anyone"), even when limited to a sector. Some corpus-wide questions deliberately avoid "companies" and "collection" to test lexical shortcuts.
- Label the wording, not the size of the corpus, and keep scope separate from answerability.
- Fiscal years: the manifest `fiscal_year` is the calendar year in which the fiscal period ends. Company naming can differ (Home Depot calls the year ended January 28, 2024 "fiscal 2023"), so questions state the company's own wording or the period-end date.
- Mentions are not relationships: a filer that names a company only as a competitor, investor or auditor location does not count as a supplier or headquarters answer. Such passages are recorded with `relevance = related`.
- Unanswerable questions cover years outside the corpus, companies outside the corpus, and related-only cases where the filing reports a nearby figure but not the one asked for.

**Additional fields.** Besides the fields required below, `questions.csv` stores `question`, `question_type`, `tickers`, `fiscal_years`, `source_urls`, `split` (router split, `train`/`eval`), `evidence_groups`, `review_status`, `reviewer_notes`, `source` and `notes`. `gold_evidence.csv` also stores `ticker` and `fiscal_year`. One evidence group corresponds to one required piece of evidence; corpus-wide questions use one group per qualifying company.

**Splits.** `Code/scripts/make_splits.py --eval-fraction 0.3 --seed 42` assigns whole template groups within each scope. Router `eval` questions are `heldout` and router `train` questions are `dev`, so routers are never scored on training questions and no `template_id` appears in both splits.

**Review.** All 65 questions and their evidence were checked against the original SEC filings (`quote_verified = true`, `review_status = approved`). A second review pass found problems in an earlier draft, mostly corpus-wide answers whose keyword searches missed differently worded passages (for example Intel naming TSMC as a manufacturer in its risk factors, and ExxonMobil reporting headcount "in thousands"). These were corrected before approval.

```bash
PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py --benchmark data/benchmark/questions.csv --gold data/benchmark/gold_evidence.csv --processed-dir data/processed/full-v1
```

Validation passes with 44/44 dev and 21/21 heldout questions verified.

## Components and data requirements

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

## Validation and remapping

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

## Retrieval metrics and interpretation

`gold_groups` includes supporting chunk IDs only. For each question with complete supporting evidence, score ranked chunk IDs at K (for example K=5 and K=10), then report overall and per-scope aggregates:

- **Recall@K (group level)**: fraction of required evidence groups with at least one chunk in the top K. Overlapping chunks from one quote cannot inflate the number of groups found. Recall without any gold groups raises `ValueError`.
- **Precision@K**: number of relevant supporting chunks in the top K divided by K; returning fewer results does not reduce the denominator.
- **Reciprocal rank@K**: `1 / rank` of the first relevant chunk within the top K, or zero if none is found. Average per-question reciprocal ranks to obtain MRR@K.
- **Full coverage@K**: whether every required group is found. Average these booleans to report the share of questions with complete retrieved evidence.

Answer evaluation also needs factual correctness, completeness, evidence support and citation validity. Track correct abstentions on unanswerable questions, false answers on unanswerable questions, and incorrect abstentions on answerable questions separately. Routing reports should include accuracy, macro F1, per-class precision/recall and confusion matrices; a correct scope label does not establish the best strategy for cost or answer quality.

## Current limitations

Benchmark-specific limits: corpus-wide absence claims were checked by keyword and phrase search over all 75 filings plus human review, which can still miss unusual wording; 65 questions give wide confidence intervals; Item 1C cross-references (JPMorganChase, Bank of America, Goldman Sachs FY2023) are counted as present; and question wording comes from a limited set of patterns, so it is less varied than real user questions.


The validator checks structure, references and chunk-ID existence; it does not verify quote text at the recorded offsets, prove that a chunk supports an answer, establish completeness across 75 filings, or perform factual answer grading. Review quotes against original HTML, confirm their offsets and containing chunks, and assess negative facts against the entire corpus before marking annotations verified. Remapping preserves human verification flags and never establishes them automatically.

The metrics are callable utilities, not a complete live evaluation runner or an automated factual judge. Standard RAG checks citation provenance and reports factual verification as `not_checked`. The RAG demo uses a mock LLM and actual canonical chunks; it is an integration demonstration, not an evaluation result. Review data quality, template leakage and human agreement before interpreting any future full-corpus benchmark scores.
