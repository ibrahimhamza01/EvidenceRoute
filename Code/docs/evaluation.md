# Evaluation utilities (Issue #10)

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

The evaluation package supports retrieval metrics, answer/citation evaluation inputs and question-routing labels. No annotated benchmark, gold answers or evaluation results are currently shipped as validated against the full 75-filing corpus. Corpus integrity and citation-provenance tests validate software behavior; they do not establish benchmark answer correctness.

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

The validator checks structure, references and chunk-ID existence; it does not verify quote text at the recorded offsets, prove that a chunk supports an answer, establish completeness across 75 filings, or perform factual answer grading. Review quotes against original HTML, confirm their offsets and containing chunks, and assess negative facts against the entire corpus before marking annotations verified. Remapping preserves human verification flags and never establishes them automatically.

The metrics are callable utilities, not a complete live evaluation runner or an automated factual judge. Standard RAG checks citation provenance and reports factual verification as `not_checked`. The RAG demo uses a mock LLM and actual canonical chunks; it is an integration demonstration, not an evaluation result. Review data quality, template leakage and human agreement before interpreting any future full-corpus benchmark scores.
