# EvidenceRoute Evaluation Benchmark

This benchmark is used to evaluate retrieval, answer generation, citations, abstention and question routing. It can be validated without Qdrant, a retrieval model or any LLM calls.

It extends the routing question set from issue #6 (`data/questions/questions.csv`) and links each question to gold evidence in the processed SEC data from issue #3.

## Files

| File | Contents |
| --- | --- |
| `data/eval/benchmark.csv` | One row per question: labels, expected answer, split and evidence status |
| `data/eval/gold_evidence.csv` | One row per evidence passage: filing, section, chunk IDs, character offsets and quote |
| `Code/scripts/remap_gold_evidence.py` | Recomputes section, chunk IDs and offsets from the quotes for any processed dataset |
| `Code/scripts/validate_benchmark.py` | Checks both files and prints review coverage |
| `Code/src/evidenceroute/evaluation/` | Validation code and retrieval metric functions |

## Benchmark fields (`benchmark.csv`)

| Field | Values | Meaning |
| --- | --- | --- |
| `question_id` | `q0001` ... | Unique ID. IDs `q0001` to `q0063` match `questions.csv` |
| `question` | text | The research question |
| `scope` | `simple`, `multi_source`, `corpus_wide` | Evidence scope, using the rules in `data/questions/LABELING.md` |
| `question_type` | `factual`, `comparison`, `multi_document` | Kind of reasoning required |
| `answerable` | `true`, `false`, `pending` | Whether the corpus supports an answer. `pending` is used only for expanded-corpus questions |
| `corpus` | `pilot`, `expanded` | `pilot` questions use the 4 pilot filings. `expanded` questions need filings that are not processed yet |
| `tickers`, `fiscal_years`, `document_ids`, `source_urls` | `;`-separated | Filings needed for a complete answer |
| `expected_answer` | text | Key facts a correct answer must contain, or the expected abstention |
| `eval_split` | `dev`, `heldout` | See "Splits" below |
| `router_split` | `train`, `eval`, empty | Split used by the issue #6 routers. Empty for new questions |
| `template_id` | text | Questions built from the same pattern share an ID |
| `evidence_groups` | integer | Number of evidence groups in `gold_evidence.csv` |
| `gold_evidence_status` | see below | Whether usable gold evidence exists |
| `quote_verified` | `true`, `false` | Whether a person checked the quotes against the original filing |
| `source` | `issue_6`, `new` | Where the question came from |
| `notes` | text | Labeling notes |

`gold_evidence_status` values:

| Value | Meaning | Used for retrieval scoring |
| --- | --- | --- |
| `complete` | Every supporting quote is mapped to a chunk | Yes |
| `incomplete_section_not_processed` | Some evidence is in Item 1C, which the parser does not extract | No |
| `none` | Unanswerable, no evidence | No (abstention scoring only) |
| `related_only` | Unanswerable, but a related passage exists | No (abstention scoring only) |
| `pending_expanded_corpus` | Needs filings not yet processed | No |

## Gold evidence fields (`gold_evidence.csv`)

| Field | Meaning |
| --- | --- |
| `question_id` | Question the passage belongs to |
| `evidence_group` | `g1`, `g2`, ... One group per piece of evidence the answer needs, usually one per filing |
| `document_id`, `ticker`, `fiscal_year`, `source_url` | Filing the passage comes from |
| `section_id` | Processed section, e.g. `item_7` |
| `chunk_ids` | Every chunk that fully contains the quote, `;`-separated. Two IDs means the quote sits in the overlap between neighbouring chunks |
| `char_start`, `char_end` | Position of the quote in the section text |
| `quote` | Exact text from the filing, whitespace-normalized |
| `relevance` | `supporting` (needed for the answer) or `related` (on topic but does not answer the question) |
| `mapping_status` | `mapped`, `section_not_processed`, `no_evidence_unanswerable` or `pending_expanded_corpus` |
| `quote_verified` | Whether a person checked the quote against the original filing |

## How gold evidence was created

1. Each quote was copied from the original SEC filing and checked by a person (`quote_verified = true`).
2. `remap_gold_evidence.py` finds each quote in the processed section text by exact string match (ignoring whitespace, curly quotes and table separators) and records every chunk that fully contains it.
3. Retrieval model outputs were not used to create or select gold evidence.

When the corpus or chunking settings change, rerun step 2. The quotes stay the same; only section, chunk IDs and offsets are recomputed.

## Splits

| Split | Use |
| --- | --- |
| `dev` | Developing and tuning prompts, retrieval settings and routing |
| `heldout` | Final reported results only. Do not tune on it |

- Questions from issue #6 keep their router split: router `train` becomes `dev`, router `eval` becomes `heldout`. This keeps the routers from being tested on their own training data.
- New pilot questions were assigned by template, mostly to `heldout`. All expanded-corpus questions are `heldout`.
- No `template_id` appears in both splits. `validate_benchmark.py` checks this.

## Retrieval metrics

Computed per question for questions with `gold_evidence_status = complete`, then averaged. A retrieved chunk is **relevant** if it appears in the `chunk_ids` of a `supporting` row. `related` rows are never counted as relevant.

**Recall@K (group level).** A question needs every evidence group. A group is found if any of its chunk IDs is in the top K results.

```latex
\text{Recall@K} = \frac{\text{number of evidence groups with at least one chunk in the top } K}{\text{number of evidence groups}}
```

Group-level recall avoids double counting a quote that appears in two overlapping chunks, and gives multi-source questions credit only when every filing is covered.

**Precision@K.** Share of the top K results that are relevant chunks.

```latex
\text{Precision@K} = \frac{\text{relevant chunks in the top } K}{K}
```

**MRR.** Reciprocal rank of the first relevant chunk, 0 if none is in the top K.

```latex
\text{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
```

Report K = 5 and K = 10, overall and by scope. Also report **full coverage@K**: the share of questions where every evidence group is found.

## Answer evaluation

Each generated answer is judged against `expected_answer` and the gold evidence:

| Criterion | Labels | Rule |
| --- | --- | --- |
| Factual correctness | correct, partial, incorrect | Correct if all key facts in `expected_answer` are present and nothing contradicts them. Numbers must match the filing (rounding to the precision in `expected_answer` is accepted) |
| Completeness | complete, incomplete | For multi-source and corpus-wide questions, every evidence group must be addressed |
| Evidence support | supported, partially supported, unsupported | Each factual claim must be supported by a cited passage |
| Citation validity | valid, invalid | A citation is valid if the chunk ID exists in the processed data, comes from the filing it claims, and was in the model's input context |

Automated judging (an LLM judge or NLI model) may be used on `dev`, but a sample of `heldout` answers must be checked by a person and the agreement reported.

## Abstention

For answerable questions the system should answer; for unanswerable questions it should say the evidence is insufficient. Report three counts, not a single abstention rate:

| Outcome | Question type | Counted as |
| --- | --- | --- |
| Correct abstention | unanswerable | correct |
| False answer | unanswerable | error (hallucination) |
| Incorrect abstention | answerable | error (missed answer) |

`related_only` questions (e.g. q0069) test whether the system wrongly uses a related figure, such as revenue instead of operating income.

## Router evaluation

Uses the `scope` labels on `heldout` pilot questions (expanded questions are excluded until their answerability is known).

- Accuracy, macro F1, per-class precision and recall, confusion matrix
- Latency and cost per routing decision (zero cost for the classical routers)
- Report separately for the 15 issue #6 eval questions and the new held-out questions. The new corpus-wide questions avoid the words "companies" and "collection", so a drop on them shows whether a router relies on those words

A route is counted correct when it matches the scope label. Later system evaluation should also check whether the chosen strategy gave the best quality for its cost, since the scope label does not prove which strategy is best.

## Execution metadata (recorded during later system evaluation)

Write one JSON line per question per run to `Code/results/<run_name>/runs.jsonl` with:

| Field | Meaning |
| --- | --- |
| `run_id`, `timestamp`, `git_commit` | Identify the run |
| `question_id` | Benchmark question |
| `route_predicted`, `strategy_executed` | Router output and the strategy actually run |
| `retrieved_chunk_ids` | Ranked list of chunk IDs |
| `answer`, `cited_chunk_ids`, `abstained` | Model output |
| `llm_calls`, `retrieval_calls`, `input_tokens`, `output_tokens` | Work done |
| `latency_ms` | Total, plus per stage (`routing`, `retrieval`, `generation`) |
| `cost_usd` | Estimated cost from token counts |
| `recursion_depth` | For RLM runs |
| `errors`, `retries` | Failures |
| `models` | Model names and versions used |

Latency is reported as p50 and p95 per strategy and scope.

## Review coverage

`validate_benchmark.py` prints the current coverage. At the time of writing, all 15 issue #6 held-out questions are verified; new questions need review. Rerun the script for current numbers.

## Known limitations

- **Item 1C is not processed.** The parser extracts Items 1, 1A, 7, 8 and 15. Three pilot questions (q0041, q0049, q0050) cite Item 1C, so they are excluded from retrieval scoring until it is added.
- **Pilot size.** Two companies and two fiscal years. Corpus-wide questions cover four filings, which is much easier than the planned 25-company corpus.
- **Question source.** Questions were drafted with LLM assistance from a small set of patterns and reviewed by the team. Wording is less varied than real user questions.
- **Expected answers** were written from the verified quotes and should be reviewed alongside them.
- **Negative facts.** Some corpus-wide answers depend on a filing not mentioning something (e.g. NVIDIA filings do not mention GLOBALFOUNDRIES). These were checked by searching the pilot filings and must be rechecked when the corpus grows.

## Commands

Run from the repository root:

```bash
# Process the pilot filings (needed only for chunk ID checks and remapping)
PYTHONPATH=Code/src python Code/scripts/process_filings.py --dataset pilot

# Validate the benchmark and print review coverage
PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py --processed-dir data/processed/pilot-v1

# Recompute chunk IDs and offsets after reprocessing
PYTHONPATH=Code/src python Code/scripts/remap_gold_evidence.py --processed-dir data/processed/pilot-v1

# Tests (no processed data, Qdrant or LLM needed)
PYTHONPATH=Code/src python -m pytest Code/tests/test_benchmark.py -q
```
