# Question Scope Labeling Rules

These rules define how each research question in the EvidenceRoute routing dataset is labeled. The label describes the **scope of evidence a complete answer needs**, not which strategy is best and not how hard the question is.

Pilot corpus: NVIDIA FY2023, NVIDIA FY2024, AMD FY2023, AMD FY2024 (see `data/filings_manifest.csv`).

## The three labels

### `simple`
A complete answer comes from **one filing** (one company, one fiscal year), usually one section, with a single retrieval.

- "What supply-chain risks did NVIDIA disclose in its FY2024 10-K?"
- "How many employees did AMD report at the end of fiscal 2023?"
- "Which reportable segment had the highest revenue in NVIDIA's FY2024 10-K?"

### `multi_source`
A complete answer needs **two or more specific retrievals that can be listed from the question itself**: several named companies, several named years, or several sections. Each piece can be searched separately and then combined.

- "Compare NVIDIA's and AMD's export-control disclosures in FY2024."
- "How did AMD's description of competition change between FY2023 and FY2024?"
- "How does NVIDIA's FY2024 risk discussion of data center demand relate to what it reports in MD&A?"

### `corpus_wide`
The set of relevant filings is **not named in the question**. Answering it completely means checking every filing (or every company) in the collection, because you cannot know in advance which ones are relevant.

- "Which companies introduced a new AI-related risk in FY2024 that was not in their FY2023 filing?"
- "Across all filings, which companies mention reliance on a single foundry?"
- "How many companies in the collection disclose a cybersecurity incident?"

## Decision procedure

Apply in order and stop at the first match:

1. Is the set of relevant companies left open ("which companies", "any company", "how many companies", "across the collection")? Label it **corpus_wide**.
2. Does the question need **more than one** filing or section to answer (several companies, several fiscal years reported in different filings, or several sections)? Label it **multi_source**.
3. Otherwise (one company, one fiscal year), label it **simple**.

## Edge cases

| Situation | Label | Reason |
| --- | --- | --- |
| "Compare NVIDIA and AMD in FY2023 and FY2024" (covers all 4 pilot filings) | multi_source | The filings are listed in the question. Label the question's form, not the pilot's size. |
| "Which company spent more on R&D in FY2024?" | corpus_wide | The candidate companies are not named. With only 2 companies this is small, but the label must still hold on the full 25-company corpus. |
| "...in NVIDIA's latest 10-K" | simple | Resolves to one filing. |
| One filing, two sections (risk factors and MD&A) | multi_source | Two separate retrievals are needed. |
| A change from the prior year that one filing already reports ("revenue was $60.9 billion, up 126% from a year ago") | simple | The comparison is stated inside one filing, so one retrieval answers it. If the answer needs the older filing itself, it is multi_source. |
| A list question inside one filing ("List all risks in Item 1A of AMD FY2024") | simple | Long answer, but still one document and one section. |
| Question the corpus cannot answer (e.g. FY2022 data, or Intel) | Label scope as usual | Set `answerable = false`. Scope and answerability are separate. |

If you still cannot decide, pick the label you think fits, set `ambiguous = true`, and explain in `notes`. Ambiguous questions are reviewed together and logged in `data/questions/labeling_log.md`.

## Avoiding shortcut cues

A classifier should learn scope, not keywords. When writing questions:

- Do not let one word decide the label. Write some `multi_source` questions without "compare", and some `simple` questions that use "which" or "compare" inside a single filing.
- Vary phrasing, length and sentence structure within each class.
- Give questions written from the same template the same `template_id`. All questions sharing a `template_id` must go in the same split (train or eval), never both.
- Use only facts that appear in the original SEC filings. Do not invent filing facts.

## Fields recorded per question

| Field | Description |
| --- | --- |
| `question_id` | Unique ID, e.g. `q0001` |
| `question` | Question text |
| `scope` | `simple`, `multi_source` or `corpus_wide` |
| `question_type` | `factual`, `comparison` or `multi_document` |
| `answerable` | `true` if the pilot corpus contains the answer |
| `document_ids` | Relevant `document_id` values from the manifest, separated by `;` |
| `source_urls` | Matching SEC URLs, separated by `;` |
| `sections` | Expected sections, e.g. `Item 1A; Item 7` |
| `template_id` | Shared ID for questions built from the same pattern |
| `split` | `train` or `eval` |
| `manually_verified` | `true` once a person checked the evidence in the filing |
| `evidence` | Short supporting quote(s) for verified questions |
| `ambiguous` | `true` if the label was hard to decide |
| `notes` | Reasoning for ambiguous labels or other comments |

## Train/eval split

Splits are assigned by `Code/scripts/make_splits.py` (seed 42, about 25% eval per scope). Questions with the same `template_id` always land in the same split, so the classifier is never tested on a rewording of a training question. Each scope gets roughly the same eval share, so all three labels appear in eval.

Pilot result: 16 train and 5 eval questions per scope (48 train, 15 eval).

## Pilot limitations

- Only 2 companies and 2 years, so `corpus_wide` questions cover a small collection and are easier than they will be at full scale.
- Results on this pilot are a sanity check for the pipeline, not final conclusions.
