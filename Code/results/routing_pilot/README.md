# Routing Baselines: Pilot Results

These are pilot results from 4 filings (NVIDIA and AMD, FY2023 and FY2024). They check that the pipeline works end to end and are not final conclusions.

Produced by `PYTHONPATH=Code/src python Code/scripts/train_routers.py` (seed 42).

## Setup

| | |
| --- | --- |
| Questions | 63 (21 per scope, including 1 unanswerable per scope) |
| Train / eval | 48 / 15, split by `template_id` so no template appears in both |
| Features | TF-IDF on words and two-word phrases |
| Models | Logistic Regression (`max_iter=1000`), Multinomial Naive Bayes (`alpha=1.0`), default settings otherwise |

## Results

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| TF-IDF + Logistic Regression | 0.667 | 0.634 |
| TF-IDF + Multinomial Naive Bayes | 0.667 | 0.634 |

Per-class precision / recall (identical for both models):

| Scope | Precision | Recall | Support |
| --- | --- | --- | --- |
| simple | 0.50 | 0.20 | 5 |
| multi_source | 0.50 | 0.80 | 5 |
| corpus_wide | 1.00 | 1.00 | 5 |

Confusion matrix (identical for both models; rows are true labels, columns are predictions):

| True / Predicted | simple | multi_source | corpus_wide |
| --- | --- | --- | --- |
| simple | 1 | 4 | 0 |
| multi_source | 1 | 4 | 0 |
| corpus_wide | 0 | 0 | 5 |

## Observations

- **corpus_wide is predicted perfectly, likely through a shortcut.** Almost every corpus_wide question contains "companies" or "in the collection". The models probably learned these phrases rather than question scope. The fine-tuned transformer router should be checked for the same behavior with Integrated Gradients.
- **simple and multi_source are mostly confused.** Both classes name companies and fiscal years, so word-based features cannot easily tell one filing from two. Four of five simple questions were predicted as multi_source.
- **Both models made the same predictions** on every eval question, so on this data the choice of classical classifier does not matter.

## Limitations

- Only 2 companies and 2 fiscal years. corpus_wide questions cover a very small collection.
- 15 eval questions (5 per scope). One question changes per-class recall by 20 points, so differences between models are not meaningful at this size.
- All questions were written by the team from a small set of patterns. Results on human-written or more varied questions will likely be lower.
- No hyperparameter tuning was done; with this little data, tuning would overfit the eval set.
