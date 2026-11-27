# Question routing baselines (Issues #6 and #18)

[Quick start](../README.md) · [Documentation index](../../README.md#documentation)

Classical routers predict a question's scope (`simple`, `multi_source` or `corpus_wide`) and return a `RouterResult`. The label describes the scope of evidence needed for a complete answer, not which research strategy is best or how difficult a question is. A `simple` question is answered by one filing/section; `multi_source` needs multiple named sources or sections; `corpus_wide` leaves the relevant filers open and requires searching across the collection.

## Components and model architecture

| File | Purpose |
| --- | --- |
| `Code/src/evidenceroute/routing/baselines.py` | scikit-learn pipelines, `BaselineRouter` and CSV split loader |
| `Code/scripts/make_splits.py` | Assigns train/eval groups by `template_id` within each scope, using seed 42 by default |
| `Code/scripts/train_routers.py` | Trains both classifiers, evaluates the supplied eval split and writes metrics and confusion matrices |
| `Code/tests/test_routing_baselines.py` | Offline schema, confidence-range, determinism and invalid-model tests |

Both classifiers use `TfidfVectorizer(lowercase=True, ngram_range=(1, 2))`: words and two-word phrases become features. The `logreg` pipeline uses `LogisticRegression(max_iter=1000, random_state=seed)`. The `naive_bayes` pipeline uses `MultinomialNB(alpha=1.0)`, with add-one smoothing. `BaselineRouter.fit(questions, scopes)` trains the pipeline; `predict` and `predict_many` select the class with the highest `predict_proba` value and return its probability as `RouterResult.confidence`.

These are classical baselines. Their confidence is the classifier's probability, not a guarantee of answer correctness, and the scope prediction alone does not prove which downstream strategy will deliver the best quality for its cost.

## Training data and procedure

Supply a reviewed annotation CSV for the canonical corpus. Training reads `question`, `scope` and `split` (`train` or `eval`); split assignment additionally needs `template_id`. The reviewed full-corpus annotation set is `data/benchmark/questions.csv` (Issue #18); its `split` column is the router split and matches `router_split`. Results from that set are in `Code/results/routing_full/` (see below).

```bash
# Assign splits only when annotations are added or changed; this rewrites the CSV.
python Code/scripts/make_splits.py --questions /path/to/annotations.csv --eval-fraction 0.25 --seed 42

# Fit both classifiers and evaluate the explicit eval split.
PYTHONPATH=Code/src python Code/scripts/train_routers.py --questions /path/to/annotations.csv --out-dir /path/to/results --seed 42

# Test the router implementations without external data or model downloads.
PYTHONPATH=Code/src python -m pytest Code/tests/test_routing_baselines.py -q
```

`make_splits.py` sorts and shuffles templates deterministically within each scope, then assigns whole template groups to eval until the target fraction is reached. Group sizes can make the realized fraction differ from 25%. Keep a template's scope consistent and verify that no template appears in both splits, so a classifier is never evaluated on a rewording of a training question. Fit features and classifiers on the training split only; reserve eval questions for reporting.

`train_routers.py` writes `metrics.json`, `confusion_logreg.csv` and `confusion_naive_bayes.csv` to the requested output directory. The report records the annotation path, seed, train/eval counts and model configurations, plus accuracy, macro F1, per-class precision, recall, F1 and support, and latency (`train_seconds` and `inference_ms_per_question`, measured with `time.perf_counter`; these vary by machine and run, unlike the accuracy metrics). Confusion-matrix rows are true labels and columns are predictions in the order `simple`, `multi_source`, `corpus_wide`.

Review wording for shortcuts such as associating every corpus-wide question with "which companies" or "in the collection". Keep scope and answerability separate: a question can have an identifiable scope while lacking supporting evidence. An absence claim must be checked against the full canonical corpus before it is used as an abstention label.

## Full-corpus results (Issue #18)

Trained on the 44 `train` questions and evaluated on the 21 `eval` questions of `data/benchmark/questions.csv` (seed 42, settings as above, no tuning).

```bash
PYTHONPATH=Code/src python Code/scripts/train_routers.py --questions data/benchmark/questions.csv --out-dir Code/results/routing_full --seed 42
```

| Model | Accuracy | Macro F1 | Training | Inference |
| --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 0.857 (18/21) | 0.859 | about 0.004 s | about 0.02 ms per question |
| TF-IDF + Multinomial Naive Bayes | 0.762 (16/21) | 0.760 | about 0.002 s | about 0.02 ms per question |

Per-class precision / recall (support):

| Scope | Logistic Regression | Naive Bayes |
| --- | --- | --- |
| simple | 0.89 / 0.89 (9) | 0.88 / 0.78 (9) |
| multi_source | 0.75 / 0.86 (7) | 0.60 / 0.86 (7) |
| corpus_wide | 1.00 / 0.80 (5) | 1.00 / 0.60 (5) |

Logistic Regression confusion matrix (rows are true labels):

| True / Predicted | simple | multi_source | corpus_wide |
| --- | --- | --- | --- |
| simple | 8 | 1 | 0 |
| multi_source | 1 | 6 | 0 |
| corpus_wide | 0 | 1 | 4 |

Error analysis:

- Corpus-wide questions without "companies" or "collection" are the main failure. "Does anyone in these filings describe a manufacturing relationship with UMC" was predicted `multi_source` by both models, and "Who buys wafers from GLOBALFOUNDRIES?" by Naive Bayes. Both held-out corpus-wide questions that use "companies" or "collection" were classified correctly, which supports the shortcut concern above.
- Lexical cues mislead in both directions: a one-filing question about Microsoft's headcount "split between the U.S. and other countries" was predicted `multi_source`, and "How do Caterpillar and Deere describe..." (no "compare") was predicted `simple`.
- Confidence on the errors is about 0.4, close to the 1/3 baseline for three classes.

With 21 evaluation questions, one question is about five accuracy points, so the gap between the two models is not statistically meaningful. These are baselines for the planned larger routing dataset and fine-tuned transformer router.

