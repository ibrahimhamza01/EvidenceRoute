# Question routing baselines (Issue #6)

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

Supply a reviewed annotation CSV for the canonical corpus. Training reads `question`, `scope` and `split` (`train` or `eval`); split assignment additionally needs `template_id`. No validated full-corpus routing annotation set or routing results are shipped. The paths below are user-supplied annotations and result locations, not additional SEC corpora.

```bash
# Assign splits only when annotations are added or changed; this rewrites the CSV.
python Code/scripts/make_splits.py --questions /path/to/annotations.csv --eval-fraction 0.25 --seed 42

# Fit both classifiers and evaluate the explicit eval split.
PYTHONPATH=Code/src python Code/scripts/train_routers.py --questions /path/to/annotations.csv --out-dir /path/to/results --seed 42

# Test the router implementations without external data or model downloads.
PYTHONPATH=Code/src python -m pytest Code/tests/test_routing_baselines.py -q
```

`make_splits.py` sorts and shuffles templates deterministically within each scope, then assigns whole template groups to eval until the target fraction is reached. Group sizes can make the realized fraction differ from 25%. Keep a template's scope consistent and verify that no template appears in both splits, so a classifier is never evaluated on a rewording of a training question. Fit features and classifiers on the training split only; reserve eval questions for reporting.

`train_routers.py` writes `metrics.json`, `confusion_logreg.csv` and `confusion_naive_bayes.csv` to the requested output directory. The report records the annotation path, seed, train/eval counts and model configurations, plus accuracy, macro F1 and per-class precision, recall, F1 and support. Confusion-matrix rows are true labels and columns are predictions in the order `simple`, `multi_source`, `corpus_wide`.

Review wording for shortcuts such as associating every corpus-wide question with "which companies" or "in the collection". Keep scope and answerability separate: a question can have an identifiable scope while lacking supporting evidence. An absence claim must be checked against the full canonical corpus before it is used as an abstention label.
