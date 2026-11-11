"""Classical baseline routers: TF-IDF features with Logistic Regression or Naive Bayes.

Each router predicts the scope of a research question (simple, multi_source or
corpus_wide) and returns it as a RouterResult, the shared schema in schemas.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from evidenceroute.schemas import RouterResult

ModelKind = Literal["logreg", "naive_bayes"]
SEED = 42


def build_pipeline(kind: ModelKind, seed: int = SEED) -> Pipeline:
    """Return an untrained TF-IDF + classifier pipeline."""
    # Words and two-word phrases ("which companies", "in its") become features.
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))

    if kind == "logreg":
        classifier = LogisticRegression(max_iter=1000, random_state=seed)
    elif kind == "naive_bayes":
        classifier = MultinomialNB(alpha=1.0)  # alpha=1.0 is standard add-one smoothing
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


class BaselineRouter:
    """Wraps a scikit-learn pipeline so predictions come back as RouterResult objects."""

    def __init__(self, kind: ModelKind, seed: int = SEED) -> None:
        self.kind = kind
        self.pipeline = build_pipeline(kind, seed)

    def fit(self, questions: list[str], scopes: list[str]) -> "BaselineRouter":
        self.pipeline.fit(questions, scopes)
        return self

    def predict(self, question: str) -> RouterResult:
        return self.predict_many([question])[0]

    def predict_many(self, questions: list[str]) -> list[RouterResult]:
        # predict_proba gives one probability per class; the highest one is the prediction.
        probabilities = self.pipeline.predict_proba(questions)
        classes = self.pipeline.classes_
        results = []
        for row in probabilities:
            best = row.argmax()
            results.append(RouterResult(scope=classes[best], confidence=float(row[best])))
        return results


def load_split(path: Path, split: str) -> tuple[list[str], list[str]]:
    """Read questions.csv and return (questions, scopes) for one split ("train" or "eval")."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if not rows:
        raise ValueError(f"No questions with split={split!r} in {path}. Run make_splits.py first.")
    return [r["question"] for r in rows], [r["scope"] for r in rows]
