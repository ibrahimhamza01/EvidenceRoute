"""Offline tests for the baseline routers: small made-up questions, no files needed."""

import pytest

from evidenceroute.routing.baselines import BaselineRouter
from evidenceroute.schemas import RouterResult

QUESTIONS = [
    "How many employees did Company X report in 2024?",
    "What was Company X's revenue in its 2023 10-K?",
    "Compare Company X and Company Y revenue in 2024.",
    "How did Company X's risks change from 2023 to 2024?",
    "Which companies in the collection rely on TSMC?",
    "Which companies mention climate change risk?",
]
SCOPES = ["simple", "simple", "multi_source", "multi_source", "corpus_wide", "corpus_wide"]


@pytest.mark.parametrize("kind", ["logreg", "naive_bayes"])
def test_router_returns_valid_router_result(kind):
    router = BaselineRouter(kind).fit(QUESTIONS, SCOPES)
    result = router.predict("Which companies disclose cybersecurity risks?")
    assert isinstance(result, RouterResult)
    assert result.scope in {"simple", "multi_source", "corpus_wide"}
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("kind", ["logreg", "naive_bayes"])
def test_router_is_deterministic(kind):
    a = BaselineRouter(kind).fit(QUESTIONS, SCOPES).predict_many(QUESTIONS)
    b = BaselineRouter(kind).fit(QUESTIONS, SCOPES).predict_many(QUESTIONS)
    assert a == b


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        BaselineRouter("random_forest")
