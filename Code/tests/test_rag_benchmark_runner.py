"""Tests for the shared SEC benchmark integration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/benchmark_standard_rag.py"
)

spec = importlib.util.spec_from_file_location(
    "rag_benchmark_runner", SCRIPT
)

benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def result(chunk_id):
    return SimpleNamespace(
        chunk=SimpleNamespace(chunk_id=chunk_id)
    )


def test_overlapping_gold_chunks():
    retriever = Mock()

    retriever.search.return_value = [
        result("a2"),
        result("b1"),
    ]

    questions = [{
        "question_id": "q1",
        "question": "Compare two companies",
        "scope": "multi_source",
        "gold_groups": {
            "g1": {"a1", "a2"},
            "g2": {"b1"},
        },
    }]

    report = benchmark.evaluate(
        retriever, questions, k=2
    )

    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_precision_at_k"] == 1.0
    assert report["mean_mrr_at_k"] == 1.0
    assert report["full_coverage_rate"] == 1.0


def test_partial_evidence_coverage():
    retriever = Mock()

    retriever.search.return_value = [
        result("wrong"),
        result("a1"),
    ]

    questions = [{
        "question_id": "q2",
        "question": "Compare disclosures",
        "scope": "multi_source",
        "gold_groups": {
            "g1": {"a1", "a2"},
            "g2": {"b1"},
        },
    }]

    report = benchmark.evaluate(
        retriever, questions, k=2
    )

    assert report["mean_recall_at_k"] == 0.5
    assert report["mean_mrr_at_k"] == 0.5
    assert report["full_coverage_rate"] == 0.0


def test_load_shared_benchmark(tmp_path):
    questions = tmp_path / "questions.csv"
    gold = tmp_path / "gold.csv"

    questions.write_text(
        "question_id,question,scope,eval_split,"
        "answerable,review_status,quote_verified,"
        "gold_evidence_status\n"
        "q1,Compare companies,multi_source,heldout,"
        "true,approved,true,complete\n"
    )

    gold.write_text(
        "question_id,evidence_group,chunk_ids,"
        "relevance,mapping_status,quote_verified\n"
        "q1,g1,a1;a2,supporting,mapped,true\n"
        "q1,g2,b1,supporting,mapped,true\n"
        "q1,g3,c1,related,mapped,true\n"
    )

    rows = benchmark.load_questions(
        questions, gold, "heldout"
    )

    assert len(rows) == 1

    assert rows[0]["gold_groups"] == {
        "g1": {"a1", "a2"},
        "g2": {"b1"},
    }


def test_invalid_k():
    with pytest.raises(ValueError):
        benchmark.evaluate(Mock(), [], k=0)


def test_empty_benchmark():
    with pytest.raises(ValueError):
        benchmark.evaluate(Mock(), [], k=5)
