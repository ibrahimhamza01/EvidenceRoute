"""Tests for the Standard RAG benchmark runner."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

from evidenceroute.schemas import Chunk, RetrievalResult


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_standard_rag.py"
spec = importlib.util.spec_from_file_location("rag_benchmark_runner", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def make_result(chunk_id):
    text = "Supply chain risks."

    return RetrievalResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="test-doc",
            company_name="NVIDIA",
            ticker="NVDA",
            fiscal_year=2024,
            section_id="item-1a",
            section_name="Risk Factors",
            chunk_index=0,
            text=text,
            char_start=0,
            char_end=len(text),
            source_url="https://www.sec.gov/",
        ),
        score=1.0,
    )


def test_benchmark_metrics():
    retriever = Mock()
    retriever.search.return_value = [
        make_result("wrong"),
        make_result("correct"),
    ]

    report = benchmark.evaluate(
        retriever,
        [{
            "question_id": "q1",
            "question": "What risks are reported?",
            "relevant_ids": {"correct"},
        }],
        k=2,
    )

    assert report["questions_evaluated"] == 1
    assert report["mean_recall_at_k"] == 1.0
    assert report["mean_mrr_at_k"] == 0.5
    assert report["mean_latency_ms"] >= 0


def test_load_questions(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,relevant_chunk_ids\n"
        "q1,What risks?,chunk-a;chunk-b\n"
    )

    questions = benchmark.load_questions(path)

    assert len(questions) == 1
    assert questions[0]["relevant_ids"] == {"chunk-a", "chunk-b"}


def test_reject_unlabeled_questions(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,relevant_chunk_ids\n"
        "q1,What risks?,\n"
    )

    with pytest.raises(ValueError):
        benchmark.load_questions(path)


def test_reject_duplicate_questions(tmp_path):
    path = tmp_path / "questions.csv"
    path.write_text(
        "question_id,question,relevant_chunk_ids\n"
        "q1,Question one?,chunk-a\n"
        "q1,Question two?,chunk-b\n"
    )

    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.load_questions(path)
