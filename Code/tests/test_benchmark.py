"""Offline tests for the evaluation benchmark: no processed data, Qdrant or LLM calls needed."""

from pathlib import Path

import pytest

from evidenceroute.evaluation.benchmark import gold_groups, read_csv, validate
from evidenceroute.evaluation.gold import containing_chunks, locate_quote
from evidenceroute.evaluation.metrics import (
    full_coverage_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)

ROOT = Path(__file__).resolve().parents[2]

# --- Retrieval metrics -------------------------------------------------------

# Two evidence groups; the second quote sits in the overlap of chunks b1 and b2.
GOLD = {"g1": {"a1"}, "g2": {"b1", "b2"}}


def test_recall_counts_groups_not_chunks():
    assert recall_at_k(GOLD, ["x", "b1", "b2"], k=3) == 0.5  # b1 and b2 are the same group
    assert recall_at_k(GOLD, ["a1", "x", "b2"], k=3) == 1.0
    assert recall_at_k(GOLD, ["x", "y", "a1"], k=2) == 0.0  # a1 is outside the top 2


def test_precision_and_reciprocal_rank():
    ranked = ["x", "a1", "y", "b2"]
    assert precision_at_k(GOLD, ranked, k=4) == 0.5
    assert reciprocal_rank(GOLD, ranked, k=4) == 0.5  # first relevant chunk at rank 2
    assert reciprocal_rank(GOLD, ["x", "y"], k=2) == 0.0


def test_full_coverage_requires_every_group():
    assert full_coverage_at_k(GOLD, ["a1", "b1"], k=2)
    assert not full_coverage_at_k(GOLD, ["a1", "x"], k=2)


def test_recall_without_gold_raises():
    with pytest.raises(ValueError):
        recall_at_k({}, ["a1"], k=1)


# --- Quote matching ------------------------------------------------------------

def test_locate_quote_ignores_table_separators_and_curly_quotes():
    section = "Intro.\nResearch and development expenses | $ | 8,675 | | $ | 7,339\nNVIDIA’s results."
    start, end = locate_quote(section, "Research and development expenses $ 8,675 $ 7,339")
    assert section[start:end].startswith("Research") and section[start:end].endswith("7,339")
    assert locate_quote(section, "NVIDIA's results.") is not None
    assert locate_quote(section, "not in the text") is None


def test_containing_chunks_requires_full_containment():
    chunks = [
        {"chunk_id": "c0", "document_id": "d", "section_id": "s", "char_start": 0, "char_end": 100},
        {"chunk_id": "c1", "document_id": "d", "section_id": "s", "char_start": 80, "char_end": 200},
    ]
    assert containing_chunks(chunks, "d", "s", 85, 95) == ["c0", "c1"]  # inside the overlap
    assert containing_chunks(chunks, "d", "s", 90, 150) == ["c1"]
    assert containing_chunks(chunks, "d", "other", 85, 95) == []


# --- Validation ------------------------------------------------------------------

def make_question(**overrides):
    row = {
        "question_id": "q1", "scope": "simple", "answerable": "true", "corpus": "pilot",
        "eval_split": "dev", "router_split": "", "template_id": "t1", "document_ids": "d1",
        "expected_answer": "42", "gold_evidence_status": "complete", "quote_verified": "true",
    }
    row.update(overrides)
    return row


def make_evidence(**overrides):
    row = {
        "question_id": "q1", "evidence_group": "g1", "document_id": "d1", "section_id": "item_7",
        "chunk_ids": "c1", "source_url": "https://www.sec.gov/x.htm", "relevance": "supporting",
        "mapping_status": "mapped",
    }
    row.update(overrides)
    return row


MANIFEST = [{"document_id": "d1", "source_url": "https://www.sec.gov/x.htm"}]


def test_valid_toy_benchmark_passes():
    assert validate([make_question()], [make_evidence()], MANIFEST, chunk_ids={"c1"}) == []


def test_validation_catches_template_leakage():
    questions = [make_question(), make_question(question_id="q2", eval_split="heldout")]
    evidence = [make_evidence(), make_evidence(question_id="q2")]
    errors = validate(questions, evidence, MANIFEST)
    assert any("appears in both" in e for e in errors)


def test_validation_catches_missing_chunk_and_bad_labels():
    errors = validate([make_question(scope="hard")], [make_evidence(chunk_ids="zzz")], MANIFEST, chunk_ids={"c1"})
    assert any("invalid scope" in e for e in errors)
    assert any("not found in processed data" in e for e in errors)


def test_multi_source_needs_two_evidence_groups():
    errors = validate([make_question(scope="multi_source")], [make_evidence()], MANIFEST)
    assert any("at least 2 evidence groups" in e for e in errors)


def test_gold_groups_ignore_related_passages():
    evidence = [make_evidence(), make_evidence(evidence_group="g2", chunk_ids="c2", relevance="related")]
    assert gold_groups(evidence, "q1") == {"g1": {"c1"}}


# --- The committed benchmark files ---------------------------------------------------

def test_committed_benchmark_is_valid():
    benchmark = read_csv(ROOT / "data/eval/benchmark.csv")
    gold = read_csv(ROOT / "data/eval/gold_evidence.csv")
    manifest = read_csv(ROOT / "data/filings_manifest.csv")
    assert validate(benchmark, gold, manifest) == []
