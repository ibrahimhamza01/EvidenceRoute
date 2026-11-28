"""Tests for RAG-specific citation quality."""

from types import SimpleNamespace

from evidenceroute.evaluation.rag_quality import (
    citation_id_precision,
)


def result(chunk_id):
    return SimpleNamespace(
        chunk=SimpleNamespace(chunk_id=chunk_id)
    )


def test_valid_citation():
    assert citation_id_precision(
        "Supported claim [chunk:a1].",
        [result("a1")],
    ) == 1.0


def test_invalid_citation():
    assert citation_id_precision(
        "Unsupported claim [chunk:fake].",
        [result("a1")],
    ) == 0.0


def test_mixed_citations():
    assert citation_id_precision(
        "Claim [chunk:a1] and [chunk:fake].",
        [result("a1")],
    ) == 0.5


def test_no_citations():
    assert citation_id_precision(
        "No citations.", [result("a1")]
    ) == 0.0
