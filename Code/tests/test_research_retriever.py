"""Tests for research retrieval construction."""

from unittest.mock import Mock, patch

import pytest

from evidenceroute.research.config import StandardRAGConfig
from evidenceroute.research.retriever import (
    build_research_retriever,
)


def test_bm25_builder():
    corpus = Mock()

    with patch(
        "evidenceroute.research.retriever.BM25Retriever"
    ) as builder:
        result = build_research_retriever(corpus, "bm25")

    builder.assert_called_once_with(corpus)
    assert result is builder.return_value


def test_hybrid_builder(tmp_path):
    corpus = Mock()

    with patch(
        "evidenceroute.retrieval.factory.build_hybrid_retriever"
    ) as builder:
        result = build_research_retriever(
            corpus,
            "hybrid",
            embedder_name="hashing",
            index_dir=tmp_path,
        )

    builder.assert_called_once()
    assert result is builder.return_value


def test_hybrid_reranking_builder(tmp_path):
    corpus = Mock()

    with patch(
        "evidenceroute.retrieval.factory.build_hybrid_retriever"
    ) as hybrid, patch(
        "evidenceroute.retrieval.rerank.build_reranking_retriever"
    ) as reranker:

        result = build_research_retriever(
            corpus,
            "hybrid-rerank",
            embedder_name="hashing",
            index_dir=tmp_path,
        )

    hybrid.assert_called_once()
    reranker.assert_called_once()

    args = reranker.call_args.args

    assert args[0] is hybrid.return_value
    assert args[1].enabled is True
    assert result is reranker.return_value


def test_invalid_retrieval_method():
    with pytest.raises(ValueError):
        build_research_retriever(Mock(), "invalid")


@pytest.mark.parametrize(
    "method",
    ["bm25", "hybrid", "hybrid-rerank"],
)
def test_config_supports_retrieval_methods(method):
    config = StandardRAGConfig(retrieval_method=method)
    assert config.retrieval_method == method
