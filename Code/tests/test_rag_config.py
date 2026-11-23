"""Tests for Standard RAG configuration."""

import pytest
from pydantic import ValidationError

from evidenceroute.research.config import StandardRAGConfig


def test_default_configuration():
    config = StandardRAGConfig()

    assert config.retrieval_method == "bm25"
    assert config.top_k == 5
    assert config.llm_provider == "gemini"
    assert config.corpus_path.name == "chunks.parquet"
    assert config.build_filters() is None


def test_custom_configuration():
    config = StandardRAGConfig(
        corpus_path="custom/chunks.parquet",
        top_k=10,
        tickers=["NVDA"],
        fiscal_years=[2024],
    )

    assert config.top_k == 10
    assert str(config.corpus_path) == "custom/chunks.parquet"

    filters = config.build_filters()

    assert filters is not None
    assert filters.tickers == ("NVDA",) or filters.tickers == ["NVDA"]
    assert 2024 in filters.fiscal_years


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValidationError):
        StandardRAGConfig(top_k=top_k)


def test_unsupported_retrieval_method():
    with pytest.raises(ValidationError):
        StandardRAGConfig(retrieval_method="unknown")


def test_unsupported_llm_provider():
    with pytest.raises(ValidationError):
        StandardRAGConfig(llm_provider="unknown")


def test_configuration_is_immutable():
    config = StandardRAGConfig()

    with pytest.raises(ValidationError):
        config.top_k = 20
