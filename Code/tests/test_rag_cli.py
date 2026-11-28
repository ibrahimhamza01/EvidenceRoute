"""Tests for the executable Standard RAG CLI."""

import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidenceroute.research.config import StandardRAGConfig
from evidenceroute.research.llm_client import LLMResult


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_standard_rag.py"
)

spec = importlib.util.spec_from_file_location("run_standard_rag_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_mock_llm_cites_real_chunk_id():
    prompt = (
        "Cite evidence using [chunk:CHUNK_ID].\n"
        "Evidence 1\n"
        "Chunk ID: real-sec-chunk-123\n"
        "Passage:\n"
        "The company reports supply chain risks."
    )

    result = cli.MockCitingLLM().generate(prompt)

    assert isinstance(result, LLMResult)
    assert "[chunk:real-sec-chunk-123]" in result.text
    assert "[chunk:CHUNK_ID]" not in result.text


def test_mock_llm_without_evidence():
    result = cli.MockCitingLLM().generate(
        "No SEC evidence available."
    )

    assert "Insufficient evidence" in result.text


def test_mock_provider_configuration():
    config = StandardRAGConfig(llm_provider="mock")

    assert config.llm_provider == "mock"


def test_invalid_provider_configuration():
    with pytest.raises(ValidationError):
        StandardRAGConfig(llm_provider="unknown")


def test_cli_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        cli.main([
            "--question", "What risks are reported?",
            "--top-k", "0",
            "--provider", "mock",
        ])


def test_cli_mock_full_corpus(capsys):
    if not cli.DEFAULT_CORPUS.exists():
        pytest.skip("Local SEC corpus not available")

    exit_code = cli.main([
        "--question",
        "What semiconductor supply chain risks are reported?",
        "--top-k", "3",
        "--provider", "mock",
    ])

    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["response"]["strategy"] == "rag"
    assert output["response"]["citations"]
    assert output["metrics"] is not None


def test_cli_ticker_filter(capsys):
    if not cli.DEFAULT_CORPUS.exists():
        pytest.skip("Local SEC corpus not available")

    exit_code = cli.main([
        "--question", "What supply chain risks are reported?",
        "--ticker", "NVDA",
        "--top-k", "3",
        "--provider", "mock",
    ])

    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["response"]["citations"]
    assert all(
        citation["ticker"] == "NVDA"
        for citation in output["response"]["citations"]
    )
