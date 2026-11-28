"""Offline tests for Claude integration."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evidenceroute.research.claude_client import ClaudeClient
from evidenceroute.research.llm_client import LLMClientError


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeClient()


@patch("anthropic.Anthropic")
def test_empty_prompt(mock_sdk):
    client = ClaudeClient(api_key="test-key")

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("   ")

    mock_sdk.return_value.messages.create.assert_not_called()


@patch("anthropic.Anthropic")
def test_claude_generation(mock_sdk):
    mock_sdk.return_value.messages.create.return_value = (
        SimpleNamespace(
            model="claude-sonnet-4-6",
            content=[
                SimpleNamespace(
                    type="text",
                    text="Supported SEC answer [chunk:a1].",
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=25,
            ),
        )
    )

    client = ClaudeClient(
        api_key="test-key",
        model_name="claude-sonnet-4-6",
        max_tokens=512,
    )

    result = client.generate("Analyze SEC evidence.")

    assert result.text == "Supported SEC answer [chunk:a1]."
    assert result.model_name == "claude-sonnet-4-6"
    assert result.input_tokens == 100
    assert result.output_tokens == 25
    assert result.latency_seconds >= 0

    request = mock_sdk.return_value.messages.create.call_args.kwargs

    assert request["max_tokens"] == 512
    assert request["model"] == "claude-sonnet-4-6"


@patch("anthropic.Anthropic")
def test_api_failure(mock_sdk):
    mock_sdk.return_value.messages.create.side_effect = (
        RuntimeError("Simulated API failure")
    )

    client = ClaudeClient(api_key="test-key")

    with pytest.raises(LLMClientError):
        client.generate("Analyze SEC evidence.")


@patch("anthropic.Anthropic")
def test_custom_endpoint(mock_sdk):
    ClaudeClient(
        api_key="test-key",
        base_url="https://example.com",
        timeout_seconds=30,
        max_retries=0,
    )

    options = mock_sdk.call_args.kwargs

    assert options["base_url"] == "https://example.com"
    assert options["timeout"] == 30
    assert options["max_retries"] == 0
