
import pytest

from evidenceroute.research.llm_client import (
    GeminiClient,
    LLMClientError,
    LLMResult,
)


class MockLLMClient:
    """A fake LLM that makes no API calls."""

    def generate(self, prompt: str) -> LLMResult:
        return LLMResult(
            text="The supplied SEC passage describes a risk.",
            model_name="mock-model",
            latency_seconds=0.01,
            input_tokens=25,
            output_tokens=12,
        )


def test_mock_client_returns_answer():
    client = MockLLMClient()
    result = client.generate("Summarize this SEC evidence.")

    assert "describes a risk" in result.text
    assert result.model_name == "mock-model"


def test_mock_client_tracks_metadata():
    client = MockLLMClient()
    result = client.generate("Test prompt")

    assert result.input_tokens == 25
    assert result.output_tokens == 12
    assert result.latency_seconds >= 0


def test_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(api_key=None)


def test_gemini_rejects_empty_prompt():
    client = GeminiClient(api_key="test-placeholder")

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        client.generate("   ")


def test_llm_error_type():
    error = LLMClientError("Test failure")
    assert isinstance(error, Exception)
