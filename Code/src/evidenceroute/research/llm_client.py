
"""Configurable LLM client for SEC research."""

import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    """Text and execution metadata returned by an LLM."""

    text: str
    model_name: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    """Interface that real and mock LLM clients implement."""

    def generate(self, prompt: str) -> LLMResult:
        ...


class LLMClientError(Exception):
    """Raised when an LLM request fails."""


class GeminiClient:
    """Gemini implementation of the LLMClient interface."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        from google import genai

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or "gemini-2.5-flash"
        )
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise ValueError(
                "Missing GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(
            api_key=self.api_key,
            http_options={
                "timeout": int(timeout_seconds * 1000)
            },
        )

    def generate(self, prompt: str) -> LLMResult:
        """Generate a response and record execution metadata."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        from google.genai import types

        start = time.perf_counter()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )

            elapsed = time.perf_counter() - start

            usage = response.usage_metadata

            return LLMResult(
                text=response.text or "",
                model_name=self.model_name,
                latency_seconds=elapsed,
                input_tokens=(
                    usage.prompt_token_count
                    if usage else None
                ),
                output_tokens=(
                    usage.candidates_token_count
                    if usage else None
                ),
            )

        except Exception as exc:
            raise LLMClientError(
                f"LLM generation failed: {type(exc).__name__}"
            ) from exc
