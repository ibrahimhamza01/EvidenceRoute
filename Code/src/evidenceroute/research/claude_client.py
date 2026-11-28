"""Claude integration for SEC research workflows."""

import os
import time

from evidenceroute.research.llm_client import (
    LLMResult,
    LLMClientError,
)


class ClaudeClient:
    """Claude implementation of the existing LLM interface."""

    def __init__(
        self,
        api_key=None,
        model_name=None,
        base_url=None,
        timeout_seconds=60.0,
        max_tokens=2048,
        max_retries=1,
    ):
        from anthropic import Anthropic

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        self.model_name = (
            model_name
            or os.getenv("ANTHROPIC_MODEL")
            or "claude-sonnet-4-6"
        )

        self.base_url = (
            base_url or os.getenv("ANTHROPIC_BASE_URL")
        )

        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self.max_tokens = max_tokens

        options = {
            "api_key": self.api_key,
            "timeout": timeout_seconds,
            "max_retries": max_retries,
        }

        if self.base_url:
            options["base_url"] = self.base_url

        self.client = Anthropic(**options)

    def generate(self, prompt: str) -> LLMResult:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        start = time.perf_counter()

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            text = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            return LLMResult(
                text=text,
                model_name=response.model,
                latency_seconds=time.perf_counter() - start,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        except Exception as exc:
            raise LLMClientError(
                f"Claude generation failed: {type(exc).__name__}"
            ) from exc
