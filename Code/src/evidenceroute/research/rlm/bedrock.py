"""Amazon Bedrock Converse support for the Claude root client."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from contextlib import closing
from time import monotonic, sleep

import boto3
from botocore.config import Config
from botocore import exceptions as aws_errors
from pydantic import Field

from evidenceroute.research.rlm.claude import (
    _DELEGATE,
    _FINISH,
    _retry_after,
    ClaudeConfig,
    ClaudeModelError,
    ClaudeRootClient,
    ClaudeTransport,
    ProviderTransportError,
)
from evidenceroute.research.rlm.controller import ModelAttempt
from evidenceroute.research.rlm.tools import TOOL_USE_ID_PATTERN


class BedrockConfig(ClaudeConfig):
    """An account-specific Converse model or inference profile identifier."""

    model: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,2047}$")
    region: str | None = None


def _client_error(error: aws_errors.ClientError) -> ProviderTransportError:
    code = error.response.get("Error", {}).get("Code")
    if code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"}:
        return ProviderTransportError("aws_expired_token")
    if code in {
        "UnrecognizedClientException", "InvalidClientTokenId", "InvalidSignatureException",
        "SignatureDoesNotMatch", "UnauthorizedException", "AuthFailure", "MissingAuthenticationToken",
        "MissingAuthenticationTokenException", "InvalidAccessKeyId",
    }:
        return ProviderTransportError("aws_authentication")
    if code in {"AccessDeniedException", "AccessDenied", "UnauthorizedOperation"}:
        return ProviderTransportError("aws_access_denied")
    if code in {"ValidationException", "ResourceNotFoundException"}:
        return ProviderTransportError("aws_configuration")
    if code in {"ModelTimeoutException", "RequestTimeout", "RequestTimeoutException"}:
        return ProviderTransportError("aws_timeout")
    headers = error.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    retry_after = _retry_after(headers.get("retry-after"))
    if code in {"ThrottlingException", "TooManyRequestsException"}:
        return ProviderTransportError("aws_throttled", retryable=True, retry_after=retry_after)
    if code in {"ServiceUnavailableException", "InternalServerException", "ModelNotReadyException"}:
        return ProviderTransportError("aws_unavailable", retryable=True, retry_after=retry_after)
    return ProviderTransportError("aws_request_failed")


class BedrockConverseTransport:
    """Send Converse requests using the supplied Boto3 session's credential chain."""

    def __init__(self, session: boto3.Session, *, region: str):
        self._session = session
        self._region = region

    def send(self, payload: dict, *, timeout_seconds: float) -> dict:
        # Create a client per attempt so each request uses its current time budget.
        # The session and its credential provider remain shared across attempts.
        config = Config(
            connect_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            retries={"mode": "standard", "total_max_attempts": 1},
        )
        try:
            with closing(self._session.client(
                "bedrock-runtime", region_name=self._region, config=config,
            )) as client:
                return client.converse(**payload)
        except aws_errors.ClientError as exc:
            error = _client_error(exc)
            choice = payload.get("toolConfig", {}).get("toolChoice", {}).get("tool", {}).get("name")
            if exc.response.get("Error", {}).get("Code") == "ValidationException":
                if choice == _FINISH:
                    error = ProviderTransportError("aws_required_finish_validation")
                elif choice == _DELEGATE:
                    error = ProviderTransportError("aws_required_delegation_validation")
            usage = exc.response.get("usage")
            if isinstance(usage, dict):
                error.usage = usage
            raise error from None
        except (aws_errors.NoCredentialsError, aws_errors.PartialCredentialsError):
            raise ProviderTransportError("aws_missing_credentials") from None
        except (aws_errors.CredentialRetrievalError, aws_errors.SSOError, aws_errors.TokenRetrievalError):
            raise ProviderTransportError("aws_authentication") from None
        except (aws_errors.NoRegionError, aws_errors.InvalidRegionError, aws_errors.ParamValidationError,
                aws_errors.ProfileNotFound, aws_errors.ConfigParseError):
            raise ProviderTransportError("aws_configuration") from None
        except aws_errors.ConnectTimeoutError:
            raise ProviderTransportError("aws_connect_timeout", retryable=True) from None
        except aws_errors.ReadTimeoutError:
            raise ProviderTransportError("aws_timeout") from None
        except aws_errors.BotoCoreError:
            raise ProviderTransportError("aws_transport_failed") from None


class BedrockRootClient(ClaudeRootClient):
    """Reuse the Claude loop with Converse wire formats and native history."""

    _tool_use_id_pattern = TOOL_USE_ID_PATTERN
    _known_stop_reasons = frozenset({
        "end_turn", "tool_use", "max_tokens", "stop_sequence", "guardrail_intervened", "content_filtered",
        "malformed_model_output", "malformed_tool_use", "model_context_window_exceeded",
    })

    def __init__(
        self, transport: ClaudeTransport, *, config: BedrockConfig,
        clock: Callable[[], float] = monotonic, wait: Callable[[float], None] = sleep,
    ):
        super().__init__(transport, config=config, clock=clock, wait=wait)

    @classmethod
    def from_env(cls, *, disable_retries: bool = False, clock: Callable[[], float] = monotonic) -> BedrockRootClient:
        if not os.getenv("BEDROCK_MODEL_ID"):
            raise ValueError("Set BEDROCK_MODEL_ID to an inference profile or model authorized for your account.")
        config = BedrockConfig.from_env(model_variable="BEDROCK_MODEL_ID")
        try:
            session = boto3.Session()
            region = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or session.region_name
        except aws_errors.BotoCoreError:
            raise ValueError("Could not load AWS profile configuration.") from None
        if not region:
            raise ValueError("Set BEDROCK_REGION, AWS_REGION, AWS_DEFAULT_REGION, or a region in your AWS profile.")
        updates = {"region": region}
        if disable_retries:
            updates["max_retries"] = 0
        return cls(BedrockConverseTransport(session, region=region), config=config.model_copy(update=updates), clock=clock)

    def _payload(self, tools: list[dict]) -> dict:
        policy = self._finalization_policy()
        choice = {"tool": {"name": _FINISH}} if policy and policy.finalization_tool_choice == "required" else {"auto": {}}
        if not self._finish_allowed:
            choice = {"tool": {"name": _DELEGATE}}
        return self._converse_payload(
            tools, choice, self._system_prompt(), self._messages,
            policy.finalization_max_output_tokens if policy else self.config.max_output_tokens,
        )

    def _converse_payload(
        self, tools: list[dict], choice: dict, system: str, messages: list[dict], output_cap: int,
    ) -> dict:
        inference = {"maxTokens": output_cap}
        if self.config.temperature is not None:
            inference["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            inference["topP"] = self.config.top_p
        return {
            "modelId": self.config.model,
            "system": [{"text": system}],
            "messages": messages,
            "inferenceConfig": inference,
            "toolConfig": {"toolChoice": choice, "tools": [
                {"toolSpec": {
                    "name": tool["name"], "description": tool["description"],
                    "inputSchema": {"json": tool["input_schema"]},
                }}
                for tool in tools
            ]},
        }

    @staticmethod
    def _question_message(question: str) -> dict:
        return {"role": "user", "content": [{"text": question}]}

    @staticmethod
    def _result_block(use_id: str, observation: str, body: dict) -> dict:
        return {"toolResult": {"toolUseId": use_id, "content": [{"json": body}]}}

    @staticmethod
    def _assistant_message(response: dict) -> dict | None:
        output = response.get("output") if isinstance(response, dict) else None
        message = output.get("message") if isinstance(output, dict) else None
        if isinstance(message, dict) and message.get("role") == "assistant":
            if isinstance(message.get("content"), list) and message["content"]:
                return message
        return None

    @staticmethod
    def _stop_reason(response: dict):
        return response.get("stopReason")

    @staticmethod
    def _tool_uses(blocks: list) -> list[dict]:
        uses = []
        for block in blocks:
            if not isinstance(block, dict) or len(block) != 1:
                raise ClaudeModelError("invalid_content")
            if "toolUse" in block:
                use = block["toolUse"]
                if not isinstance(use, dict):
                    raise ClaudeModelError("invalid_tool_use")
                uses.append({"id": use.get("toolUseId"), "name": use.get("name"), "input": use.get("input")})
            elif not (
                isinstance(block.get("text"), str) or isinstance(block.get("reasoningContent"), dict)
            ):
                raise ClaudeModelError("invalid_content")
        return uses

    def _structured_tool_arguments(
        self, response: dict, reason: str | None, name: str, *, missing: str, invalid: str,
    ) -> dict:
        """Decode one forced tool reply for a stateless Converse request."""
        if reason == "max_tokens":
            raise ClaudeModelError("truncated_response")
        assistant = self._assistant_message(response)
        if reason != "tool_use" or assistant is None:
            raise ClaudeModelError(missing)
        uses = self._tool_uses(assistant["content"])
        if len(uses) != 1:
            raise ClaudeModelError(invalid)
        use, = uses
        if (use["name"] != name or not isinstance(use["input"], dict) or not isinstance(use["id"], str)
                or not re.fullmatch(self._tool_use_id_pattern, use["id"])):
            raise ClaudeModelError(invalid)
        return use["input"]

    def _usage_attempt(self, latency: float, response: dict) -> ModelAttempt:
        usage = response.get("usage") if isinstance(response, dict) else None
        metrics = response.get("metrics") if isinstance(response, dict) else None
        latency_ms = metrics.get("latencyMs") if isinstance(metrics, dict) else None
        counts = {}
        if isinstance(usage, dict):
            for key, source in (
                ("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"),
                ("cache_read_input_tokens", "cacheReadInputTokens"),
                ("cache_creation_input_tokens", "cacheWriteInputTokens"),
            ):
                count = usage.get(source)
                if type(count) is int and count >= 0:
                    counts[key] = count
        return ModelAttempt(
            self.config.model, latency, **counts,
            provider_latency_seconds=latency_ms / 1_000 if type(latency_ms) is int and latency_ms >= 0 else None,
        )
