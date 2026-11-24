"""Exercise the Claude Messages adapter without credentials or provider requests."""

import json
import traceback
from collections import deque

import pytest
import requests

from evidenceroute.research.rlm.claude import (
    AnthropicMessagesTransport,
    ClaudeConfig,
    ClaudeModelError,
    ClaudeRootClient,
    MessagesTransportError,
)
from evidenceroute.research.rlm.tools import tool_schemas


def use(name="list_filings", args=None, use_id="toolu_test"):
    return {"type": "tool_use", "id": use_id, "name": name, "input": {} if args is None else args}


def message(*blocks, reason="tool_use", usage=None):
    return {
        "type": "message", "stop_reason": reason, "content": list(blocks),
        "usage": usage, "model": "claude-sonnet-4-6",
    }


class FakeTransport:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []

    def send(self, payload, *, timeout_seconds):
        self.calls.append((payload, timeout_seconds))
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


def decide(client, observation=None, *, remaining=10.0):
    return client.next_action("NVIDIA FY2024?", tool_schemas(), observation, remaining_seconds=remaining)


def test_tool_use_round_trip_preserves_history_and_evidence_tool_schemas():
    thinking = {"type": "thinking", "thinking": "Use filing metadata.", "signature": "test_signature"}
    intro = {"type": "text", "text": "I will look up the filing."}
    first = use(args={"ticker": "NVDA", "fiscal_year": 2024}, use_id="toolu_1")
    second = use("get_chunk", {"chunk_id": "chunk-1"}, "toolu_2")
    finish = use("finish_research", {"answer": "Answer.", "cited_chunk_ids": ["chunk-1"]}, "toolu_3")
    transport = FakeTransport(message(thinking, intro, first), message(second), message(finish))
    client = ClaudeRootClient(transport)
    action = decide(client)
    assert action == {"type": "tool", "name": "list_filings", "arguments": first["input"]}
    first_payload, timeout = transport.calls[0]
    assert 0 < timeout <= 10
    assert first_payload["model"] == "claude-sonnet-4-6"
    assert first_payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert first_payload["tools"][:-1] == tool_schemas()
    finish_schema = first_payload["tools"][-1]
    assert finish_schema["name"] == "finish_research"
    assert finish_schema["input_schema"]["required"] == ["answer"]
    assert finish_schema["input_schema"]["additionalProperties"] is False
    assert "temperature" not in first_payload and "top_p" not in first_payload
    assert "cache_control" not in first_payload

    ok = '{"ok":true,"result":{"document_id":"fixture"}}'
    assert decide(client, ok)["name"] == "get_chunk"
    assert client.last_observation_delivered
    second_payload, _ = transport.calls[1]
    assert second_payload["messages"] == [
        {"role": "user", "content": "NVIDIA FY2024?"},
        {"role": "assistant", "content": [thinking, intro, first]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": ok, "is_error": False}
        ]},
    ]

    failure = '{"ok":false,"error":{"code":"unknown_id"}}'
    assert decide(client, failure) == {"type": "finish", **finish["input"]}
    assert transport.calls[2][0]["messages"] == second_payload["messages"] + [
        {"role": "assistant", "content": [second]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": failure, "is_error": True}
        ]},
    ]


@pytest.mark.parametrize(("response", "error"), [
    (None, "invalid_message"),
    (message({"type": "text", "text": "Plain answer."}, reason="end_turn"), "missing_finish"),
    (message(use(), reason="max_tokens"), "truncated_response"),
    (message(reason="refusal"), "unexpected_stop_reason"),
    (message(use(), use(use_id="other")), "multiple_tool_uses"),
    (message({"type": "text", "text": "No call."}), "missing_tool_use"),
    (message({"type": "text"}, use()), "invalid_content"),
    (message(use(args=["invalid"])), "invalid_tool_use"),
    (message(use(use_id="")), "invalid_tool_use"),
    (message(use("finish_research", {"type": "tool", "answer": "Invalid."})), "invalid_finish_payload"),
])
def test_unexpected_responses_do_not_create_a_decision(response, error):
    provider = FakeTransport(response)
    client = ClaudeRootClient(provider)
    with pytest.raises(ClaudeModelError, match=error):
        decide(client)
    assert len(provider.calls) == 1
    assert client.last_attempts[0].error == error


def test_new_question_resets_history_and_duplicate_ids_are_rejected_within_a_run():
    provider = FakeTransport(message(use()), message(use()), message(use()))
    client = ClaudeRootClient(provider)
    decide(client)
    with pytest.raises(ClaudeModelError, match="invalid_tool_use"):
        decide(client, '{"ok":true}')
    decide(client)
    assert provider.calls[2][0]["messages"] == [{"role": "user", "content": "NVIDIA FY2024?"}]


def test_request_retries_preserve_payload_respect_retry_after_and_record_usage():
    clock = FakeClock()
    provider = FakeTransport(
        MessagesTransportError("http", status=429, retry_after=1.5),
        message(use(), usage={
            "input_tokens": 30, "output_tokens": 6,
            "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0,
        }),
    )
    client = ClaudeRootClient(provider, clock=clock, wait=clock.wait)
    decide(client, remaining=2.0)
    assert provider.calls[0][0] == provider.calls[1][0]
    assert [timeout for _, timeout in provider.calls] == [2.0, 0.5]
    failed, success = client.last_attempts
    assert failed.error == "http_429" and failed.input_tokens is None
    assert (success.input_tokens, success.output_tokens) == (30, 6)
    assert (success.cache_read_input_tokens, success.cache_creation_input_tokens) == (800, 0)
    assert success.model == "claude-sonnet-4-6"
    assert success.response_model == "claude-sonnet-4-6"


@pytest.mark.parametrize("error", [
    MessagesTransportError("http", status=401),
    MessagesTransportError("http", status=403),
    MessagesTransportError("http", status=400),
    MessagesTransportError("http", status=429),
    MessagesTransportError("timeout"),
    MessagesTransportError("http", status=529, retry_after=4.0),
])
def test_nonretryable_failures_and_retry_delays_outside_budget_stop(error):
    clock = FakeClock()
    provider = FakeTransport(error)
    client = ClaudeRootClient(provider, clock=clock, wait=clock.wait)
    with pytest.raises(ClaudeModelError):
        decide(client, remaining=2.0)
    assert len(provider.calls) == 1
    assert clock.now == 0.0


def test_retry_limit_is_bounded():
    clock = FakeClock()
    provider = FakeTransport(*(MessagesTransportError("http", status=529) for _ in range(3)))
    client = ClaudeRootClient(provider, config=ClaudeConfig(max_retries=2), clock=clock, wait=clock.wait)
    with pytest.raises(ClaudeModelError, match="http_529"):
        decide(client)
    assert len(client.last_attempts) == len(provider.calls) == 3
    assert clock.now == 0.75


def test_timeout_clamps_to_config_and_late_response_retains_usage():
    clock = FakeClock()

    class SlowTransport:
        def send(self, payload, *, timeout_seconds):
            assert timeout_seconds == 0.5
            clock.now += 1.5
            return message(use(), usage={"output_tokens": 12})

    client = ClaudeRootClient(SlowTransport(), config=ClaudeConfig(timeout_seconds=0.5), clock=clock)
    with pytest.raises(ClaudeModelError, match="deadline"):
        decide(client, remaining=1.0)
    attempt, = client.last_attempts
    assert (attempt.latency_seconds, attempt.output_tokens, attempt.error) == (1.5, 12, "deadline")


def test_configuration_can_omit_or_opt_in_to_sampling(monkeypatch):
    names = {
        "CLAUDE_MODEL": "provider-sonnet",
        "CLAUDE_MAX_OUTPUT_TOKENS": "512",
        "CLAUDE_TIMEOUT_SECONDS": "1.5",
        "CLAUDE_TEMPERATURE": "0.2",
        "CLAUDE_MAX_RETRIES": "0",
    }
    monkeypatch.delenv("CLAUDE_TOP_P", raising=False)
    for key, value in names.items():
        monkeypatch.setenv(key, value)
    config = ClaudeConfig.from_env()
    provider = FakeTransport(message(use()))
    decide(ClaudeRootClient(provider, config=config))
    payload, timeout = provider.calls[0]
    assert (payload["model"], payload["max_tokens"], payload["temperature"], timeout) == (
        "provider-sonnet", 512, 0.2, 1.5,
    )
    assert "top_p" not in payload
    monkeypatch.setenv("CLAUDE_TOP_P", "0.8")
    with pytest.raises(ValueError, match="sampling settings"):
        ClaudeConfig.from_env()
    monkeypatch.delenv("CLAUDE_TEMPERATURE")
    provider = FakeTransport(message(use()))
    decide(ClaudeRootClient(provider, config=ClaudeConfig.from_env()))
    assert provider.calls[0][0]["top_p"] == 0.8


def test_native_transport_sets_messages_endpoint_and_never_prints_credentials(monkeypatch, caplog):
    secret = "fixture-secret"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/prefix/v1/")
    for name in ("CLAUDE_MODEL", "CLAUDE_MAX_OUTPUT_TOKENS", "CLAUDE_TIMEOUT_SECONDS",
                 "CLAUDE_TEMPERATURE", "CLAUDE_TOP_P", "CLAUDE_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)

    def post(url, **kwargs):
        assert url == "https://gateway.example/prefix/v1/messages"
        assert kwargs["headers"] == {"x-api-key": secret, "anthropic-version": "2023-06-01"}
        assert not kwargs["allow_redirects"]
        raise requests.ReadTimeout(f"provider exception with {secret}")

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ClaudeModelError, match="timeout") as failure:
        decide(ClaudeRootClient.from_env())
    assert secret not in "".join(traceback.format_exception(failure.value))
    assert secret not in caplog.text

    with pytest.raises(ValueError) as bad_url:
        AnthropicMessagesTransport(api_key=secret, base_url=f"https://{secret}@host.example")
    assert secret not in str(bad_url.value)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        ClaudeRootClient.from_env()


def test_native_transport_sanitizes_http_bodies_and_parses_success(monkeypatch):
    expected = message(use())
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(expected).encode()
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    transport = AnthropicMessagesTransport(api_key="fixture-secret")
    assert transport.send({}, timeout_seconds=1) == expected
    response.status_code = 401
    response._content = b'{"error":{"message":"fixture-secret"}}'
    with pytest.raises(MessagesTransportError, match="^http_401$"):
        transport.send({}, timeout_seconds=1)
    response.status_code = 200
    response._content = b"non-JSON fixture-secret"
    with pytest.raises(MessagesTransportError, match="^invalid_json$") as error:
        transport.send({}, timeout_seconds=1)
    assert "fixture-secret" not in "".join(traceback.format_exception(error.value))
