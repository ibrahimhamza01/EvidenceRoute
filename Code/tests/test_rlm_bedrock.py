"""Verify Converse wire formats and AWS failures without reaching AWS."""

import json
import traceback
from collections import deque

import boto3
import pytest
from botocore import UNSIGNED
from botocore import exceptions as aws_errors
from botocore.config import Config
from botocore.stub import ANY, Stubber

from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockConverseTransport, BedrockRootClient
from evidenceroute.research.rlm.claude import ClaudeModelError
from evidenceroute.research.rlm.tools import tool_schemas

MODEL = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"


def tool_use(name="get_chunk", args=None, use_id="test-use"):
    return {"toolUse": {"name": name, "input": {} if args is None else args, "toolUseId": use_id}}


def result(*blocks, reason="tool_use", usage=None):
    return {
        "output": {"message": {"role": "assistant", "content": list(blocks)}},
        "stopReason": reason,
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15} if usage is None else usage,
        "metrics": {"latencyMs": 125},
    }


class StubSession:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.clients = []
        self.requests = []
        self.closed = 0

    def client(self, service, *, region_name, config):
        self.clients.append((service, region_name, config))
        return self

    def converse(self, **payload):
        self.requests.append(payload)
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed += 1


def root(session, **options):
    return BedrockRootClient(
        BedrockConverseTransport(session, region=REGION), config=BedrockConfig(model=MODEL), **options,
    )


def decide(client, observation=None, *, remaining=3.0):
    return client.next_action("NVIDIA FY2024?", tool_schemas(), observation, remaining_seconds=remaining)


def test_converse_round_trip_preserves_assistant_and_ids():
    reasoning = {"reasoningContent": {"reasoningText": {"text": "Choose a chunk.", "signature": "sig"}}}
    first = result(reasoning, {"text": "I will read a chunk."}, tool_use(args={"chunk_id": "c1"}))
    second = result(tool_use("get_filing", {"document_id": "f1"}, "second-use"))
    final = result(tool_use("finish_research", {"answer": "Answer.", "cited_chunk_ids": ["c1"]}, "finish-use"))
    session = StubSession(first, second, final)
    client = root(session)
    assert decide(client) == {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": "c1"}}
    first_request = session.requests[0]
    assert first_request["modelId"] == MODEL
    assert first_request["inferenceConfig"] == {"maxTokens": 1024}
    assert first_request["toolConfig"]["toolChoice"] == {"auto": {}}
    advertised = [tool["toolSpec"] for tool in first_request["toolConfig"]["tools"]]
    assert advertised[:-1] == [
        {"name": tool["name"], "description": tool["description"], "inputSchema": {"json": tool["input_schema"]}}
        for tool in tool_schemas()
    ]
    assert advertised[-1]["name"] == "finish_research"
    assert advertised[-1]["inputSchema"]["json"]["required"] == ["answer"]

    okay = {"ok": True, "result": {"chunk_id": "c1"}}
    assert decide(client, json.dumps(okay))["name"] == "get_filing"
    second_messages = session.requests[1]["messages"]
    assert second_messages == [
        {"role": "user", "content": [{"text": "NVIDIA FY2024?"}]},
        first["output"]["message"],
        {"role": "user", "content": [{"toolResult": {"toolUseId": "test-use", "content": [{"json": okay}]}}]},
    ]
    failure = {"ok": False, "error": {"code": "unknown_id"}}
    assert decide(client, json.dumps(failure)) == {
        "type": "finish", "answer": "Answer.", "cited_chunk_ids": ["c1"],
    }
    assert client.last_observation_delivered
    assert session.requests[2]["messages"] == second_messages + [
        second["output"]["message"],
        {"role": "user", "content": [{"toolResult": {"toolUseId": "second-use", "content": [{"json": failure}]}}]},
    ]
    assert session.closed == 3


def test_real_boto3_client_validates_converse_shapes_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_PROFILE", raising=False)
    sdk_client = boto3.Session(region_name=REGION).client(
        "bedrock-runtime", config=Config(signature_version=UNSIGNED),
    )
    session = StubSession()
    monkeypatch.setattr(session, "client", lambda *a, **kw: sdk_client)
    expected = {
        "modelId": MODEL, "system": ANY, "messages": ANY,
        "inferenceConfig": {"maxTokens": 1024}, "toolConfig": ANY,
    }
    client = root(session)
    lookup = result(tool_use(args={"chunk_id": "c1"}))
    with Stubber(sdk_client) as stub:
        stub.add_response("converse", lookup, expected)
        stub.add_response("converse", result(tool_use("finish_research", {"answer": "Fixture."}, "finish")),
                          {**expected, "messages": [
                              {"role": "user", "content": [{"text": "NVIDIA FY2024?"}]},
                              lookup["output"]["message"],
                              {"role": "user", "content": [{"toolResult": {
                                  "toolUseId": "test-use", "content": [{"json": {"ok": True}}],
                              }}]},
                          ]})
        assert decide(client)["arguments"] == {"chunk_id": "c1"}
        assert decide(client, '{"ok":true}') == {"type": "finish", "answer": "Fixture."}
        stub.assert_no_pending_responses()


@pytest.mark.parametrize(("response", "error"), [
    (result(reason="max_tokens"), "truncated_response"),
    (result({"text": "An answer in prose."}, reason="end_turn"), "missing_finish"),
    (result({"text": "Partial output."}, reason="malformed_tool_use"), "unexpected_stop_reason"),
    (result(tool_use(), tool_use()), "invalid_tool_use"),
    (result(tool_use(), tool_use(use_id="bad id")), "invalid_tool_use"),
    (result(tool_use(args=["not an object"])), "invalid_tool_use"),
    (result({"toolUse": None}), "invalid_tool_use"),
    (result({"text": "There was no tool."}), "missing_tool_use"),
    ({"output": {}}, "invalid_message"),
])
def test_partial_or_invalid_converse_response_never_becomes_a_decision(response, error):
    client = root(StubSession(response))
    with pytest.raises(ClaudeModelError, match=error):
        decide(client)
    assert client.last_attempts[0].error == error


def test_guardrail_response_does_not_confirm_observation_delivery():
    session = StubSession(result(tool_use()), result({"text": "Blocked."}, reason="guardrail_intervened"))
    client = root(session)
    decide(client)
    with pytest.raises(ClaudeModelError, match="unexpected_stop_reason"):
        decide(client, '{"ok":true}')
    assert not client.last_observation_delivered


def client_error(code, *, retry_after=None):
    return aws_errors.ClientError({
        "Error": {"Code": code, "Message": "fixture-secret-must-stay-private"},
        "ResponseMetadata": {"HTTPHeaders": {"retry-after": retry_after} if retry_after else {}},
    }, "Converse")


@pytest.mark.parametrize(("error", "code"), [
    (client_error("ExpiredTokenException"), "aws_expired_token"),
    (client_error("UnrecognizedClientException"), "aws_authentication"),
    (client_error("AccessDeniedException"), "aws_access_denied"),
    (client_error("ValidationException"), "aws_configuration"),
    (aws_errors.NoCredentialsError(), "aws_missing_credentials"),
    (aws_errors.PartialCredentialsError(provider="env", cred_var="AWS_SESSION_TOKEN"), "aws_missing_credentials"),
    (aws_errors.ReadTimeoutError(endpoint_url="https://fixture-secret-must-stay-private"), "aws_timeout"),
])
def test_auth_configuration_and_ambiguous_timeouts_do_not_retry_or_leak(error, code, caplog):
    session = StubSession(error)
    client = root(session, wait=lambda _: pytest.fail("Non-retryable AWS error was retried"))
    with pytest.raises(ClaudeModelError, match=code) as failure:
        decide(client)
    assert "fixture-secret-must-stay-private" not in "".join(traceback.format_exception(failure.value))
    assert "fixture-secret-must-stay-private" not in caplog.text
    assert len(session.requests) == 1
    assert client.last_attempts[0].error == code


def test_retry_shrinks_timeouts_and_records_bedrock_usage():
    now = [0.0]

    def wait(seconds):
        now[0] += seconds

    session = StubSession(client_error("ThrottlingException", retry_after="1.5"), result(
        tool_use(), usage={
            "inputTokens": 30, "outputTokens": 6, "totalTokens": 36,
            "cacheReadInputTokens": 800, "cacheWriteInputTokens": 0,
        },
    ))
    client = root(session, clock=lambda: now[0], wait=wait)
    decide(client, remaining=2.0)
    assert session.requests[0] == session.requests[1]
    for (_, region, config), timeout in zip(session.clients, [2.0, 0.5], strict=True):
        assert region == REGION
        assert config.connect_timeout == config.read_timeout == timeout
        assert config.retries == {"mode": "standard", "total_max_attempts": 1}
    failed, success = client.last_attempts
    assert failed.error == "aws_throttled"
    assert failed.input_tokens is None
    assert (success.model, success.provider_latency_seconds) == (MODEL, 0.125)
    assert (success.input_tokens, success.output_tokens) == (30, 6)
    assert (success.cache_read_input_tokens, success.cache_creation_input_tokens) == (800, 0)


def test_one_attempt_when_retry_after_exceeds_remaining_time():
    session = StubSession(client_error("ThrottlingException", retry_after="10"))
    client = root(session, wait=lambda _: pytest.fail("Delay exceeds budget"))
    with pytest.raises(ClaudeModelError, match="aws_throttled"):
        decide(client, remaining=2.0)
    assert len(session.requests) == 1


def test_from_env_uses_unmodified_boto3_session_credential_chain(monkeypatch):
    monkeypatch.delenv("BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "direct-model")
    monkeypatch.setenv("AWS_PROFILE", "course")
    with pytest.raises(ValueError, match="BEDROCK_MODEL_ID"):
        BedrockRootClient.from_env()
    monkeypatch.setenv("BEDROCK_MODEL_ID", MODEL)
    session = StubSession(result(tool_use()), result(tool_use()))
    session.region_name = "us-west-2"
    monkeypatch.setattr(boto3, "Session", lambda: session)
    decide(BedrockRootClient.from_env())
    assert session.clients[0][0:2] == ("bedrock-runtime", "us-west-2")
    assert session.requests[0]["modelId"] == MODEL
    monkeypatch.setenv("BEDROCK_REGION", REGION)
    decide(BedrockRootClient.from_env())
    assert session.clients[1][1] == REGION
