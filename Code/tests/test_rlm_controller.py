"""Exercise the root tool loop using a synthetic local filing and a scripted client."""

import csv
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidenceroute.ingestion.processor import chunk_sections, write_parquet
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.claude import ClaudeRootClient, MessagesTransportError, ProviderTransportError
from evidenceroute.research.rlm.controller import (
    RootController,
    RootLimits,
    ScriptedModelClient,
    action_schema,
)
from evidenceroute.research.rlm.coverage import ResearchScope, SectionTarget
from evidenceroute.research.rlm.tools import CorpusToolDispatcher, tool_schemas
from evidenceroute.retrieval import ChunkFilter
from evidenceroute.schemas import Filing, Section

DOCUMENT = "0001045810_0001045810-24-000001"
RISK = "item_1a"
SOURCE = "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000001/fixture.htm"


@pytest.fixture
def corpus(tmp_path, request):
    large = getattr(request, "param", False)
    filing = Filing(
        document_id=DOCUMENT, company_name="NVIDIA fixture", ticker="NVDA", cik="0001045810",
        accession_number="0001045810-24-000001", form_type="10-K", fiscal_year=2024,
        report_period_end="2024-01-28", filing_date="2024-02-21",
        primary_document="fixture.htm", source_url=SOURCE,
    )
    record = filing.model_dump(mode="json", exclude_none=True)
    manifest = tmp_path / "filings_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    sections = [
        Section(document_id=DOCUMENT, section_id="item_1", section_name="Business",
                source_url=SOURCE, text="A synthetic business description."),
        Section(document_id=DOCUMENT, section_id=RISK, section_name="Risk Factors",
                source_url=SOURCE, text="Supplier disruptions can affect shipments. Café. " * (700 if large else 5)),
    ]
    write_parquet(sections, chunk_sections(sections, filing, max_chars=1800 if large else 60, overlap_chars=0),
                  tmp_path)
    return CorpusEnvironment(manifest_path=manifest, processed_dir=tmp_path)


def tool(tool_name, **arguments):
    return {"type": "tool", "name": tool_name, "arguments": arguments}


def finish(*chunk_ids):
    return {"type": "finish", "answer": "Scripted fixture answer.", "cited_chunk_ids": list(chunk_ids)}


def test_tool_and_action_schemas_are_json_serializable_and_closed(corpus):
    schemas = {schema["name"]: schema["input_schema"] for schema in tool_schemas()}
    assert set(schemas) == {
        "list_companies", "list_filings", "get_filing", "list_sections",
        "get_section", "read_chunks", "get_chunk", "search_chunks",
    }
    assert all(schema["additionalProperties"] is False for schema in schemas.values())
    assert schemas["get_chunk"]["required"] == ["chunk_id"]
    for name in ("read_chunks", "search_chunks"):
        limit = schemas[name]["properties"]["limit"]
        assert (limit["default"], limit["maximum"]) == (1, 10)
    assert action_schema()["discriminator"]["propertyName"] == "type"
    json.dumps([schemas, action_schema()])
    dispatched = CorpusToolDispatcher(corpus, max_response_bytes=24_000)
    result = json.loads(dispatched.dispatch("get_filing", {"document_id": DOCUMENT}).content)
    assert result["result"]["filing_date"] == "2024-02-21"


def test_scripted_root_hands_back_observations_and_preserves_provenance(corpus):
    filters = {"tickers": [" nvda "], "fiscal_years": [2024], "section_ids": [RISK]}
    expected = corpus.search_chunks("disruptions", filters=ChunkFilter.model_validate(filters),
                                    limit=1).items[0].chunk
    model = ScriptedModelClient([
        tool("list_filings", ticker="NVDA", fiscal_year=2024),
        tool("search_chunks", query="disruptions", filters=filters, limit=1),
        tool("get_chunk", chunk_id=expected.chunk_id),
        tool("read_chunks", document_id=DOCUMENT, section_id=RISK, limit=1),
        finish(expected.chunk_id),
    ])
    trace = RootController(corpus, model).run("What supplier risks did NVIDIA disclose in FY2024?")
    assert trace.stop_reason == "finished"
    assert trace.answer == "Scripted fixture answer."
    assert trace.cited_chunk_ids == (expected.chunk_id,)
    assert trace.tool_calls == 4
    assert model.observations[0] is None
    assert [json.loads(obs) for obs in model.observations[1:]] == [
        step.observation for step in trace.steps[:-1]
    ]
    ref, = trace.coverage.search_discovered
    assert (ref.document_id, ref.section_id, ref.chunk_id) == (DOCUMENT, RISK, expected.chunk_id)
    assert (ref.ticker, ref.fiscal_year, ref.source_url) == ("NVDA", 2024, SOURCE)
    assert trace.steps[1].validated_arguments["filters"]["tickers"] == ["NVDA"]
    assert trace.steps[2].observation["result"] == expected.model_dump(mode="json")
    assert ref in trace.coverage.visible_chunks
    assert trace.coverage.scope == ResearchScope()
    progress, = trace.coverage.pagination
    assert (progress.status, progress.ranges) == ("partial", ((0, 1),))
    assert all(step.model_seconds >= 0 for step in trace.steps)
    assert all(step.tool_seconds >= 0 for step in trace.steps[:-1])
    json.dumps(asdict(trace), allow_nan=False)


def test_complete_search_and_section_listing_leave_research_scope_unknown(corpus):
    model = ScriptedModelClient([
        tool("list_sections", document_id=DOCUMENT),
        tool("search_chunks", query="disruptions", limit=10),
        finish(),
    ])
    trace = RootController(corpus, model).run("Find supplier disclosures.")
    assert trace.steps[1].observation["result"]["next_offset"] is None
    assert trace.coverage.search_discovered == trace.coverage.visible_chunks
    assert trace.coverage.visible_chunks
    assert trace.coverage.scope.required_filings is None
    assert trace.coverage.scope.required_sections is None
    assert [(p.operation, p.status) for p in trace.coverage.pagination] == [
        ("list_sections", "complete")
    ]


def test_invalid_host_scope_is_rejected_before_the_model_runs(corpus):
    model = ScriptedModelClient([finish()])
    scope = ResearchScope(required_sections=(SectionTarget(DOCUMENT, "missing"),))
    with pytest.raises(KeyError, match="missing"):
        RootController(corpus, model).run("Inspect that section.", scope=scope)
    assert model.observations == []


def test_pagination_requires_all_ranges_and_keeps_access_types_separate(corpus):
    section = corpus.get_section(DOCUMENT, RISK, limit=1).metadata
    count = section.chunk_count
    model = ScriptedModelClient([
        tool("list_sections", document_id=DOCUMENT, offset=1, limit=1),
        tool("read_chunks", document_id=DOCUMENT, section_id=RISK, offset=count - 1, limit=1),
        tool("get_section", document_id=DOCUMENT, section_id=RISK, offset=10),
        tool("list_sections", document_id=DOCUMENT, limit=1),
        tool("get_section", document_id=DOCUMENT, section_id=RISK, limit=10),
        tool("read_chunks", document_id=DOCUMENT, section_id=RISK, limit=count - 1),
        finish(),
    ])
    scope = ResearchScope((DOCUMENT,), (SectionTarget(DOCUMENT, RISK),))
    trace = RootController(corpus, model).run("Inspect all of Item 1A.", scope=scope)
    assert trace.coverage.scope == scope
    pagination = {entry.operation: entry for entry in trace.coverage.pagination}
    assert pagination["read_chunks"].ranges == ((0, count),)
    assert pagination["get_section"].ranges == ((0, section.char_count),)
    assert pagination["list_sections"].ranges == ((0, 2),)
    assert all(entry.status == "complete" for entry in pagination.values())
    assert len(trace.coverage.visible_chunks) == count
    assert trace.coverage.search_discovered == ()


def test_last_page_alone_and_undelivered_page_do_not_complete_pagination(corpus):
    model = ScriptedModelClient([
        tool("read_chunks", document_id=DOCUMENT, section_id=RISK, offset=1, limit=10),
        tool("read_chunks", document_id=DOCUMENT, section_id=RISK, limit=1),
    ])
    trace = RootController(corpus, model, limits=RootLimits(max_iterations=2)).run("Read the risks.")
    first, last = trace.steps
    assert first.observation["result"]["next_offset"] is None
    assert first.observation_sent
    assert not last.observation_sent
    assert last.evidence[0] not in trace.coverage.visible_chunks
    progress, = trace.coverage.pagination
    assert progress.status == "partial"
    assert progress.ranges == ((1, progress.total),)
    assert trace.stop_reason == "iteration_limit"


def test_invalid_actions_and_arguments_are_rejected_and_recoverable(corpus, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Invalid arguments reached a corpus method")

    monkeypatch.setattr(corpus, "read_chunks", forbidden)
    model = ScriptedModelClient([
        {"type": "tool", "name": "get_chunk"},
        tool("__getattribute__", name="_chunks"),
        tool("read_chunks", document_id=DOCUMENT, limit=True),
        tool("get_filing", document_id=DOCUMENT, path="/tmp/unrestricted"),
        tool("search_chunks", query="risk", filters={"fiscal_years": ["2024"]}),
        {"type": "finish", "answer": "   "},
        finish("made-up-chunk"),
        finish(),
    ])
    trace = RootController(corpus, model).run("Inspect the filing.")
    assert trace.stop_reason == "finished"
    assert trace.tool_calls == 4
    assert [step.error["code"] for step in trace.steps[:-1]] == [
        "invalid_action", "unknown_tool", "invalid_arguments", "invalid_arguments",
        "invalid_arguments", "invalid_action", "unseen_citation",
    ]
    assert all(step.validated_arguments is None for step in trace.steps[:-1])
    assert all(step.observation_sent for step in trace.steps[:-1])


def test_oversized_search_is_reported_without_making_hits_model_visible(corpus):
    search = tool("search_chunks", query="disruptions", limit=1)
    full = CorpusToolDispatcher(corpus, max_response_bytes=24_000).dispatch(
        search["name"], search["arguments"],
    )
    assert "Café" in full.content
    budget = len(full.content)
    assert budget < len(full.content.encode("utf-8"))
    model = ScriptedModelClient([search, finish(full.evidence[0].chunk_id), finish()])
    trace = RootController(corpus, model, limits=RootLimits(max_response_bytes=budget)).run("Search.")
    assert trace.steps[0].error["code"] == "response_too_large"
    assert trace.steps[0].error["response_bytes"] == len(full.content.encode("utf-8"))
    assert "suggested_limit" not in trace.steps[0].error
    assert "get_section" in trace.steps[0].error["message"]
    assert trace.steps[1].error["code"] == "unseen_citation"
    assert all(len(obs.encode("utf-8")) <= budget for obs in model.observations[1:])
    assert trace.coverage.visible_chunks == ()
    assert trace.coverage.pagination == ()
    assert trace.coverage.search_discovered == full.evidence


@pytest.mark.parametrize("corpus", [True], indirect=True, ids=["long-risk-section"])
def test_demo_sized_search_rejects_ten_hits_and_recovers_at_same_offset(corpus):
    args = {"query": "disruptions", "filters": {"section_ids": [RISK]}, "offset": 2}
    requested = tool("search_chunks", **args, limit=10)
    full = CorpusToolDispatcher(corpus, max_response_bytes=100_000).dispatch(
        requested["name"], requested["arguments"],
    )
    assert len(full.evidence) == 10
    expected = json.loads(full.content)["result"]["items"][0]["chunk"]
    model = ScriptedModelClient([requested, tool("search_chunks", **args), finish(expected["chunk_id"])])
    trace = RootController(corpus, model, limits=RootLimits(max_response_bytes=6_000)).run("Find a risk.")
    rejected, retried, _ = trace.steps
    assert rejected.error == {
        "code": "response_too_large", "max_response_bytes": 6_000,
        "response_bytes": len(full.content.encode("utf-8")), "suggested_limit": 1,
        "message": "No results delivered. Retry with suggested_limit and the same offset and filters.",
    }
    assert rejected.error["response_bytes"] > 6_000
    assert rejected.observation_sent and rejected.evidence == ()
    assert "result" not in rejected.observation
    assert len(rejected.search_discovered) == 10
    assert retried.validated_arguments["limit"] == 1
    assert retried.validated_arguments["offset"] == rejected.validated_arguments["offset"] == 2
    assert retried.validated_arguments["filters"] == rejected.validated_arguments["filters"]
    assert retried.observation["result"]["items"][0]["chunk"] == expected
    assert retried.observation["result"]["next_offset"] == 3
    assert retried.observation_sent and trace.stop_reason == "finished"
    assert trace.coverage.visible_chunks == (full.evidence[0],)
    assert len(trace.coverage.search_discovered) == 10
    default_read = CorpusToolDispatcher(corpus, max_response_bytes=6_000).dispatch(
        "read_chunks", {"document_id": DOCUMENT, "section_id": RISK},
    )
    assert default_read.arguments["limit"] == 1
    assert len(default_read.evidence) == 1
    assert json.loads(default_read.content)["result"]["next_offset"] == 1


def test_oversized_section_recommends_smaller_character_window_and_respects_minimum_cap(corpus):
    args = {"document_id": DOCUMENT, "section_id": RISK, "offset": 10}
    full = corpus.get_section(DOCUMENT, RISK, offset=10)
    dispatcher = CorpusToolDispatcher(corpus, max_response_bytes=128)
    response = dispatcher.dispatch("get_section", args)
    assert len(response.content.encode("utf-8")) <= 128
    body = json.loads(response.content)
    assert "result" not in body
    assert body["error"]["code"] == "response_too_large"
    assert body["error"]["suggested_limit"] == len(full.text) // 2


def test_finish_is_allowed_at_tool_budget_but_an_extra_tool_is_rejected(corpus):
    limits = RootLimits(max_tool_calls=1)
    lookup = tool("get_filing", document_id=DOCUMENT)
    model = ScriptedModelClient([lookup, finish()])
    assert RootController(corpus, model, limits=limits).run("Find it.").stop_reason == "finished"
    model = ScriptedModelClient([tool("invalid_tool"), lookup])
    trace = RootController(corpus, model, limits=limits).run("Find it.")
    assert trace.stop_reason == "tool_call_limit"
    assert trace.tool_calls == 1
    assert trace.steps[-1].error["code"] == "tool_call_limit"
    assert trace.steps[-1].tool_seconds is None


def test_repeated_invalid_actions_exhaust_iteration_budget(corpus):
    model = ScriptedModelClient([None] * 4)
    trace = RootController(corpus, model, limits=RootLimits(max_iterations=3)).run("Find it.")
    assert trace.stop_reason == "iteration_limit"
    assert len(trace.steps) == len(model.observations) == 3
    assert trace.tool_calls == 0
    assert trace.answer is None


def test_tool_errors_are_observed_and_exhausted_script_is_a_terminal_model_error(corpus, monkeypatch):
    def broken():
        raise RuntimeError("internal path should stay out of the observation")

    monkeypatch.setattr(corpus, "list_companies", lambda **kwargs: broken())
    model = ScriptedModelClient([tool("get_chunk", chunk_id="missing"), tool("list_companies")])
    trace = RootController(corpus, model).run("Inspect the corpus.")
    assert trace.stop_reason == "model_error"
    assert trace.tool_calls == 2
    assert [step.error["code"] for step in trace.steps] == ["unknown_id", "tool_error", "model_error"]
    assert trace.steps[1].error["message"] == "RuntimeError"
    assert trace.steps[1].observation_sent
    assert trace.steps[2].observation is None


@pytest.mark.parametrize("slow_part", ["model", "tool"])
def test_deadline_stops_before_accepting_late_work(corpus, monkeypatch, slow_part):
    now = [0.0]
    lookup = corpus.get_filing

    class TimedModel(ScriptedModelClient):
        def next_action(self, question, tools, observation, *, remaining_seconds):
            assert remaining_seconds == 1.0
            if slow_part == "model":
                now[0] = 1.0
            return super().next_action(question, tools, observation, remaining_seconds=remaining_seconds)

    def timed_lookup(document_id):
        now[0] = 1.0
        return lookup(document_id)

    monkeypatch.setattr(corpus, "get_filing", timed_lookup)
    model = TimedModel([tool("get_filing", document_id=DOCUMENT), finish()])
    trace = RootController(corpus, model, limits=RootLimits(deadline_seconds=1.0),
                           clock=lambda: now[0]).run("Find the filing.")
    assert trace.stop_reason == "deadline"
    assert trace.elapsed_seconds == 1.0
    assert model.observations == [None]
    assert trace.tool_calls == (0 if slow_part == "model" else 1)
    assert not trace.steps[0].observation_sent


def test_claude_adapter_uses_dispatcher_validation_and_controller_finish(corpus):
    chunk = corpus.read_chunks(DOCUMENT, RISK, limit=1).items[0]
    responses = iter([
        ("list_filings", {"fiscal_year": "2024"}),
        ("get_chunk", {"chunk_id": chunk.chunk_id}),
        ("finish_research", {"answer": "Premature answer.", "cited_chunk_ids": ["unseen"]}),
        ("finish_research", {"answer": "Supplier disruptions can affect shipments.",
                             "cited_chunk_ids": [chunk.chunk_id]}),
    ])
    seen = []

    class Transport:
        def send(self, payload, *, timeout_seconds):
            name, arguments = next(responses)
            seen.append(payload["messages"][-1])
            return {
                "type": "message", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": f"toolu_{len(seen)}",
                             "name": name, "input": arguments}],
                "usage": {"input_tokens": 25, "output_tokens": 10},
            }

    trace = RootController(corpus, ClaudeRootClient(Transport())).run("What risk is disclosed?")
    assert trace.stop_reason == "finished"
    assert trace.cited_chunk_ids == (chunk.chunk_id,)
    assert trace.tool_calls == 2
    assert len(trace.model_attempts) == 4
    assert sum(attempt.input_tokens for attempt in trace.model_attempts) == 100
    assert sum(attempt.output_tokens for attempt in trace.model_attempts) == 40
    invalid, lookup, unseen = (turn["content"][0] for turn in seen[1:])
    assert invalid["tool_use_id"] == "toolu_1" and invalid["is_error"]
    assert json.loads(invalid["content"])["error"]["code"] == "invalid_arguments"
    assert lookup["tool_use_id"] == "toolu_2" and not lookup["is_error"]
    assert json.loads(lookup["content"])["result"]["source_url"] == SOURCE
    assert unseen["tool_use_id"] == "toolu_3" and unseen["is_error"]
    assert json.loads(unseen["content"])["error"]["code"] == "unseen_citation"
    assert trace.coverage.visible_chunks[0].chunk_id == chunk.chunk_id


def test_provider_failure_leaves_answer_unfinished_and_chunk_delivery_unconfirmed(corpus):
    chunk = corpus.read_chunks(DOCUMENT, RISK, limit=1).items[0]

    class Transport:
        def send(self, payload, *, timeout_seconds):
            if len(payload["messages"]) > 1:
                raise MessagesTransportError("http", status=401)
            return {
                "type": "message", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_chunk",
                             "input": {"chunk_id": chunk.chunk_id}}],
            }

    trace = RootController(corpus, ClaudeRootClient(Transport())).run("What risk is disclosed?")
    assert trace.stop_reason == "model_error"
    assert trace.answer is None
    assert trace.steps[0].evidence[0].chunk_id == chunk.chunk_id
    assert not trace.steps[0].observation_sent
    assert trace.coverage.visible_chunks == ()
    assert trace.model_attempts[-1].error == "http_401"


@pytest.mark.parametrize("expired", [False, True], ids=["finished", "expired"])
def test_bedrock_converse_integrates_with_existing_dispatch_and_coverage(corpus, expired):
    chunk = corpus.read_chunks(DOCUMENT, RISK, limit=1).items[0]
    script = iter([
        ("get_chunk", {"chunk_id": chunk.chunk_id}, "lookup"),
        ("finish_research", {"answer": "Fixture answer.", "cited_chunk_ids": [chunk.chunk_id]}, "finish"),
    ])

    class Transport:
        def send(self, payload, *, timeout_seconds):
            name, arguments, use_id = next(script)
            if use_id == "finish":
                tool_result = payload["messages"][-1]["content"][0]["toolResult"]
                assert tool_result["toolUseId"] == "lookup"
                assert tool_result["content"][0]["json"]["result"]["source_url"] == SOURCE
                if expired:
                    raise ProviderTransportError("aws_expired_token")
            return {
                "output": {"message": {"role": "assistant", "content": [{
                    "toolUse": {"toolUseId": use_id, "name": name, "input": arguments},
                }]}},
                "stopReason": "tool_use", "usage": {"inputTokens": 7, "outputTokens": 3},
            }

    client = BedrockRootClient(Transport(), config=BedrockConfig(model="global.anthropic.claude-sonnet-4-6"))
    trace = RootController(corpus, client).run("What risk is disclosed?")
    assert trace.tool_calls == 1 and len(trace.model_attempts) == 2
    assert trace.steps[0].observation_sent is not expired
    if expired:
        assert trace.stop_reason == "model_error"
        assert trace.answer is None and trace.coverage.visible_chunks == ()
        assert trace.model_attempts[-1].error == "aws_expired_token"
    else:
        assert trace.stop_reason == "finished"
        assert trace.cited_chunk_ids == (chunk.chunk_id,)
        assert trace.coverage.visible_chunks[0].source_url == SOURCE
        assert sum(attempt.input_tokens for attempt in trace.model_attempts) == 14


@pytest.mark.parametrize("detailed", [False, True], ids=["summary", "trace"])
def test_demo_reports_configured_limits_and_optional_trace(corpus, detailed, monkeypatch, capsys):
    expected = corpus.search_chunks("disruptions", limit=1).items[0].chunk

    class DemoModel(ScriptedModelClient):
        config = SimpleNamespace(model="fixture-model")

        def next_action(self, question, tools, observation, *, remaining_seconds):
            search = next(schema for schema in tools if schema["name"] == "search_chunks")
            assert "4500 UTF-8 bytes" in search["description"]
            assert "limit=1" in question
            assert 0 < remaining_seconds <= 12
            return super().next_action(question, tools, observation, remaining_seconds=remaining_seconds)

    model = DemoModel([
        tool("search_chunks", query="disruptions", limit=0),
        tool("search_chunks", query="disruptions"), finish(expected.chunk_id),
    ])
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    monkeypatch.setattr(BedrockRootClient, "from_env", lambda: model)
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "demo_rlm_claude.py"))
    flags = [
        "--live", "--max-iterations", "3", "--max-tool-calls", "2",
        "--max-response-bytes", "4500", "--deadline-seconds", "12",
    ]
    assert demo["main"](flags + (["--trace"] if detailed else [])) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["limits"] == {
        "max_iterations": 3, "max_tool_calls": 2, "max_response_bytes": 4500, "deadline_seconds": 12,
    }
    assert report["stop_reason"] == "finished"
    assert expected.text not in output
    assert ("steps" in report) is detailed
    if detailed:
        failed, valid, finished = report["steps"]
        assert failed["requested_action"]["arguments"]["limit"] == 0
        assert failed["validated_arguments"] is None
        assert failed["observation_error"]["code"] == "invalid_arguments"
        assert failed["observation_sent"] and valid["observation_sent"]
        assert valid["validated_arguments"]["limit"] == 1
        assert valid["observation_chunk_ids"] == valid["search_discovered_chunk_ids"] == [expected.chunk_id]
        assert finished["requested_action"]["type"] == "finish"
        assert not finished["observation_sent"]


@pytest.mark.parametrize("args", [[], ["--live", "--max-response-bytes", "1"]])
def test_demo_rejects_unauthorized_or_invalid_run_before_provider_setup(args, monkeypatch):
    monkeypatch.setattr(BedrockRootClient, "from_env", lambda: pytest.fail("Provider setup was attempted."))
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "demo_rlm_claude.py"))
    with pytest.raises(SystemExit) as error:
        demo["main"](args)
    assert error.value.code == 2
