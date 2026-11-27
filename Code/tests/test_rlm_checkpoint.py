"""Durable structured state and targeted recursive continuation with local fixtures."""

import json
import runpy
from copy import copy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.checkpoint import (
    CheckpointError, continue_research, create_checkpoint, load_checkpoint, plan_continuation, save_checkpoint,
)
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import ResearchIntent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WORKSPACE = EXAMPLES.parents[1]


@pytest.fixture
def checkpoint_path():
    path = WORKSPACE / f"rlm-checkpoint-test-{uuid4().hex}.json"
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


@pytest.fixture(scope="module")
def demo():
    return runpy.run_path(str(EXAMPLES / "demo_rlm_checkpoint.py"))


def make_initial(corpus, demo, *, batches=2, companies=("AAPL", "AMZN")):
    intent = ResearchIntent(
        question=demo["QUESTION"], company_selection="explicit", companies=companies, fiscal_years=(2024,),
        section_ids=("item_1a",), objective="review", topic="FX costs and component supply", comprehensive=False,
        inferred_fields=(),
    )
    plan = resolve_research_plan(intent, corpus)
    discovery = DiscoverySpec.from_intent(intent, ("foreign currency", "semiconductor products"))
    result = execute_corpus_research(plan, demo["_runner"](corpus), limits=demo["_limits"](batches),
                                     max_groups=1, discovery=discovery)
    return create_checkpoint(corpus, result)


@pytest.fixture(scope="module")
def initial(corpus, demo):
    return make_initial(corpus, demo)


def test_round_trip_retains_partial_work_and_runs_only_failed_assignment(corpus, demo, initial, checkpoint_path):
    save_checkpoint(checkpoint_path, initial)
    loaded = load_checkpoint(checkpoint_path, corpus)
    assert loaded == initial
    assert loaded.plan.required_target_count == 2 and loaded.execution_status == "incomplete"
    assert [finding.evidence[0].source.ticker for finding in loaded.findings] == ["AAPL", "AMZN"]
    original_apple, original_amazon = loaded.findings
    assert loaded.assignments[0].pending_reason is None and loaded.assignments[1].pending_reason == "child_failed"
    assert [group.ticker for group in plan_continuation(loaded).groups] == ["AMZN"]
    assert [group.ticker for group in plan_continuation(loaded, ("AAPL",)).groups] == ["AAPL"]

    updated = continue_research(loaded, demo["_runner"](corpus, continuation=True), limits=demo["_limits"](1))
    save_checkpoint(checkpoint_path, updated)
    restored = load_checkpoint(checkpoint_path, corpus)
    assert restored == updated and restored.plan == initial.plan
    assert restored.execution_status == "finished" and restored.research_status == "incomplete"
    assert restored.findings[:2] == (original_apple, original_amazon) and len(restored.findings) == 3
    last = restored.runs[-1]
    assert last.target_companies == ("AMZN",) and last.batches[0].companies == ("AMZN",)
    assert all("requires 2 filing-section targets" in node.invocation.question for node in last.coverage.invocations
               if node.invocation.depth == 0)
    assert [node.acknowledged_prior_finding_ids for node in last.coverage.invocations] == [(), (original_amazon.finding_id,)]
    assert last.coverage.invocations[0].selected_inherited_finding_ids[-1] == original_amazon.finding_id
    coverage = restored.merged_coverage(corpus)
    assert coverage.required_target_count == initial.plan.required_target_count
    assert tuple(row.status for row in coverage.targets) == ("partial", "partial")
    assert coverage.targets[0] == initial.runs[0].coverage.targets[0]
    assert coverage.targets[1].failed_invocation_ids == initial.runs[0].coverage.targets[1].failed_invocation_ids
    assert len(coverage.targets[1].delivered_span_ids) == 2
    assert len(coverage.targets[1].acknowledged_finding_ids) == 2
    assert len(restored.citations) == 2 and len(restored.evidence) == 3
    assert restored.historical_usage == initial.current_usage
    assert restored.current_usage.corpus_attempts == 3 and restored.cumulative_usage.corpus_attempts == 7
    assert restored.historical_usage.lexical_chunks_scanned == 97 and restored.current_usage.lexical_chunks_scanned == 44
    assert not plan_continuation(restored).groups


def test_default_selects_unattempted_and_preserves_unresolvable_denominator(corpus, demo):
    checkpoint = make_initial(corpus, demo, batches=1, companies=("AAPL", "AMZN", "Unlisted"))
    selection = plan_continuation(checkpoint)
    assert [group.ticker for group in selection.groups] == ["AMZN"] and selection.blocked_companies == ("unlisted",)
    assert checkpoint.plan.required_target_count == 3
    assert checkpoint.merged_coverage(corpus).targets[1].status == "none"
    assert checkpoint.assignments[1].outcome is None
    with pytest.raises(CheckpointError, match="original research plan"):
        plan_continuation(checkpoint, ("NVDA",))


def test_exhausted_continuation_keeps_state_and_records_zero_use(corpus, demo, initial):
    def forbidden(_):
        pytest.fail("An exhausted continuation must spend zero model decisions.")

    runner = demo["_runner"](corpus, continuation=True)
    runner.model_factory = forbidden
    limits = demo["_limits"](1).model_copy(update={"max_provider_attempts": 0})
    updated = continue_research(initial, runner, limits=limits)
    assert updated.runs[-1].stop_reason == "insufficient_provider_attempts"
    assert not updated.runs[-1].batches and updated.current_usage.corpus_attempts == updated.current_usage.lexical_chunks_scanned == 0
    assert updated.findings == initial.findings and updated.merged_coverage(corpus) == initial.merged_coverage(corpus)
    assert updated.historical_usage.provider_attempts == updated.cumulative_usage.provider_attempts
    assert updated.execution_status == "incomplete" and [group.ticker for group in plan_continuation(updated).groups] == ["AMZN"]


def test_explicit_failed_revisit_preserves_previously_completed_assignment(corpus, demo, initial):
    def fail(_):
        raise RuntimeError("Scripted setup failure")

    runner = demo["_runner"](corpus, continuation=True)
    runner.model_factory = fail
    updated = continue_research(initial, runner, limits=demo["_limits"](1), companies=("AAPL",))
    assert updated.runs[-1].target_companies == ("AAPL",)
    assert updated.runs[-1].groups[0].pending_reason == "batch_error"
    assert updated.assignments == initial.assignments and updated.findings == initial.findings
    assert [group.ticker for group in plan_continuation(updated).groups] == ["AMZN"]


@pytest.mark.parametrize("acknowledged", [False, True])
def test_prior_finding_requires_acknowledged_delivery_and_is_scoped(corpus, demo, initial, acknowledged):
    tool, finish = demo["_tool"], demo["_finish"]
    sessions = {}

    class Client(ScriptedModelClient):
        def next_action(self, *args, **kwargs):
            if len(self.observations) == 2:
                self.last_observation_delivered = acknowledged
            else:
                self.last_observation_delivered = True
            return super().next_action(*args, **kwargs)

    def factory(invocation):
        if invocation.depth:
            scripts = [finish("d1"), tool("read_prior_findings", limit=4), finish("d1"), finish()]
        else:
            scripts = [tool("delegate_research", question="Review Amazon.", goal="Check history.",
                            partitions=[asdict(pair) for pair in invocation.partitions]), finish("d1"), finish()]
        sessions[invocation.depth] = Client(scripts) if invocation.depth else ScriptedModelClient(scripts)
        return sessions[invocation.depth]

    runner = RecursiveResearch(corpus, factory, shared_limits=demo["_runner"](corpus).shared_limits)
    updated = continue_research(initial, runner, limits=demo["_limits"](1))
    first_error = json.loads(sessions[1].observations[1])["error"]
    assert first_error["code"] == "invalid_findings"
    offered = json.loads(sessions[1].observations[2])["result"]["findings"]
    assert [item["finding_id"] for item in offered] == [initial.findings[1].finding_id]
    root, child = updated.runs[-1].coverage.invocations
    assert bool(child.acknowledged_prior_finding_ids) == acknowledged
    assert bool(root.cited_chunk_ids) == acknowledged
    if not acknowledged:
        assert json.loads(sessions[1].observations[3])["error"]["code"] == "invalid_findings"


@pytest.mark.parametrize(("field", "value", "message"), [
    ("span_id", "0:1", "unknown span"),
    ("quote", "fabricated passage", "canonical text"),
    ("chunk_id", "no-such-chunk", "unknown chunk"),
    ("chunk_id", "4d1474550060251037c5f981", "outside its authorized scope"),
    ("source_url", "https://example.org", "canonical source"),
    ("document_id", "no-such-document", "canonical source"),
])
def test_load_rejects_invalid_canonical_evidence(corpus, initial, checkpoint_path, field, value, message):
    data = initial.model_dump(mode="json")
    item = data["runs"][0]["coverage"]["invocations"][1]["findings"][0]["evidence"][0]
    target = item if field in ("span_id", "quote") else item["source"]
    target[field] = value
    checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(checkpoint_path, corpus)


@pytest.mark.parametrize("change", ["corpus", "plan", "invocation", "coverage", "unacknowledged", "usage"])
def test_load_rejects_inconsistent_saved_state(corpus, initial, checkpoint_path, change):
    data = initial.model_dump(mode="json")
    run = data["runs"][0]
    if change == "corpus":
        data["corpus_identity"] = "sha256:" + "0" * 64
        error = "corpus identity"
    elif change == "plan":
        data["plan"]["groups"].pop()
        error = "original scope"
    elif change == "invocation":
        run["coverage"]["invocations"][1]["invocation"]["partitions"][0]["section_id"] = "item_7"
        error = "authorized batch"
    elif change == "coverage":
        run["coverage"]["targets"][0]["status"] = "complete"
        error = "coverage summary"
    elif change == "unacknowledged":
        run["coverage"]["invocations"][1]["direct_span_ids"] = []
        error = "lacks recorded evidence access"
    else:
        run["usage"]["corpus_attempts"] += 1
        error = "batch usage"
    checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match=error):
        load_checkpoint(checkpoint_path, corpus)


def test_fingerprint_compares_canonical_content(corpus, initial, checkpoint_path):
    altered = copy(corpus)
    altered.__dict__.pop("snapshot_identity", None)
    altered._filings = corpus._filings.copy()
    key = next(iter(altered._filings))
    filing = altered._filings[key]
    altered._filings[key] = filing.model_copy(update={"primary_document": "different.htm"})
    assert altered.snapshot_identity != corpus.snapshot_identity
    save_checkpoint(checkpoint_path, initial)
    with pytest.raises(CheckpointError, match="corpus identity"):
        load_checkpoint(checkpoint_path, altered)


@pytest.mark.parametrize(("payload", "message"), [
    (b'{"version": 4}', "Unsupported"),
    (b'{"version": true}', "missing or invalid"),
    (b'{"version": 1, "version": 1}', "duplicate keys"),
    (b'{"version": 1,', "Malformed"),
    (b'\xff', "Malformed"),
    (b'{"version": 1, "runs": []}', "does not match its schema"),
    (b'{"version": 1, "value": NaN}', "nonfinite number"),
    (b'{"version": 1, "x": ' + b"[" * 2_000, "Malformed"),
])
def test_load_rejects_malformed_and_unsupported_inputs(corpus, checkpoint_path, payload, message):
    checkpoint_path.write_bytes(payload)
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(checkpoint_path, corpus)


def test_load_checks_file_size_and_nested_schema(corpus, initial, checkpoint_path, monkeypatch):
    with pytest.raises(CheckpointError, match="regular file"):
        load_checkpoint(checkpoint_path.parent, corpus)
    data = initial.model_dump(mode="json")
    data["plan"]["groups"][0]["unexpected"] = "must be rejected"
    checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="does not match its schema"):
        load_checkpoint(checkpoint_path, corpus)
    monkeypatch.setattr("evidenceroute.research.rlm.checkpoint.MAX_CHECKPOINT_BYTES", 128)
    with pytest.raises(CheckpointError, match="input size limit"):
        load_checkpoint(checkpoint_path, corpus)


def test_atomic_save_failure_keeps_old_file(corpus, initial, checkpoint_path, monkeypatch):
    save_checkpoint(checkpoint_path, initial)
    old = checkpoint_path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("mock replace failure")

    monkeypatch.setattr("evidenceroute.research.rlm.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="mock replace failure"):
        save_checkpoint(checkpoint_path, initial)
    assert checkpoint_path.read_bytes() == old
    assert not list(checkpoint_path.parent.glob(f".{checkpoint_path.name}.*.tmp"))


def test_comparative_checkpoint_preserves_assessments_and_cumulative_provider_usage(corpus, checkpoint_path):
    from test_rlm_corpus_research import CountedClient, overall, resolve, runner_for

    plan = resolve(corpus, ["AMD", "AAPL"])
    runner = runner_for(corpus, plan, client=CountedClient)
    result = execute_corpus_research(plan, runner, limits=overall(max_batches=1), max_groups=1,
                                     scan_terms=("substrate",))
    checkpoint = create_checkpoint(corpus, result)
    save_checkpoint(checkpoint_path, checkpoint)
    restored = load_checkpoint(checkpoint_path, corpus)
    assert restored.comparisons == (*result.root_comparisons, *result.acknowledged_child_comparisons)
    assert restored.current_usage.provider_attempts == 5
    updated = continue_research(restored, runner_for(corpus, plan, client=CountedClient), limits=overall(max_batches=1))
    assert updated.execution_status == "finished" and updated.research_status == "incomplete"
    assert updated.historical_usage.provider_attempts == updated.current_usage.provider_attempts == 5
    assert updated.cumulative_usage.provider_attempts == 10 and updated.cumulative_usage.reported_tokens == 92
    assert updated.findings[:2] == restored.findings and len(updated.citations) == 4
    assert len(updated.comparisons) == 4


def test_retained_unacknowledged_finding_requires_new_native_delivery(corpus, demo, monkeypatch):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient
    from test_rlm_recursive import converse_reply, stub_bedrock

    runner = demo["_runner"](corpus)
    normal_factory = runner.model_factory

    def with_failed_delivery(invocation):
        model = normal_factory(invocation)
        ticker = corpus.get_filing(invocation.partitions[0].document_id).ticker
        if invocation.depth == 0 and ticker == "AMZN":
            model.last_observation_delivered = False
        return model

    runner.model_factory = with_failed_delivery
    intent = ResearchIntent(
        question=demo["QUESTION"], company_selection="explicit", companies=("AMZN",), fiscal_years=(2024,),
        section_ids=("item_1a",), objective="review", topic="Supply risk", comprehensive=False, inferred_fields=(),
    )
    plan = resolve_research_plan(intent, corpus)
    result = execute_corpus_research(plan, runner, limits=demo["_limits"](1),
                                     discovery=DiscoverySpec.from_intent(intent, ("semiconductor products",)))
    initial = create_checkpoint(corpus, result)
    assert initial.findings == result.retained_unacknowledged_child_findings and not initial.citations
    fid = initial.findings[0].finding_id
    assign = demo["_tool"]("delegate_research", question="Review Amazon.", goal="Check history.",
                           partitions=[asdict(pair) for pair in plan.resolved_targets])
    finish = demo["_finish"]("d1")
    sessions = stub_bedrock(monkeypatch, [
        [converse_reply(assign, "assign"), converse_reply(finish, "root.finish")],
        [converse_reply(demo["_tool"]("read_prior_findings", limit=1), "history"), converse_reply(finish, "child.finish")],
    ])
    resumed = RecursiveResearch(corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
                                shared_limits=runner.shared_limits)
    updated = continue_research(initial, resumed, limits=demo["_limits"](1))
    assert updated.execution_status == "finished" and updated.findings == initial.findings
    assert updated.current_usage.provider_attempts == 4 and updated.current_usage.corpus_attempts == 1
    assert updated.runs[-1].coverage.invocations[1].acknowledged_prior_finding_ids == (fid,)
    payload = sessions[1].requests[1]
    result_block = payload["messages"][-1]["content"][0]["toolResult"]
    assert result_block["toolUseId"] == "history"
    assert result_block["content"][0]["json"]["result"]["findings"][0]["finding_id"] == fid
    root, child = updated.runs[-1].coverage.invocations
    assert root.selected_inherited_finding_ids == child.selected_inherited_finding_ids == (fid,)


def test_checkpoint_preserves_the_interpretation_charge(corpus, checkpoint_path):
    from evidenceroute.research.rlm.bedrock import BedrockConfig
    from evidenceroute.research.rlm.interpretation import BedrockIntentInterpreter
    from evidenceroute.research.rlm.natural_language import research_question

    example = runpy.run_path(str(EXAMPLES / "demo_rlm_question.py"))
    question = "What foreign exchange risks does Apple disclose in FY2024?"
    factory = example["_scripted_research"](corpus, question)
    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(
        max_depth=1, max_delegations=1, max_partitions=1, max_provider_attempts=8, max_reported_tokens=10_000,
        max_corpus_attempts=5,
    ))
    limits = runpy.run_path(str(EXAMPLES / "demo_rlm_checkpoint.py"))["_limits"](6)

    def interpreter():
        return BedrockIntentInterpreter(example["ScriptedInterpretationTransport"](), config=BedrockConfig(model="fixture"))

    result = research_question(question, runner, limits=limits, interpreter_factory=interpreter)
    checkpoint = create_checkpoint(corpus, result)
    save_checkpoint(checkpoint_path, checkpoint)
    restored = load_checkpoint(checkpoint_path, corpus)
    assert restored.plan.intent.question == question and restored.discovery == result.interpretation.outcome.discovery
    assert restored.interpretation.charged_tokens == result.interpretation.charged_tokens > 0
    assert restored.current_usage.provider_attempts == 1 and restored.current_usage.corpus_attempts == 2
    assert restored.cumulative_usage == result.usage
