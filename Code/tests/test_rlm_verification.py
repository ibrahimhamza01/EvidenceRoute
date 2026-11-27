"""Offline semantic verification through native Converse and historical checkpoints."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.checkpoint import CheckpointError, continue_research, load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.claude import ProviderTransportError
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import observation_json
from evidenceroute.research.rlm.verification import (
    BedrockSemanticVerifier, VerificationLimits, record_verification, verification_report, verify_checkpoint,
)
from evidenceroute.research.rlm.verification_records import RecoveryCandidate, comparison_id

WORKSPACE = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


@pytest.fixture(scope="module")
def demo():
    return runpy.run_path(str(WORKSPACE / "Code/examples/demo_rlm_verification.py"))


@pytest.fixture(scope="module")
def apple(corpus, demo):
    return demo["fixture_checkpoint"](corpus, "AAPL")


@pytest.fixture(scope="module")
def alphabet(corpus, demo):
    return demo["fixture_checkpoint"](corpus, "GOOGL")


@pytest.fixture
def checkpoint_path():
    path = WORKSPACE / f"rlm-verification-test-{uuid4().hex}.json"
    yield path
    path.unlink(missing_ok=True)


class Transport:
    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def send(self, payload, *, timeout_seconds):
        self.requests.append((payload, timeout_seconds))
        return self.respond(payload)


def _client(transport, **options):
    return BedrockSemanticVerifier(transport, config=BedrockConfig(model="fixture", max_retries=2, max_output_tokens=768),
                                   **options)


def _native(reviews, *, stop="tool_use", input_tokens=600, output_tokens=200):
    return {
        "stopReason": stop, "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "private provider content"}}},
            {"toolUse": {"toolUseId": "review:test.1", "name": "submit_verification", "input": {"reviews": reviews}}},
        ]}},
        "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens},
    }


def _cases(payload):
    return json.loads(payload["messages"][0]["content"][0]["text"])["cases"]


def _fixture_response(demo, payload):
    return _native([demo["_fixture_review"](case) for case in _cases(payload)])


def test_apple_reviews_compound_claims_from_selected_spans_and_preserves_incomplete_state(corpus, apple, demo):
    transport = Transport(lambda payload: _fixture_response(demo, payload))
    run = verify_checkpoint(apple, corpus, _client(transport), limits=VerificationLimits())
    assert [item.status for item in run.items] == ["partially_supported", "partially_supported", "supported", "contradicted"]
    assert run.usage.provider_attempts == 2 and run.usage.reported_tokens == run.usage.charged_tokens == 1_600
    assert run.usage.estimated_tokens > 0 and run.usage.attempts_with_missing_usage == 0
    assert len({request["messages"][0]["content"][0]["text"] for request, _ in transport.requests}) == 2
    assert all(len(request["messages"]) == 1 and request["messages"][0]["role"] == "user" for request, _ in transport.requests)
    payload, _ = transport.requests[0]
    assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_verification"}}
    assert [item["toolSpec"]["name"] for item in payload["toolConfig"]["tools"]] == ["submit_verification"]
    assert payload["inferenceConfig"]["maxTokens"] == 768
    assert "analysis_status" not in json.dumps(_cases(payload))
    strengthening, hedging = _cases(payload)
    assert [item["text"] for item in strengthening["selected_evidence"]] == [apple.findings[0].evidence[0].quote]
    assert [item["text"] for item in hedging["selected_evidence"]] == [apple.findings[1].evidence[0].quote]
    assert "product components" not in strengthening["selected_evidence"][0]["text"]
    assert "option contracts" not in hedging["selected_evidence"][0]["text"]
    assert {"848:1098", "1099:1275", "1276:1441", "1442:1667"} <= passage_spans(corpus.get_chunk(demo["APPLE_CHUNK"]).text).keys()
    assert strengthening["selected_evidence"][0]["text"].startswith("dollar,")
    assert any("component-cost" in (item.gap or "") for item in run.items[0].components)
    assert run.items[1].components[0].status == "insufficient_evidence"
    assert run.items[1].components[1].evidence[0].finding_id == apple.findings[1].finding_id

    updated = record_verification(apple, corpus, run)
    report = verification_report(updated)
    assert updated.findings == apple.findings and updated.runs == apple.runs
    assert report["execution_status"] == report["research_status"] == "incomplete"
    assert report["required_targets"] == 1 and len(report["evidence_gaps"]) == 2
    assert all(item.partitions == apple.plan.resolved_targets for item in report["evidence_gaps"])
    assert report["research_usage"].provider_attempts == 0 and report["combined_usage"].provider_attempts == 2
    source = apple.findings[0].evidence[0].source
    assert report["answer_citations"] == report["finding_citations"] == (source,)
    assert report["conclusion_citations"] == ()
    assert report["answer"].count(f"excerpt [{source.chunk_id}]") == 3
    assert "private provider content" not in observation_json(report)


def test_alphabet_distinguishes_more_specific_wording_from_explicit_contradiction(corpus, alphabet, demo):
    transport = Transport(lambda payload: _fixture_response(demo, payload))
    run = verify_checkpoint(alphabet, corpus, _client(transport), limits=VerificationLimits(), finding_ids=())
    conflict, comparable, different = run.items
    assert all(item.kind == "comparison" for item in run.items)
    assert [item.target_id for item in run.items] == [comparison_id(comparison) for comparison in alphabet.comparisons]
    assert conflict.status == "insufficient_evidence"
    assert conflict.comparison.earlier_year == "nonmention"
    assert conflict.comparison.relationship == comparable.comparison.relationship == "more_specific_wording"
    assert conflict.comparison.novelty == "shared_risk_in_selected_evidence"
    assert comparable.status == "supported"
    assert different.comparison.relationship == "different_risk"
    assert different.comparison.novelty == "possible_novelty_needs_review"
    contexts = [case for payload, _ in transport.requests for case in _cases(payload)]
    assert all({item["year"] for item in case["selected_evidence"]} == {2023, 2024} for case in contexts)
    assert "Investigator judgment" not in json.dumps(contexts)
    updated = record_verification(alphabet, corpus, run)
    assert updated.comparisons == alphabet.comparisons and updated.research_status == "incomplete"
    report = verification_report(updated)
    assert report["evidence_gaps"][0].partitions == alphabet.plan.resolved_targets[:1]
    assert all(c["disagreement_unresolved"] and c["verified_relationship"] is None for c in report["conclusions"])
    assert all(not c["relationship_evidence"] for c in report["conclusions"])
    assert report["conclusion_citations"] == report["answer_citations"] == ()
    assert "Investigator disagreement remains unresolved." in report["answer"]


@pytest.mark.parametrize(("relationship", "comparison_index", "statement"), [
    ("same_risk", 1, "The selected earlier and later disclosures describe the same underlying risk."),
    ("more_specific_wording", 1,
     "The later disclosure provides more specific wording about an underlying risk described in both years."),
    ("different_risk", 2,
     "The selected disclosures describe different underlying risks. Filing-wide novelty remains unresolved."),
    ("unclear", 2, None),
])
def test_report_cites_relationship_evidence_separately_from_components(
    corpus, alphabet, demo, relationship, comparison_index, statement,
):
    comparison = alphabet.comparisons[comparison_index]
    research_run, = alphabet.runs
    checkpoint = alphabet.model_copy(update={"runs": (
        research_run.model_copy(update={"root_comparisons": (comparison,)}),
    )})
    unverified = verification_report(checkpoint)
    assert unverified["conclusions"][0]["verified_relationship"] is None
    assert unverified["answer_citations"] == ()

    def response(payload):
        reviews = []
        for case in _cases(payload):
            review = demo["_fixture_review"](case)
            if case["kind"] == "comparison":
                review["comparison"]["relationship"] = relationship
                review["components"][0]["evidence"] = ["e2"]
            reviews.append(review)
        return _native(reviews)

    run = verify_checkpoint(checkpoint, corpus, _client(Transport(response)), limits=VerificationLimits(),
                            finding_ids=() if relationship == "unclear" else None)
    updated = record_verification(checkpoint, corpus, run)
    report = verification_report(updated)
    conclusion, = report["conclusions"]
    review = next(item for item in run.items if item.kind == "comparison")
    assert conclusion["novelty"] == "unresolved"

    earlier, later = (
        next(f.evidence[0].source for f in alphabet.findings if f.finding_id == fid)
        for fid in comparison.finding_ids
    )
    if statement is None:
        assert conclusion["verified_relationship"] is None and not conclusion["relationship_evidence"]
        assert review.comparison.relationship_evidence
        assert "The relationship remains unresolved." in report["answer"]
        assert f"Verifier-supported component. {review.components[0].statement} [{later.chunk_id}]" in report["answer"]
        assert report["answer_citations"] == report["conclusion_citations"] == (later,)
    else:
        assert conclusion["verified_relationship"] == relationship
        assert conclusion["relationship_evidence"] == review.comparison.relationship_evidence
        assert f"{statement} [{earlier.chunk_id}, {later.chunk_id}]" in report["answer"]
        assert report["answer_citations"] == report["conclusion_citations"] == (earlier, later)
        assert set(report["finding_citations"]) > {earlier, later}
    assert "Reviewed FY" not in report["answer"]
    assert report["citations"] == alphabet.citations
    assert report["original_findings"] == alphabet.findings
    assert report["original_comparisons"] == (comparison,)
    assert tuple(item for item in report["items"] if item.status != "unverified") == run.items
    assert report["research_usage"] == alphabet.cumulative_usage
    assert report["verification_usage"] == run.usage


@pytest.mark.parametrize("status", ["supported", "partially_supported", "contradicted"])
def test_report_cites_only_publishable_recovery_relationships(corpus, alphabet, status):
    earlier, later, _ = alphabet.findings
    proposal = RecoveryCandidate(ticker="GOOGL", years=(2023, 2024), finding_ids=(earlier.finding_id, later.finding_id))

    def response(payload):
        case, = _cases(payload)
        components = [{"statement": "Supply interruptions can delay expansion.", "status": "supported", "evidence": ["e1"]}]
        if status == "partially_supported":
            components.append({"statement": "The delay is certain.", "status": "insufficient_evidence",
                               "evidence": [], "gap": "The disclosure describes a possibility."})
        elif status == "contradicted":
            components[0]["status"] = "contradicted"
        return _native([{
            "ref": case["ref"], "components": components,
            "comparison": {"relationship": "more_specific_wording", "relationship_evidence": ["e1", "e2", "e3"],
                           "earlier_year": "supports_risk", "earlier_evidence": ["e1"]},
        }])

    run = verify_checkpoint(alphabet, corpus, _client(Transport(response)), limits=VerificationLimits(),
                            finding_ids=(), comparison_ids=(), recovery_ids=(proposal.target_id,), proposals=(proposal,))
    report = verification_report(record_verification(alphabet, corpus, run))
    recovery, = report["recovered_comparisons"]
    assert recovery["status"] == status
    assert recovery["relationship_evidence"] == run.items[0].comparison.relationship_evidence
    assert recovery["novelty"] == "unresolved"
    if status == "contradicted":
        assert recovery["verified_relationship"] is None
        assert report["answer_citations"] == report["conclusion_citations"] == ()
        assert "Checkpoint recovery [contradicted]. The relationship remains unresolved." in report["answer"]
    else:
        assert recovery["verified_relationship"] == "more_specific_wording"
        sources = (earlier.evidence[0].source, later.evidence[0].source)
        assert report["answer_citations"] == report["conclusion_citations"] == sources
        assert (f"Checkpoint recovery [{status}]. The later disclosure provides more specific wording about an "
                f"underlying risk described in both years. [{sources[0].chunk_id}, {sources[1].chunk_id}]") in report["answer"]


@pytest.mark.parametrize("problem", ["nonmention_contradiction", "wrong_year", "missing_year", "different_risk"])
def test_comparative_relationship_validation_and_qualified_novelty(corpus, alphabet, demo, problem):
    def response(payload):
        review = demo["_fixture_review"](_cases(payload)[0])
        comparison = review["comparison"]
        if problem == "nonmention_contradiction":
            review["components"] = [{"statement": "The earlier passage opposes the later risk.",
                                     "status": "contradicted", "evidence": ["e1"]}]
        elif problem == "wrong_year":
            comparison["earlier_evidence"] = ["e2"]
        elif problem == "missing_year":
            comparison["relationship_evidence"] = ["e2", "e3"]
        else:
            comparison["relationship"] = "different_risk"
        return _native([review])

    run = verify_checkpoint(alphabet, corpus, _client(Transport(response)), limits=VerificationLimits(),
                            finding_ids=(), comparison_ids=(comparison_id(alphabet.comparisons[0]),))
    item, = run.items
    if problem == "different_risk":
        assert item.comparison.novelty == "possible_novelty_needs_review" and item.comparison.earlier_year == "nonmention"
    else:
        assert item.status == "unverified" and item.unverified_reason == "invalid_assessment"


@pytest.mark.parametrize(("problem", "reason"), [
    ("tool_id", "invalid_response"), ("duplicate_tools", "invalid_response"), ("duplicate_reviews", "invalid_response"),
    ("wrong_ref", "invalid_response"), ("truncated", "truncated_response"), ("missing", "missing_assessment"),
    ("bad_evidence", "invalid_assessment"), ("unexpected_quote", "invalid_assessment"),
])
def test_native_verifier_rejects_malformed_output_without_erasing_valid_peers(corpus, apple, demo, problem, reason):
    def response(payload):
        result = _fixture_response(demo, payload)
        blocks = result["output"]["message"]["content"]
        tool = blocks[-1]["toolUse"]
        reviews = tool["input"]["reviews"]
        if problem == "tool_id":
            tool["toolUseId"] = "invalid tool ID"
        elif problem == "duplicate_tools":
            blocks.append(blocks[-1])
        elif problem == "duplicate_reviews":
            reviews[1] = reviews[0]
        elif problem == "wrong_ref":
            reviews[0]["ref"] = "v4"
        elif problem == "truncated":
            result["stopReason"] = "max_tokens"
        elif problem == "missing":
            reviews.pop(0)
        elif problem == "bad_evidence":
            reviews[0]["components"][1]["evidence"] = ["e9"]
        else:
            reviews[0]["components"][0]["quote"] = "private provider content"
        return result

    run = verify_checkpoint(apple, corpus, _client(Transport(response)), limits=VerificationLimits(max_items=2))
    assert run.usage.provider_attempts == 1
    assert run.items[0].status == "unverified" and run.items[0].unverified_reason == reason
    assert all(item.unverified_reason == "item_limit" for item in run.items[2:])
    if problem in {"missing", "bad_evidence", "unexpected_quote"}:
        assert run.items[1].status == "partially_supported"
    else:
        assert run.items[1].status == "unverified"
    assert "private provider content" not in run.model_dump_json()
    if reason == "invalid_assessment":
        assert run.items[0].diagnostics
        assert run.items[0].diagnostics[0].field.startswith("components")


@pytest.mark.parametrize(("options", "reason"), [
    ({"max_provider_attempts": 0}, "attempt_limit"),
    ({"max_tokens": 0}, "token_limit"),
    ({"max_tokens": 50}, "token_limit"),
    ({"max_input_bytes": 256}, "input_limit"),
    ({"max_items": 0}, "item_limit"),
])
def test_verification_preflight_limits_make_no_provider_requests(corpus, apple, options, reason):
    def forbidden(_):
        pytest.fail("Budget forbids the provider call.")

    run = verify_checkpoint(apple, corpus, _client(Transport(forbidden)), limits=VerificationLimits(**options))
    assert all(item.status == "unverified" and item.unverified_reason == reason for item in run.items)
    assert run.usage.provider_attempts == run.usage.charged_tokens == 0


@pytest.mark.parametrize("case", ["attempt", "reported_overage", "output", "deadline", "transport"])
def test_late_or_failed_verification_settles_usage_without_implicit_acceptance(corpus, apple, demo, case):
    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()
    limits = VerificationLimits(batch_size=1, max_provider_attempts=1 if case == "attempt" else 2,
                                max_tokens=10_000, max_output_tokens=350, deadline_seconds=5.0)

    def response(payload):
        if case == "transport":
            raise ProviderTransportError("aws_timeout", retryable=True)
        result = _fixture_response(demo, payload)
        if case == "reported_overage":
            result["usage"]["inputTokens"] = 20_000
        if case == "output":
            result["usage"]["outputTokens"] = 351
        if case == "deadline":
            clock.value = 6.0
        return result

    transport = Transport(response)
    run = verify_checkpoint(apple, corpus, _client(transport, clock=clock), limits=limits, clock=clock)
    assert all(timeout <= limits.deadline_seconds for _, timeout in transport.requests)
    assert all(payload["inferenceConfig"]["maxTokens"] == 350 for payload, _ in transport.requests)
    if case == "attempt":
        assert run.items[0].status == "partially_supported" and run.items[1].unverified_reason == "attempt_limit"
    else:
        code = {"reported_overage": "token_limit", "output": "output_limit", "deadline": "deadline",
                "transport": "transport_error"}[case]
        assert run.items[0].unverified_reason == code and all(item.status == "unverified" for item in run.items)
    assert run.usage.provider_attempts == len(transport.requests)
    assert run.usage.provider_attempts == (2 if case in {"output", "transport"} else 1)
    if case == "reported_overage":
        assert run.usage.charged_tokens == 20_200
    if case == "transport":
        assert run.usage.attempts_with_missing_usage == 2
        assert run.usage.charged_tokens == run.usage.estimated_tokens > 0


def test_round_trip_and_cumulative_usage_preserve_historical_assessments(corpus, apple, demo, checkpoint_path):
    save_checkpoint(checkpoint_path, apple)
    assert "verification_runs" not in json.loads(checkpoint_path.read_text())
    restored = load_checkpoint(checkpoint_path, corpus)
    run = verify_checkpoint(restored, corpus, _client(Transport(lambda payload: _fixture_response(demo, payload))),
                            limits=VerificationLimits())
    recorded = record_verification(restored, corpus, run)
    repeat = verify_checkpoint(recorded, corpus, _client(Transport(lambda payload: _fixture_response(demo, payload))),
                               finding_ids=(apple.findings[2].finding_id,), limits=VerificationLimits(max_items=1))
    recorded = record_verification(recorded, corpus, repeat)
    stopped = verify_checkpoint(recorded, corpus, _client(Transport(lambda _: pytest.fail("no send"))),
                                limits=VerificationLimits(max_provider_attempts=0))
    recorded = record_verification(recorded, corpus, stopped)
    save_checkpoint(checkpoint_path, recorded)
    reloaded = load_checkpoint(checkpoint_path, corpus)
    assert reloaded == recorded and reloaded.version == 2
    assert reloaded.verification_usage.provider_attempts == reloaded.combined_usage.provider_attempts == 3
    assert reloaded.verification_usage.reported_tokens == 2_400
    assert reloaded.cumulative_usage == apple.cumulative_usage and reloaded.findings == apple.findings
    assert verification_report(reloaded)["items"] == run.items
    assert verification_report(reloaded)["current_verification_usage"].provider_attempts == 0


@pytest.mark.parametrize("change", ["source_runs", "unknown_case", "bad_evidence_index", "changed_claim"])
def test_checkpoint_load_rejects_invalid_verification_links(corpus, apple, demo, checkpoint_path, change):
    run = verify_checkpoint(apple, corpus, _client(Transport(lambda payload: _fixture_response(demo, payload))),
                            limits=VerificationLimits())
    data = record_verification(apple, corpus, run).model_dump(mode="json")
    review = data["verification_runs"][0]
    if change == "source_runs":
        review["research_run_ids"] = ["0" * 32]
        error = "source research"
    elif change == "unknown_case":
        review["items"][0]["target_id"] = "other-case"
        review["items"][0]["finding_ids"] = ["other-case"]
        error = "outside its source"
    elif change == "changed_claim":
        data["runs"][0]["coverage"]["invocations"][1]["findings"][0]["claim"] = "Claim changed under the same ID."
        error = "original claim"
    else:
        review["items"][0]["components"][1]["evidence"][0]["evidence_index"] = 2
        error = "outside its case"
    checkpoint_path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match=error):
        load_checkpoint(checkpoint_path, corpus)


def test_continuation_preserves_old_verification_and_new_findings_stay_unverified(corpus, apple, demo):
    run = verify_checkpoint(apple, corpus, _client(Transport(lambda payload: _fixture_response(demo, payload))),
                            limits=VerificationLimits())
    checked = record_verification(apple, corpus, run)
    tool = demo["_tool"]

    def factory(inv):
        if inv.depth == 0:
            assign = tool("delegate_research", question=apple.plan.intent.question, goal="Investigate the component cost gap.",
                          partitions=[asdict(pair) for pair in inv.partitions])
            return ScriptedModelClient([assign, {"type": "synthesis", "synthesis": {"cite_refs": ["d1"]}}])
        draft = demo["_draft"]("Strengthening may increase component costs and hurt margins.", demo["APPLE_CHUNK"], "1099:1275")
        return ScriptedModelClient([tool("get_spans", chunk_id=demo["APPLE_CHUNK"], span_ids=["1099:1275"]),
                                    tool("submit_findings", findings=[draft]),
                                    {"type": "synthesis", "synthesis": {"cite_refs": ["s1"]}}])

    shared = SharedLimits(max_depth=1, max_provider_attempts=8, max_corpus_attempts=6, max_delegations=1, max_partitions=2)
    limits = CorpusLimits(max_batches=1, max_provider_attempts=8, max_reported_tokens=60_000, max_corpus_attempts=6,
                         max_delegations=1, max_partitions=2, deadline_seconds=60.0)
    updated = continue_research(checked, RecursiveResearch(corpus, factory, shared_limits=shared), limits=limits)
    statuses = verification_report(updated)["items"]
    assert updated.execution_status == "finished" and updated.research_status == "incomplete"
    assert statuses[:-1] == run.items and statuses[-1].unverified_reason == "not_requested"
    assert updated.verification_runs == (run,) and updated.findings[:-1] == apple.findings


def test_cli_requires_explicit_limits_for_live_verification(demo):
    with pytest.raises(SystemExit, match="2"):
        demo["main"](["--live"])
