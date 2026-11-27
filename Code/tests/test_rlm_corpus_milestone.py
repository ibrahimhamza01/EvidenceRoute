"""Canonical comparative capture, bounded continuation, and verified reporting."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.checkpoint import continue_research, create_checkpoint, load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec, discover_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.verification import BedrockSemanticVerifier, review_checkpoint, verification_report
from evidenceroute.research.rlm.verification_records import VerificationLimits


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PROFILE = COMPARATIVE_PROFILES["full-corpus"]
PHRASES = ("AI infrastructure", "AI processor", "GPU", "AI computing", "machine learning infrastructure")


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


def _plan(corpus, companies=()):
    return resolve_research_plan(interpret_intent("Which companies disclosed a new AI infrastructure risk in FY2024?", lambda *_: {
        "company_selection": "explicit" if companies else "all", "companies": companies,
        "fiscal_years": (2023, 2024), "section_ids": ("item_1a",), "topic": "AI infrastructure risk",
        "objective": "year_over_year", "comprehensive": True, "inferred_fields": (),
    }), corpus)


def _runner(corpus, factory):
    return RecursiveResearch(corpus, factory, shared_limits=PROFILE.shared_limits, local_limits=PROFILE.local_limits,
                             finalization=PROFILE.finalization, comparative_capture=True, max_root_reinvestigation_calls=0)


def _limits(batches, delegations):
    return CorpusLimits(max_batches=batches, max_provider_attempts=18 * batches, max_reported_tokens=100000 * batches,
                        max_corpus_attempts=12 * batches, max_delegations=delegations, max_partitions=2 * delegations,
                        deadline_seconds=180.0 * batches)


def test_full_corpus_checkpoint_resume_without_rescan_or_completed_research(corpus, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import DISCOVERY_PHRASES, _offline_factory
    from demo_rlm_corpus_research import _OfflineVerifierTransport
    from demo_rlm_discovery import KNOWN_TARGETS

    plan = _plan(corpus)
    spec = DiscoverySpec.from_intent(plan.intent, DISCOVERY_PHRASES)
    runner = _runner(corpus, _offline_factory(corpus, plan, compact=True))
    initial = execute_corpus_research(plan, runner, limits=_limits(13, 6).model_copy(
        update={"max_provider_attempts": 72, "max_reported_tokens": 400000, "max_corpus_attempts": 48}),
        discovery=spec)
    assert initial.stop_reason == "insufficient_delegations"
    assert initial.coverage.required_target_count == 50 and initial.group_counts["completed"] == 6
    assert initial.batches[0].companies != tuple(sorted(plan.requested_companies)[:2])
    assert KNOWN_TARGETS <= {item.chunk_id for section in initial.scan.sections for item in section.candidates}
    assert any(not row.candidates for row in initial.scan.sections)
    path = tmp_path / "corpus.json"
    checkpoint = create_checkpoint(corpus, initial)
    assert len(checkpoint.findings) == 2
    save_checkpoint(path, checkpoint)

    def unexpected(*args, **kwargs):
        pytest.fail("Continuation rescanned a settled full candidate inventory")

    monkeypatch.setattr("evidenceroute.research.rlm.corpus_research.discover_research_plan", unexpected)
    restarted = load_checkpoint(path, corpus)
    final = continue_research(restarted, runner, limits=_limits(10, 19))
    assert final.execution_status == "finished" and final.research_status == "incomplete"
    assert final.findings == restarted.findings and final.citations == restarted.citations
    assert set(final.runs[-1].target_companies).isdisjoint({ticker for batch in initial.batches for ticker in batch.companies})
    assert final.merged_coverage(corpus).required_target_count == 50
    assert len(final.assignments) == 25 and all(group.outcome.assignment_status == "completed" for group in final.assignments)
    assert final.runs[-1].batches[-1].companies and len(final.runs[-1].batches[-1].companies) == 1
    assert final.cumulative_usage.partition_assignments == 50
    assert final.current_usage.lexical_chunks_scanned == 0
    assert final.historical_usage.accepted_submissions == 2
    assert all("sec.gov" in citation.source_url for citation in final.citations)
    before = verification_report(final)
    assert all(item["status"] == "unverified" for item in before["conclusions"])
    assert "Verifier-supported component" not in before["answer"]
    verifier = BedrockSemanticVerifier(_OfflineVerifierTransport(), config=BedrockConfig(model="offline"))
    reviewed = review_checkpoint(final, corpus, verifier, limits=VerificationLimits())
    after = verification_report(reviewed)
    assert reviewed.comparisons == final.comparisons and reviewed.findings == final.findings
    assert after["conclusions"][0]["status"] == "partially_supported"
    assert "general data-center power or water constraint" in after["answer"]
    assert "specifically concerns AI" not in after["answer"]
    assert after["evidence_gaps"] and after["conclusions"][0]["novelty"] == "unresolved"
    assert after["combined_usage"].provider_attempts == after["verification_usage"].provider_attempts == 2


class NativeScript:
    def __init__(self, script):
        self.script = iter(script)
        self.requests = []

    def send(self, payload, *, timeout_seconds):
        # Keep a snapshot since the client retains and appends native history.
        self.requests.append(json.loads(json.dumps(payload)))
        name, arguments = next(self.script)
        content = [{"toolUse": {"toolUseId": f"turn-{len(self.requests)}", "name": name, "input": arguments}}]
        stop = "tool_use"
        if name == "truncated":
            stop, content = "max_tokens", [{"text": "partial"}]
        return {"stopReason": stop, "output": {"message": {"role": "assistant", "content": content}},
                "usage": {"inputTokens": 400, "outputTokens": 150}}


@pytest.mark.parametrize("terminal_failure", [True, False])
def test_native_capture_is_reserved_and_independent_records_survive(corpus, terminal_failure):
    plan = _plan(corpus, ("AMD",))
    spec = DiscoverySpec.from_intent(plan.intent, PHRASES)
    chunks = [corpus.get_chunk(key) for key in ("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf")]
    evidence = [{"chunk_id": chunk.chunk_id, "span_id": next(
        key for key, offsets in passage_spans(chunk.text).items()
        if "Data centers depend on access to clean water and reliable energy" in chunk.text[slice(*offsets)])}
        for chunk in chunks]
    reads = [("get_spans", {"chunk_id": item["chunk_id"], "span_ids": [item["span_id"]]}) for item in evidence]

    def draft(claim, refs):
        return {"claim": claim, "analysis_status": "analyzed", "evidence": refs}

    claim = "Power and water shortages can limit customers’ data center expansion."
    capture = ("submit_findings", {"findings": [
        draft(claim, evidence[:1]), draft(claim, evidence), draft("FY2023 and FY2024 face this risk.", evidence[:1]),
        draft(claim, evidence[1:]),
    ]})
    risk = "Data center power and water constraints"
    good = {"ref": "s2", "risk_category": "general_infrastructure", "entailment": "supports"}
    bad = {"ref": "s1", "risk_category": "general_infrastructure", "entailment": "nonmention",
           "rationale": "x" * 251, "infrastructure_connection": "x" * 251}
    if terminal_failure:
        finishes = [("truncated", {}), ("finish_research", {"synthesis": {"cite_refs": ["s8"]}})]
    else:
        finishes = [("finish_research", {"synthesis": {"comparisons": [
            {"risk": risk, "relationship": "conflicting", "assessments": [bad, good]},
            {"risk": risk, "relationship": "conflicting", "assessments": [
                {"ref": "s1", "risk_category": "general_infrastructure", "entailment": "nonmention"}, good,
            ]},
        ]}})]
    children = NativeScript([*reads, reads[0], capture, *finishes])
    root_synthesis = {"cite_refs": ["d1", "d2"]}
    if not terminal_failure:
        root_synthesis["comparisons"] = [{
            "risk": risk, "relationship": "same", "assessments": [
                {"ref": ref, "risk_category": "general_infrastructure", "entailment": "supports"} for ref in ("d1", "d2")
            ],
        }]
    root = NativeScript([("delegate_research", {
        "question": "Look for scoped infrastructure risks.", "goal": "Inspect both years.",
        "partitions": [asdict(pair) for pair in plan.groups[0].targets],
    }), ("finish_research", {"synthesis": root_synthesis})])
    def factory(invocation):
        return BedrockRootClient(children if invocation.depth else root, config=BedrockConfig(model="offline"))
    result = execute_corpus_research(plan, _runner(corpus, factory), limits=_limits(1, 2), discovery=spec, detailed_trace=True)
    assert result.usage.finding_submission_attempts == 1
    assert result.usage.accepted_submissions == 2 and result.usage.rejected_submission_records == 2
    assert result.usage.unused_capture_reservations == 0
    assert len(result.acknowledged_child_findings) == 2 and len(result.citations) == 2
    assert {item.source.fiscal_year for finding in result.acknowledged_child_findings for item in finding.evidence} == {2023, 2024}
    assert children.requests[3]["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_findings"}}
    assert "new_findings" not in json.dumps(children.requests[4]["toolConfig"])
    assert children.requests[4]["messages"][-1]["content"][0]["toolResult"]["toolUseId"] == "turn-4"
    diagnostic = result.diagnostics[1]
    capture_issues = diagnostic["decisions"][3]["operations"][0]["finding_validation"]
    assert {issue["code"] for issue in capture_issues} == {"mixed_finding_scope", "cross_year_claim"}
    if terminal_failure:
        assert children.requests[4]["messages"] == children.requests[5]["messages"]
        assert result.groups[0].pending_reason == "child_failed" and result.execution_status == "incomplete"
        assert result.usage.provider_attempts == 8 and result.usage.reported_tokens == 8 * 550
    else:
        assert result.execution_status == "finished"
        assert {item.outcome for item in result.acknowledged_child_comparisons} == {
            "insufficient", "possible_new_needs_review",
        }
        assert all(item.relationship == "unclear" for item in result.acknowledged_child_comparisons)
        assert all(item["disagreement_unresolved"] for item in verification_report(
            create_checkpoint(corpus, result))["conclusions"])


def test_full_corpus_expansion_gates_ai_anchors_for_unrelated_topics(corpus):
    plan = _plan(corpus)
    foreign = DiscoverySpec(topic="Foreign exchange risk", phrases=("foreign exchange", "currency fluctuations"))
    old = discover_research_plan(corpus, plan, foreign.model_copy(update={"expand_context": False}))
    new = discover_research_plan(corpus, plan, foreign)
    assert len(new.sections) == 50
    assert [[hit.chunk_id for hit in row.candidates] for row in old.sections] == [
        [hit.chunk_id for hit in row.candidates] for row in new.sections
    ]


def test_overlapping_identical_passages_reference_acknowledged_original(corpus):
    from evidenceroute.research.rlm.tools import CorpusToolDispatcher

    first, second = [corpus.get_chunk(key) for key in ("4344bb6cd24c5716179bdd20", "dbb6f8aa75a75f2684214e0e")]
    first_span, second_span = "1611:1716", "94:199"
    quote = first.text[slice(*passage_spans(first.text)[first_span])]
    assert quote == second.text[slice(*passage_spans(second.text)[second_span])]
    dispatcher = CorpusToolDispatcher(corpus, max_response_bytes=6000, include_spans=True)
    args = {"chunk_id": second.chunk_id, "span_ids": [second_span]}
    fresh = dispatcher.dispatch("get_spans", args)
    duplicate = dispatcher.dispatch("get_spans", args, delivered_spans={
        (first.document_id, first.section_id, quote): (first.chunk_id, first_span),
    })
    body = json.loads(duplicate.content)
    assert duplicate.evidence == () and body["span_ids"] == {}
    assert body["result"]["already_delivered"][second_span] == {"chunk_id": first.chunk_id, "span_id": first_span}
    assert len(duplicate.content) < len(fresh.content)


def test_explicit_reinvestigation_keeps_old_reviews_and_checks_new_conclusions(corpus, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import _offline_factory
    from demo_rlm_corpus_research import _OfflineVerifierTransport

    plan = _plan(corpus, ("AMD",))
    spec = DiscoverySpec.from_intent(plan.intent, PHRASES)
    runner = _runner(corpus, _offline_factory(corpus, plan, compact=True))
    limits = _limits(1, 1)
    checkpoint = create_checkpoint(corpus, execute_corpus_research(plan, runner, limits=limits, discovery=spec))
    assert checkpoint.version == 3
    def verifier():
        return BedrockSemanticVerifier(_OfflineVerifierTransport(), config=BedrockConfig(model="offline"))
    first = review_checkpoint(checkpoint, corpus, verifier(), limits=VerificationLimits())
    assert continue_research(first, runner, limits=limits) is first
    reinvestigated = continue_research(first, runner, limits=limits, companies=("AMD",), goal="Find a more exact AI link.")
    assert reinvestigated.findings[:2] == first.findings and len(reinvestigated.findings) == 4
    assert reinvestigated.verification_runs == first.verification_runs
    report = verification_report(reinvestigated)
    assert [item["status"] for item in report["conclusions"]] == ["unverified"]
    reviewed = review_checkpoint(reinvestigated, corpus, verifier(), limits=VerificationLimits())
    assert reviewed.verification_runs[:1] == first.verification_runs
    assert [item["status"] for item in verification_report(reviewed)["conclusions"]] == ["partially_supported"]


@pytest.mark.parametrize(("relationship", "earlier", "expected_relationship", "outcome"), [
    ("more_specific", "supports", "more_specific", "comparable_both_years"),
    ("different", "supports", "different", "insufficient"),
    ("conflicting", "contradicts", "conflicting", "conflicting"),
    ("conflicting", "nonmention", "unclear", "possible_new_needs_review"),
])
def test_comparison_separates_passage_support_relationship_and_ai_connection(
    corpus, monkeypatch, relationship, earlier, expected_relationship, outcome,
):
    from evidenceroute.research.rlm.comparison import CompactComparisonDraft, resolve_reference_comparison

    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import _offline_factory

    plan = _plan(corpus, ("AMD",))
    runner = _runner(corpus, _offline_factory(corpus, plan, compact=True))
    result = execute_corpus_research(plan, runner, limits=_limits(1, 1),
                                     discovery=DiscoverySpec.from_intent(plan.intent, PHRASES))
    findings = {f"s{index}": finding for index, finding in enumerate(result.acknowledged_child_findings, 1)}
    draft = CompactComparisonDraft.model_validate({
        "risk": "Customer data-center capacity constraints", "relationship": relationship, "assessments": [
            {"ref": "s1", "risk_category": "general_infrastructure", "entailment": earlier},
            {"ref": "s2", "risk_category": "ai_enabled_threat", "entailment": "supports", "ai_specificity": "explicit"},
        ],
    })
    comparison = resolve_reference_comparison(draft, findings, (2023, 2024))
    assert comparison.relationship == expected_relationship and comparison.outcome == outcome
    # The retained assessment distinguishes the proposed risk category and the
    # absence of an AI reference in the exact source.
    assert comparison.assessments[1].risk_category == "ai_enabled_threat"
    assert comparison.ai_specific_years == ()
