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
QUALITY = COMPARATIVE_PROFILES["full-corpus-quality"]
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


def _runner(corpus, factory, profile=PROFILE):
    return RecursiveResearch(corpus, factory, shared_limits=profile.shared_limits, local_limits=profile.local_limits,
                             finalization=profile.finalization, comparative_capture=True, max_root_reinvestigation_calls=0)


def _limits(batches, delegations, profile=PROFILE):
    shared = profile.shared_limits
    return CorpusLimits(max_batches=batches, max_provider_attempts=shared.max_provider_attempts * batches,
                        max_reported_tokens=shared.max_reported_tokens * batches,
                        max_corpus_attempts=shared.max_corpus_attempts * batches, max_delegations=delegations,
                        max_partitions=2 * delegations, deadline_seconds=shared.deadline_seconds * batches)


@pytest.mark.parametrize("profile", [PROFILE, QUALITY])
def test_full_corpus_checkpoint_resume_without_rescan_or_completed_research(corpus, monkeypatch, tmp_path, profile):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import DISCOVERY_PHRASES, _offline_factory
    from demo_rlm_corpus_research import _OfflineVerifierTransport
    from demo_rlm_discovery import KNOWN_TARGETS

    plan = _plan(corpus)
    spec = DiscoverySpec.from_intent(plan.intent, DISCOVERY_PHRASES)
    runner = _runner(corpus, _offline_factory(corpus, plan, compact=True), profile)
    initial = execute_corpus_research(plan, runner, limits=_limits(4, 6, profile), discovery=spec)
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
    final = continue_research(restarted, runner, limits=_limits(13, 25, profile))
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
        action = next(self.script)
        calls = action if isinstance(action, list) else [action]
        turn = f"turn-{len(self.requests)}"
        content = [
            {"toolUse": {"toolUseId": f"{turn}-{index}" if len(calls) > 1 else turn, "name": name, "input": arguments}}
            for index, (name, arguments) in enumerate(calls, 1)
        ]
        stop = "tool_use"
        if calls[0][0] == "truncated":
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


# Exact direct deliveries in the AMD Sonnet run 4837bdbd50f34b999b2543b88f8b0e20.
_AMD_LIVE_READS = (
    ("e7fbf3d91de1c9c912d9ab11", ("970:1167",)),
    ("e8e013b9b919159fb6168bca", ("626:787", "788:999")),
    ("26bd14970249e9ca31c54908", ("1319:1486",)),
    ("a13ada334ad8f009b2a43f01", ("359:520", "521:724")),
)
_AMD_MECHANISMS = (
    ("043757f52188a8c6fb2a2857", ("296:513",)),
    ("483314af3b949a29920abbbf", ("393:680",)),
)


def _span_calls(entries):
    return [("get_spans", {"chunk_id": key, "span_ids": list(spans)}) for key, spans in entries]


def _assign(group):
    return ("delegate_research", {
        "question": "Review the infrastructure risk disclosures.",
        "goal": "Read the mechanism in both years when available.",
        "partitions": [asdict(pair) for pair in group.targets],
    })


@pytest.mark.parametrize("case", ["abstain", "later_only", "both", "failed_finish"])
def test_capture_replays_exact_amd_evidence(corpus, monkeypatch, case):
    from evidenceroute.research.rlm.checkpoint import plan_continuation

    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import DISCOVERY_PHRASES

    profile = PROFILE if case == "abstain" else QUALITY
    plan = _plan(corpus, ("AMD",))
    spec = DiscoverySpec.from_intent(plan.intent, DISCOVERY_PHRASES)
    search = [("search_chunks", {"query": "power water shortages", "filters": {"fiscal_years": [year]}})
              for year in (2024, 2023)]
    mechanisms = (() if case == "abstain" else _AMD_MECHANISMS[1:] if case == "later_only" else _AMD_MECHANISMS)
    followup = _span_calls(mechanisms) + search
    if case == "later_only":
        followup.append(("list_candidates", asdict(plan.groups[0].targets[0])))
    drafts = [{
        "claim": "Potential power or water shortages could limit customers' data-center expansion.",
        "analysis_status": "analyzed",
        "uncertainty": "The source describes data centers generally. The AI connection is unresolved.",
        "evidence": [{"chunk_id": key, "span_id": spans[0]}],
    } for key, spans in mechanisms]
    refs = [f"s{i}" for i in range(1, len(drafts) + 1)]
    comparison = {
        "risk": "Data-center power and water constraints",
        "relationship": "same" if len(refs) == 2 else "unclear",
        "assessments": [{"ref": ref, "risk_category": "general_infrastructure", "entailment": "supports"} for ref in refs],
    }
    synthesis = {"cite_refs": refs, "comparisons": [comparison] if refs else []}
    finishes = [("finish_research", {"synthesis": synthesis})]
    if case == "failed_finish":
        finishes = [("truncated", {}), ("finish_research", {"synthesis": {"cite_refs": ["s8"]}})]
    child = NativeScript([_span_calls(_AMD_LIVE_READS), followup, ("submit_findings", {"findings": drafts}), *finishes])
    root = NativeScript([_assign(plan.groups[0]), ("finish_research", {"synthesis": {
        "cite_refs": [f"d{i}" for i in range(1, len(drafts) + 1)],
    }})])
    runner = _runner(corpus, lambda invocation: BedrockRootClient(
        child if invocation.depth else root, config=BedrockConfig(model="offline"),
    ), profile)
    result = execute_corpus_research(plan, runner, limits=_limits(1, 1, profile), discovery=spec, detailed_trace=True)
    capture = child.requests[2]
    assert capture["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_findings"}}
    assert capture["messages"][:len(child.requests[1]["messages"])] == child.requests[1]["messages"]
    assert "AI link, another year, or novelty is unresolved" in capture["system"][0]["text"]
    assert "no meaningful scoped observation" in json.dumps(capture["toolConfig"]["tools"])
    initial_prompt = child.requests[0]["messages"][0]["content"][0]["text"]
    assert "infrastructure_mechanism_hint" in initial_prompt
    for key, spans in _AMD_MECHANISMS:
        assert f'"infrastructure_mechanism_hint":{{"{key}":{json.dumps(list(spans))}}}' in initial_prompt
    for key, spans in _AMD_LIVE_READS:
        chunk = corpus.get_chunk(key)
        content = [block["toolResult"]["content"][0]["json"]["result"]
                   for message in capture["messages"] for block in message["content"]
                   if "toolResult" in block and block["toolResult"]["content"][0]["json"].get("ok")]
        for span in spans:
            assert any(record.get("spans", {}).get(span) == chunk.text[slice(*passage_spans(chunk.text)[span])]
                       and record.get("source", {}).get("chunk_id") == key for record in content)
    assert result.usage.accepted_submissions == len(drafts) and result.usage.rejected_submission_records == 0
    checkpoint = create_checkpoint(corpus, result)
    report = verification_report(checkpoint)
    assert report["assignment_evidence"][0]["retained_findings"] == len(drafts)
    assert report["assignment_evidence"][0]["novelty_unresolved"]
    if case == "abstain":
        assert result.execution_status == "finished" and result.research_status == "incomplete"
        assert report["assignment_evidence"][0]["execution"] == "completed"
        assert "Analysis unresolved. No findings were captured." in report["answer"]
        assert not plan_continuation(checkpoint).groups
        assert plan_continuation(checkpoint, ("AMD",)).groups == plan.groups
    elif case == "failed_finish":
        assert result.execution_status == "incomplete" and result.groups[0].pending_reason == "child_failed"
        assert len(result.citations) == 2 and report["continuation_companies"] == ("AMD",)
    else:
        item, = result.acknowledged_child_comparisons
        assert item.ai_specific_years == ()
        assert item.evidence_years == ((2024,) if case == "later_only" else (2023, 2024))
        assert item.outcome == ("insufficient" if case == "later_only" else "comparable_both_years")
        assert all(finding.uncertainty for finding in result.acknowledged_child_findings)


@pytest.mark.parametrize("have_hints", [True, False])
def test_quality_preserves_a_read_for_the_other_year_and_enforces_overall_budget(corpus, have_hints):
    plan = _plan(corpus, ("AMD",))
    spec = (DiscoverySpec.from_intent(plan.intent, PHRASES) if have_hints else
            DiscoverySpec.from_intent(plan.intent, ("no matching lexical hypothesis",), expand_context=False))
    first = _span_calls(_AMD_LIVE_READS[:3] + _AMD_MECHANISMS[1:])
    child = NativeScript([first, first, *_span_calls(_AMD_MECHANISMS[:1]), ("submit_findings", {"findings": []}),
                          ("finish_research", {"synthesis": {}})])
    root = NativeScript([_assign(plan.groups[0]), ("finish_research", {"synthesis": {}})])
    runner = _runner(corpus, lambda invocation: BedrockRootClient(child if invocation.depth else root,
                                                                 config=BedrockConfig(model="offline")), QUALITY)
    limits = _limits(1, 1, QUALITY)
    blocked = execute_corpus_research(plan, runner, limits=limits.model_copy(update={"max_provider_attempts": 21}), discovery=spec)
    assert not root.requests and blocked.stop_reason == "insufficient_provider_attempts"
    assert blocked.coverage.required_target_count == 2
    result = execute_corpus_research(plan, runner, limits=limits, discovery=spec)
    assert result.execution_status == "finished"
    assert bool(sum(len(row.candidates) for row in result.scan.sections)) is have_hints
    decisions = result.diagnostics[1]["decisions"]
    assert decisions[1]["operations"][-1]["error"] == "year_read_reserved"
    assert decisions[2]["operations"][0]["tool"] == "get_spans"
    assert decisions[3]["phase"] == "capture"
    assert result.usage.corpus_attempts == 9


def test_amazon_and_alphabet_positive_controls_and_outstanding_verification(corpus, tmp_path):
    from evidenceroute.research.rlm.controller import ScriptedModelClient

    evidence = {
        "AMZN": (
            ("b79aa2561327a953e141e2aa", ("16:200", "201:369")),
            ("4d1474550060251037c5f981", ("16:200", "201:369")),
        ),
        "GOOGL": (
            ("15787429383b3e9839f69716", ("608:820",)),
            ("62bad042e2e9d8334c2e6267", ()),
        ),
    }
    later = corpus.get_chunk(evidence["GOOGL"][1][0])
    span = next(key for key, offsets in passage_spans(later.text).items()
                if "may depend on the availability and pricing of third-party equipment" in later.text[slice(*offsets)])
    evidence["GOOGL"] = (evidence["GOOGL"][0], (later.chunk_id, (span,)))
    claims = {
        "AMZN": "Limited semiconductor suppliers and availability constraints could affect Amazon's AI operations.",
        "GOOGL": "Availability and pricing of third-party equipment can affect Alphabet's deployment of AI technologies.",
    }
    plan = _plan(corpus, ("AMZN", "GOOGL"))
    risk = "AI infrastructure supply dependence"

    def action(name, args):
        return {"type": "tool", "name": name, "arguments": args}

    def factory(invocation):
        if invocation.depth == 0:
            return ScriptedModelClient([*(action(*_assign(group)) for group in plan.groups),
                                        {"type": "synthesis", "synthesis": {"cite_refs": ["d1", "d2", "d3", "d4"]}}])
        ticker = corpus.get_filing(invocation.partitions[0].document_id).ticker
        entries = evidence[ticker]
        findings = [{"claim": claims[ticker], "analysis_status": "analyzed",
                     "evidence": [{"chunk_id": key, "span_id": ref} for ref in spans]} for key, spans in entries]
        comparison = {"risk": risk, "relationship": "same", "assessments": [
            {"ref": ref, "risk_category": "ai_infrastructure_dependency", "entailment": "supports", "ai_specificity": "explicit"}
            for ref in ("s1", "s2")
        ]}
        return ScriptedModelClient([*(action(*read) for read in _span_calls(entries)), action("submit_findings", {"findings": findings}),
                                    {"type": "synthesis", "synthesis": {"comparisons": [comparison]}}])

    spec = DiscoverySpec.from_intent(plan.intent, ("semiconductor products", "third party equipment", "AI infrastructure"))
    result = execute_corpus_research(plan, _runner(corpus, factory, QUALITY), limits=_limits(1, 2, QUALITY), discovery=spec)
    assert result.execution_status == "finished" and result.research_status == "incomplete"
    assert result.usage.accepted_submissions == 4
    assert {item.source.chunk_id for item in result.evidence} == {key for entries in evidence.values() for key, _ in entries}
    assert {comp.ticker: comp.ai_specific_years for comp in result.acknowledged_child_comparisons} == {
        "AMZN": (2023, 2024), "GOOGL": (2023, 2024),
    }
    checkpoint = create_checkpoint(corpus, result)
    reviewed_cases = []

    class VerifierTransport:
        def send(self, payload, *, timeout_seconds):
            cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
            reviews = []
            for case in cases:
                reviewed_cases.append((case["company"], case["kind"], tuple(e["year"] for e in case["selected_evidence"])))
                refs = [entry["ref"] for entry in case["selected_evidence"]]
                review = {"ref": case["ref"], "components": [
                    {"statement": claims[case["company"]], "status": "supported", "evidence": refs},
                ]}
                if case["kind"] == "comparison":
                    review["comparison"] = {"relationship": "same_risk", "relationship_evidence": refs,
                                            "earlier_year": "supports_risk", "earlier_evidence": [
                                                item["ref"] for item in case["selected_evidence"] if item["year"] == 2023
                                            ]}
                reviews.append(review)
            return {"stopReason": "tool_use", "usage": {"inputTokens": 500, "outputTokens": 250},
                    "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                        "toolUseId": "verified", "name": "submit_verification", "input": {"reviews": reviews},
                    }}]}}}

    def verify_pass(current, limits):
        verifier = BedrockSemanticVerifier(VerifierTransport(), config=BedrockConfig(model="offline"))
        return review_checkpoint(current, corpus, verifier, limits=limits)

    first = verify_pass(checkpoint, VerificationLimits(max_provider_attempts=1, max_items=2))
    assert verification_report(first)["outstanding_verification"] == {"findings": 4, "comparisons": 0}
    path = tmp_path / "quality-verified.json"
    save_checkpoint(path, first)
    final = verify_pass(load_checkpoint(path, corpus), VerificationLimits(max_provider_attempts=2, max_items=4))
    report = verification_report(final)
    assert report["outstanding_verification"] == {"findings": 0, "comparisons": 0}
    assert len(reviewed_cases) == len(set(reviewed_cases)) == 6
    assert {item["verified_relationship"] for item in report["conclusions"]} == {"same_risk"}
    assert {item["novelty"] for item in report["conclusions"]} == {"unresolved"}
    assert len(report["conclusion_citations"]) == 4
    assert verify_pass(final, VerificationLimits()) is final


def test_quality_resume_has_room_for_failed_initial_assignments(corpus, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from demo_rlm_comparative import DISCOVERY_PHRASES, _offline_factory

    plan = _plan(corpus)
    spec = DiscoverySpec.from_intent(plan.intent, DISCOVERY_PHRASES)
    initial = execute_corpus_research(
        plan, _runner(corpus, _offline_factory(corpus, plan, compact=True, fail_amd_finish=True), QUALITY),
        limits=_limits(3, 6, QUALITY), discovery=spec,
    )
    checkpoint = create_checkpoint(corpus, initial)
    assert initial.group_counts["completed"] == 5 and initial.group_counts["failed"] == 1
    assert len(verification_report(checkpoint)["continuation_companies"]) == 20

    runner = _runner(corpus, _offline_factory(corpus, plan, compact=True), QUALITY)
    final = continue_research(checkpoint, runner, limits=_limits(13, 25, QUALITY))
    assert final.execution_status == "finished" and final.research_status == "incomplete"
    assert final.merged_coverage(corpus).required_target_count == 50 and len(final.assignments) == 25
    assert len(final.runs[-1].target_companies) == final.current_usage.delegated_invocations == 20
    assert final.cumulative_usage.delegated_invocations == 26
    assert final.findings[:len(checkpoint.findings)] == checkpoint.findings
    assert final.current_usage.lexical_chunks_scanned == 0


# Verbatim checkpoint finding claims and cited spans from the live Alphabet investigator.
_LIVE_ALPHABET = (
    ("e0ec543a580b908a6602bba5", ("1504:1686",),
     "In FY2024, GOOGL disclosed that the availability of AI accelerators, including GPUs and custom-built TPUs, "
     "faces highly competitive and rapidly evolving industry supply capacity constraints."),
    ("62bad042e2e9d8334c2e6267", ("634:926",),
     "In FY2024, GOOGL disclosed that its ability to deploy AI technologies critical for products and services depends "
     "on the availability and pricing of third-party equipment, technical infrastructure operations costs including "
     "network capacity, energy, and equipment costs."),
    ("15787429383b3e9839f69716", ("608:820",),
     "In FY2023, GOOGL disclosed that its ability to deploy AI technologies critical for products and services may "
     "depend on the availability and pricing of third-party equipment and technical infrastructure."),
    ("e0ec543a580b908a6602bba5", ("417:578",),
     "In FY2024, GOOGL disclosed that supply interruptions affecting the company or its vendors could delay "
     "critical data center upgrades or expansions."),
)
_LIVE_AMAZON = (
    ("08d421523afa40cc084a4477", ("1605:1789",)), ("b79aa2561327a953e141e2aa", ("16:200",)),
    ("1852186e04d3918d176cac40", ("1605:1789",)), ("4d1474550060251037c5f981", ("16:200",)),
)


@pytest.mark.parametrize("pressure", [False, True])
def test_stabilized_live_alphabet_batch_and_amazon_capture(corpus, pressure):
    plan = _plan(corpus, ("AMZN", "GOOGL"))
    groups = {group.ticker: group for group in plan.groups}
    google_findings = [{"claim": claim, "analysis_status": "analyzed",
                        "evidence": [{"chunk_id": cid, "span_id": spans[0]}]}
                       for cid, spans, claim in _LIVE_ALPHABET]
    if pressure:
        for index in (0, 3):
            google_findings[index]["uncertainty"] = "Additional relevance remains unresolved. " * 9
            google_findings[index]["conflict"] = "Another interpretation requires evidence. " * 9
    google = NativeScript([_span_calls(tuple((cid, spans) for cid, spans, _ in _LIVE_ALPHABET)),
                           ("submit_findings", {"findings": google_findings}),
                           ("finish_research", {"synthesis": {}})])
    amazon_findings = [{
        "claim": "Amazon relies on a limited group of suppliers for semiconductors including GPUs for AI infrastructure.",
        "analysis_status": "analyzed", "evidence": [{"chunk_id": cid, "span_id": "16:200"}],
    } for cid in (_LIVE_AMAZON[1][0], _LIVE_AMAZON[3][0])]
    # The original Amazon response submitted an empty list. The scripted
    # positive control selects two directly supported atomic observations.
    followup = [("search_chunks", {"query": "semiconductor supplier dependence", "limit": 1})] * 4
    amazon = NativeScript([_span_calls(_LIVE_AMAZON), followup,
                           ("submit_findings", {"findings": amazon_findings}),
                           ("finish_research", {"synthesis": {"cite_refs": ["s1", "s2"]}})])
    root = NativeScript([[_assign(groups["GOOGL"]), _assign(groups["AMZN"])],
                         ("finish_research", {"synthesis": {"cite_refs": ["d2", "d3"]}})])

    def factory(invocation):
        transport = root
        if invocation.depth:
            ticker = corpus.get_filing(invocation.partitions[0].document_id).ticker
            transport = google if ticker == "GOOGL" else amazon
        return BedrockRootClient(transport, config=BedrockConfig(model="offline"))

    result = execute_corpus_research(plan, _runner(corpus, factory, QUALITY), limits=_limits(1, 2, QUALITY),
                                     discovery=DiscoverySpec.from_intent(plan.intent, PHRASES), detailed_trace=True)
    assert result.execution_status == "finished" and result.usage.accepted_submissions == 6
    step = result.diagnostics[0]["decisions"][0]
    alphabet, amzn = step["operations"]
    assert step["envelope_bytes"] <= 6000 and alphabet["observation_bytes"] <= 3000
    assert {item.source.fiscal_year for finding in result.root_selected_inherited_findings for item in finding.evidence} == {2023, 2024}
    assert set(alphabet["offered_finding_ids"]).isdisjoint(alphabet["omitted_finding_ids"])
    if pressure:
        assert alphabet["omitted_finding_ids"]
    assert set(amzn["acknowledged_finding_ids"]) == set(amzn["offered_finding_ids"])
    capture = amazon.requests[2]
    assert "Capture both years separately" in capture["system"][0]["text"]
    assert "Acknowledged chunk/span locators by year" in capture["system"][0]["text"]
    results = [block["toolResult"]["content"][0]["json"]["result"]
               for message in capture["messages"] for block in message["content"]
               if "toolResult" in block and block["toolResult"]["content"][0]["json"].get("ok")]
    for cid in (_LIVE_AMAZON[1][0], _LIVE_AMAZON[3][0]):
        assert any(entry.get("source", {}).get("chunk_id") == cid
                   and "Constraints on the availability" in entry.get("spans", {}).get("201:369", "")
                   for entry in results)
    checkpoint = create_checkpoint(corpus, result)
    refs = {proposal.ticker: proposal.finding_ids for proposal in _review_recovery(corpus, checkpoint).verification_runs[-1].proposals}
    assert refs["GOOGL"] == tuple(finding.finding_id for finding in checkpoint.findings
                                  if finding.claim in {_LIVE_ALPHABET[2][2], _LIVE_ALPHABET[1][2]})[::-1]


def _review_recovery(corpus, checkpoint):
    from evidenceroute.research.rlm.verification import recover_checkpoint

    class Verifier:
        def send(self, payload, *, timeout_seconds):
            cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
            reviews = []
            for case in cases:
                evidence = case["selected_evidence"]
                assert {item["year"] for item in evidence} == {2023, 2024}
                refs = [item["ref"] for item in evidence]
                earlier = [item["ref"] for item in evidence if item["year"] == 2023]
                if case["company"] == "GOOGL":
                    statement = "AI deployment may depend on availability and pricing of third-party equipment."
                    relationship = "more_specific_wording"
                else:
                    statement = "Amazon relies on a limited group of suppliers for semiconductor products."
                    relationship = "same_risk"
                reviews.append({"ref": case["ref"], "components": [
                    {"statement": statement, "status": "supported", "evidence": refs},
                ], "comparison": {"relationship": relationship, "relationship_evidence": refs,
                                  "earlier_year": "supports_risk", "earlier_evidence": earlier}})
            return {"stopReason": "tool_use", "usage": {"inputTokens": 500, "outputTokens": 250},
                    "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                        "toolUseId": "verified", "name": "submit_verification", "input": {"reviews": reviews},
                    }}]}}}

    return recover_checkpoint(checkpoint, corpus, BedrockSemanticVerifier(Verifier(), config=BedrockConfig(model="offline")),
                              limits=VerificationLimits(max_provider_attempts=2, max_items=4))


def test_comparison_retries_long_label_and_preserves_general_assessment(corpus):
    plan = _plan(corpus, ("AMD",))
    reads = _span_calls(_AMD_MECHANISMS)
    drafts = [{
        "claim": "Potential power or water shortages could limit customers' data-center expansion.",
        "analysis_status": "analyzed", "evidence": [{"chunk_id": key, "span_id": span[0]}],
    } for key, span in _AMD_MECHANISMS]
    assessments = [
        {"ref": "s1", "risk_category": "general_infrastructure", "entailment": "supports"},
        {"ref": "s2", "risk_category": "ai_infrastructure_dependency", "entailment": "supports",
         "ai_specificity": "contextual", "context_ref": "s1"},
    ]
    def finish(label):
        return ("finish_research", {"synthesis": {"cite_refs": ["s1", "s2"], "comparisons": [
            {"risk": label, "relationship": "more_specific", "assessments": assessments},
        ]}})
    child = NativeScript([reads * 2, reads[0], ("submit_findings", {"findings": drafts}),
                          finish("Data center risk " * 12), finish("Power and water availability")])
    root = NativeScript([_assign(plan.groups[0]), ("finish_research", {"synthesis": {"cite_refs": ["d1", "d2"]}})])
    runner = _runner(corpus, lambda invocation: BedrockRootClient(child if invocation.depth else root,
                                                                 config=BedrockConfig(model="offline")))
    result = execute_corpus_research(plan, runner, limits=_limits(1, 1),
                                     discovery=DiscoverySpec.from_intent(plan.intent, PHRASES))
    assert result.execution_status == "finished"
    assert "comparison_correction" in json.dumps(child.requests[-1]["messages"][-1])
    item, = result.acknowledged_child_comparisons
    assert item.relationship == "more_specific" and item.evidence_years == (2023, 2024)
    assert item.ai_specific_years == item.ai_contextual_years == ()
    assert {assessment.risk_category for assessment in item.assessments} == {"general_infrastructure"}


def test_one_year_verification_accepts_uncertainty_and_guards_modality(corpus):
    plan = _plan(corpus, ("GOOGL",))
    item = _LIVE_ALPHABET[1]
    child = NativeScript([*_span_calls(((item[0], item[1]),)),
                          ("submit_findings", {"findings": [{
                              "claim": item[2], "analysis_status": "analyzed",
                              "evidence": [{"chunk_id": item[0], "span_id": item[1][0]}],
                          }]}),
                          ("finish_research", {"synthesis": {"comparisons": [{
                              "risk": "Third-party infrastructure availability", "relationship": "unclear",
                              "assessments": [{"ref": "s1", "risk_category": "ai_infrastructure_dependency",
                                               "entailment": "supports", "ai_specificity": "explicit"}],
                          }]}})])
    root = NativeScript([_assign(plan.groups[0]), ("finish_research", {"synthesis": {"cite_refs": ["d1"]}})])
    runner = _runner(corpus, lambda invocation: BedrockRootClient(child if invocation.depth else root,
                                                                 config=BedrockConfig(model="offline")), QUALITY)
    result = execute_corpus_research(plan, runner, limits=_limits(1, 1, QUALITY),
                                     discovery=DiscoverySpec.from_intent(plan.intent, PHRASES))
    checkpoint = create_checkpoint(corpus, result)
    response = {"reviews": [
        {"ref": "v1", "components": [{"statement": "Deployment depends on equipment availability.",
                                     "status": "supported", "evidence": ["e1"]},
                                    {"statement": "Availability and pricing may affect AI deployment.",
                                     "status": "supported", "evidence": ["e1"]}]},
        {"ref": "v2", "components": [{"statement": "Earlier evidence could not be assessed.",
                                     "status": "insufficient_evidence", "evidence": [], "gap": "Find FY2023."}],
         "comparison": {"relationship": "unclear", "earlier_year": "insufficient"}},
    ]}
    from evidenceroute.research.rlm.verification import record_verification, verify_checkpoint
    verifier = BedrockSemanticVerifier(NativeScript([("submit_verification", response)]), config=BedrockConfig(model="offline"))
    assessment = verify_checkpoint(checkpoint, corpus, verifier, limits=VerificationLimits())
    report = verification_report(record_verification(checkpoint, corpus, assessment))
    items = {item.kind: item for item in report["items"]}
    assert items["finding"].status == "partially_supported"
    assert items["comparison"].status == "insufficient_evidence"
    assert items["comparison"].comparison.earlier_year == "insufficient"
    assert "may depend" in report["answer"] and "Deployment depends" not in report["answer"]
