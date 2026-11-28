"""Verify the question entry point through mocked Bedrock Converse responses."""

import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.claude import ProviderTransportError
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits
from evidenceroute.research.rlm.interpretation import BedrockIntentInterpreter, InterpretationLimits, ValidationIssue
from evidenceroute.research.rlm.natural_language import research_question
from evidenceroute.research.rlm.recursive import RecursiveResearch


QUESTION = "  What foreign exchange risks does Apple disclose in FY2024?  "
CHUNK_ID = "cd7988623c9a3bf545f06045"
QUOTE = "The weakening of foreign currencies relative to the U.S. dollar adversely affects the U.S. dollar value"


def _structured(*, mode="review", phrases=None, **overrides):
    return {
        "mode": mode, "explanation": None,
        "discovery_phrases": ["foreign currency", "foreign exchange"] if phrases is None else phrases,
        "requirements": {
            "company_selection": "explicit", "companies": ["Apple Inc."], "fiscal_years": [2024],
            "section_selection": "risk_factors_default", "section_ids": [],
            "objective": "review", "topic": "Foreign exchange exposure",
            "comprehensive": False, "inferred_fields": [], "scope_notes": None,
            **overrides,
        },
    }


def _blocked(mode, explanation):
    return {"mode": mode, "requirements": None, "discovery_phrases": None, "explanation": explanation}


def _reply(name, arguments, *, stop="tool_use", usage=True):
    response = {
        "stopReason": stop, "metrics": {"latencyMs": 80},
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"reasoningText": {"text": "private provider content"}}},
            {"toolUse": {"toolUseId": "test.tool:1", "name": name, "input": arguments}},
        ]}},
    }
    if usage:
        response["usage"] = {"inputTokens": 100, "outputTokens": 20, "cacheReadInputTokens": 2,
                             "cacheWriteInputTokens": 3}
    return response


def _interpretation(data, **kwargs):
    return _reply("interpret_research_question", data, **kwargs)


class MockTransport:
    def __init__(self, replies, *, clock=None, latency=0):
        self.replies = deque(replies)
        self.requests = []
        self.clock = clock
        self.latency = latency

    def send(self, payload, *, timeout_seconds):
        self.requests.append((payload, timeout_seconds))
        if self.clock is not None:
            self.clock.advance(self.latency)
        response = self.replies.popleft()
        if isinstance(response, Exception):
            raise response
        return response


class Clock:
    seconds = 0.0

    def __call__(self):
        return self.seconds

    def advance(self, seconds):
        self.seconds += seconds


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


def _limits(**updates):
    return CorpusLimits(**{
        "max_batches": 1, "max_provider_attempts": 9, "max_reported_tokens": 30_000, "max_corpus_attempts": 8,
        "max_delegations": 1, "max_partitions": 4, "deadline_seconds": 90.0, **updates,
    })


SHARED = SharedLimits(max_provider_attempts=8, max_reported_tokens=10_000, max_corpus_attempts=8,
                      max_delegations=1, max_partitions=4, max_depth=1)
CONFIG = BedrockConfig(model="offline", max_retries=1)


def _run(corpus, transport, *, question=QUESTION, research_factory=None, clock=None, interpretation_limits=None,
         interpreter_factory=None, shared_limits=None, **limits):
    def unexpected(_):
        pytest.fail("Research must be blocked by scope, status, or overall limits.")

    clock = clock if clock is not None else Clock()
    runner = RecursiveResearch(corpus, research_factory or unexpected, shared_limits=shared_limits or SHARED, clock=clock)
    return research_question(
        question, runner, limits=_limits(**limits), interpretation_limits=interpretation_limits,
        interpreter_factory=interpreter_factory or (
            lambda: BedrockIntentInterpreter(transport, config=CONFIG, clock=clock, wait=clock.advance)
        ),
    )


def test_inventory_and_original_question_reach_one_forced_structured_request(corpus):
    data = _structured(companies=["GOOGL", "Alphabet Inc."], topic="Data center risk",
                       phrases=["data center", "technical infrastructure"])
    transport = MockTransport([_interpretation(data)])
    result = _run(corpus, transport, question="What data center risk did Google disclose in FY2024?", max_batches=0)
    outcome = result.interpretation.outcome
    assert outcome.status == "ready"
    assert result.plan.requested_companies == ("GOOGL",)
    assert result.plan.required_target_count == 1
    assert outcome.intent.section_ids == ("item_1a",)
    assert outcome.intent.inferred_fields == ("section_ids",)
    assert outcome.intent.scope_notes == "No filing section was specified. This plan is limited to Item 1A (Risk Factors)."
    assert outcome.discovery.phrases == ("data center", "technical infrastructure")
    assert len(transport.requests) == 1
    payload, _ = transport.requests[0]
    assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "interpret_research_question"}}
    assert len(payload["toolConfig"]["tools"]) == 1
    schema = payload["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    requirements = schema["$defs"]["_Requirements"]
    assert "section_selection" in requirements["required"]
    assert requirements["properties"]["section_selection"]["enum"] == ["explicit", "risk_factors_default", "inferred"]
    assert schema["properties"]["mode"]["enum"] == [
        "review", "generic_synthesis", "ai_infrastructure_comparison", "ambiguous", "unsupported",
    ]
    assert "section_ids" not in requirements["properties"]["inferred_fields"]["items"]["enum"]
    assert requirements["properties"]["inferred_fields"]["items"]["enum"] == [
        "company_selection", "fiscal_years", "objective", "topic", "comprehensive",
    ]
    assert "explicit" in requirements["properties"]["section_selection"]["description"]
    assert "inferred" in requirements["properties"]["section_selection"]["description"]
    submitted = json.loads(payload["messages"][0]["content"][0]["text"])
    assert submitted["question"] == result.question == result.plan.intent.question
    inventory = submitted["corpus_inventory"]
    assert {"ticker": "GOOGL", "names": ["Alphabet Inc."], "fiscal_years": [2022, 2023, 2024]} in inventory["companies"]
    assert len(inventory["companies"]) == 25
    assert {"id": "item_1a", "title": "Risk Factors"} in inventory["sections"]
    assert inventory["latest_fiscal_year"] == 2024
    assert "private provider content" not in str(result.interpretation)


@pytest.mark.parametrize("explicit", [False, True])
def test_corpus_wide_scope_uses_latest_snapshot_year(corpus, explicit):
    inferred = ["fiscal_years"]
    if not explicit:
        inferred.extend(["company_selection", "comprehensive"])
    data = _structured(company_selection="all", companies=[], comprehensive=True,
                       inferred_fields=inferred, scope_notes="Use the latest snapshot year.")
    question = "List every company in this corpus with foreign exchange risk." if explicit else (
        "Which companies discuss foreign exchange risk?"
    )
    result = _run(corpus, MockTransport([_interpretation(data)]),
                  question=question, max_batches=0)
    assert result.plan.intent.company_selection == "all"
    assert result.plan.intent.comprehensive
    assert result.plan.intent.fiscal_years == (2024,)
    assert result.plan.intent.inferred_fields == (*inferred, "section_ids")
    assert result.plan.intent.scope_notes.startswith("Use the latest snapshot year.")
    assert ("company_selection" in result.plan.intent.inferred_fields) is not explicit
    assert len(result.plan.requested_companies) == result.plan.required_target_count == 25
    assert result.research.research_status == "incomplete"


def test_explicit_unavailable_scope_remains_in_plan(corpus):
    data = _structured(companies=["Apple Inc.", "Unlisted Corp."], fiscal_years=[2024, 2025],
                       section_selection="explicit", section_ids=["item_1a", "item_9a"])
    result = _run(corpus, MockTransport([_interpretation(data)]), question=(
        "Review foreign exchange disclosures by Apple and Unlisted Corp. in FY2024 and FY2025 Items 1A and 9A."
    ))
    plan = result.plan
    assert plan.required_target_count == 8 and len(plan.resolved_targets) == 1
    assert {entry.reason for entry in plan.missing_requirements} == {"missing_section", "missing_filing", "unknown_company"}
    assert {entry.fiscal_year for entry in plan.missing_requirements if entry.reason == "missing_filing"} == {2025}
    assert {entry.section_id for entry in plan.missing_requirements if entry.reason == "missing_section"} == {"item_9a"}
    assert all(group.pending_reason == "unresolved_requirements" for group in result.research.groups)
    assert result.research.research_status == "incomplete"


def test_explicit_multiple_sections_are_preserved(corpus):
    data = _structured(section_selection="explicit", section_ids=["item_7", "item_8"])
    result = _run(corpus, MockTransport([_interpretation(data)]), max_batches=0,
                  question="What foreign exchange risks did Apple disclose in FY2024 Items 7 and 8?")
    assert result.plan.intent.section_ids == ("item_7", "item_8")
    assert result.plan.intent.inferred_fields == () and result.plan.intent.scope_notes is None
    assert {target.section_id for target in result.plan.resolved_targets} == {"item_7", "item_8"}
    assert result.plan.missing_requirements == ()


def test_explicit_item_7a_remains_unavailable(corpus):
    data = _structured(section_selection="explicit", section_ids=["item_7a"])
    result = _run(corpus, MockTransport([_interpretation(data)]),
                  question="What foreign exchange risks did Apple disclose in FY2024 Item 7A?")
    assert result.plan.intent.section_ids == ("item_7a",)
    assert result.plan.intent.inferred_fields == ()
    assert result.plan.resolved_targets == ()
    missing, = result.plan.missing_requirements
    assert (missing.company, missing.fiscal_year, missing.section_id, missing.reason) == (
        "AAPL", 2024, "item_7a", "missing_section",
    )
    assert result.research.groups[0].pending_reason == "unresolved_requirements"
    assert result.research.batches == ()


@pytest.mark.parametrize(("data", "question", "status", "reason"), [
    (_blocked("unsupported", "The comparison concerns foreign exchange risk."),
     "Compare Apple's FY2023 and FY2024 foreign exchange risk.", "unsupported", "unsupported"),
    (_blocked("ambiguous", "Which company and fiscal years should be compared?"),
     "How did the company's risk change recently?", "ambiguous", "ambiguous"),
    (_blocked("ambiguous", "Which filing section should be reviewed?"),
     "What did Apple disclose in FY2024?", "ambiguous", "ambiguous"),
    (_structured(mode="ai_infrastructure_comparison", objective="year_over_year", fiscal_years=[2022, 2023, 2024]),
     "Compare AI infrastructure risks in 2022, 2023 and 2024.", "unsupported", "unsupported_comparison_years"),
])
def test_blocked_questions_have_explicit_outcomes(corpus, data, question, status, reason):
    transport = MockTransport([_interpretation(data)])
    result = _run(corpus, transport, question=question)
    assert (result.interpretation.outcome.status, result.interpretation.outcome.stop_reason) == (status, reason)
    assert result.plan is result.research is None
    assert result.usage.provider_attempts == len(transport.requests) == 1
    assert result.interpretation.correction_attempt is None


@pytest.mark.parametrize(("data", "issue"), [
    (_structured(fiscal_years=["2024"]), ValidationIssue("requirements.fiscal_years[0]", "schema", "int_type")),
    (_structured(comprehensive=1), ValidationIssue("requirements.comprehensive", "schema", "bool_type")),
    (_structured(extra_field="private provider content"), ValidationIssue("requirements", "schema", "extra_forbidden")),
    (_structured(fiscal_years=[2023], inferred_fields=["fiscal_years"], scope_notes="Use the latest snapshot year."),
     ValidationIssue("requirements.fiscal_years", "inconsistent_requirements", "invalid_inferred_years")),
    (_structured(inferred_fields=["topic"], scope_notes=None),
     ValidationIssue("requirements.scope_notes", "inconsistent_requirements", "scope_notes_required")),
    (_structured(inferred_fields=["segments"], scope_notes="Inferred section."),
     ValidationIssue("requirements.inferred_fields[0]", "schema", "literal_error",
                     "company_selection, fiscal_years, objective, topic, comprehensive")),
    (_structured(section_selection="explicit"),
     ValidationIssue("requirements.section_ids", "inconsistent_requirements", "explicit_sections_required")),
    (_structured(companies=[]),
     ValidationIssue("requirements.companies", "inconsistent_requirements", "company_selection_mismatch")),
    (_structured(mode="ai_infrastructure_comparison", objective="year_over_year", fiscal_years=[2024]),
     ValidationIssue("requirements.fiscal_years", "inconsistent_requirements", "comparison_years_required")),
    (_structured(objective="year_over_year", fiscal_years=[2023, 2024]),
     ValidationIssue("requirements.objective", "inconsistent_requirements", "objective_mode_mismatch")),
    (_structured(phrases=[]), ValidationIssue("discovery_phrases", "discovery_phrase", "too_short")),
    (_structured(phrases=["..."]), ValidationIssue("discovery_phrases", "discovery_phrase", "non_lexical_phrase")),
    (_structured(phrases=["risk"] * 33), ValidationIssue("discovery_phrases", "discovery_phrase", "too_long")),
    (_structured(phrases=["x" * 201]), ValidationIssue("discovery_phrases[0]", "discovery_phrase", "string_too_long")),
    ({**_structured(), "discovery_phrases": None},
     ValidationIssue("discovery_phrases", "discovery_phrase", "value_required_for_mode")),
    ({"mode": "review"}, ValidationIssue("requirements", "missing_field", "missing")),
    (_structured(comprehensive=float("nan")), ValidationIssue("submission", "schema", "json_invalid")),
])
def test_invalid_structured_requirements_and_phrases_fail_closed(corpus, data, issue):
    transport = MockTransport([_interpretation(data)])
    result = _run(corpus, transport, max_provider_attempts=1)
    assert result.interpretation.outcome.stop_reason == "provider_attempt_limit"
    assert result.interpretation.attempts[0].error == "invalid_interpretation"
    assert result.usage.provider_attempts == len(transport.requests) == 1
    assert result.interpretation.correction_attempt is None
    failure, = result.interpretation.validation_failures
    assert failure.attempt == 1 and issue in failure.issues
    assert result.research is None
    assert "private provider content" not in str(result.interpretation)


def test_correction_uses_fresh_native_request_and_bounded_sanitized_feedback(corpus):
    first = _structured(**{"model-generated secret key": "private model-generated value"})
    first["requirements"].pop("scope_notes")
    first["private extra field"] = "private top-level value"
    first["discovery_phrases"] = [0, 1, 2, 3, 4, 5]
    corrected = _structured(companies=["AAPL"], phrases=["foreign currency", "exchange rate fluctuations"])
    transport = MockTransport([_interpretation(first), _interpretation(corrected)])
    result = _run(corpus, transport, max_batches=0)
    assert result.plan.requested_companies == ("AAPL",)
    assert result.plan.intent.fiscal_years == (2024,)
    assert not result.plan.intent.comprehensive
    assert result.interpretation.outcome.discovery.phrases == ("foreign currency", "exchange rate fluctuations")
    assert result.interpretation.correction_attempt == 2
    failure, = result.interpretation.validation_failures
    assert failure.attempt == 1 and len(failure.issues) == 4
    assert ValidationIssue("requirements.scope_notes", "missing_field", "missing") in failure.issues
    original, correction = (request for request, _ in transport.requests)
    assert original["inferenceConfig"] == correction["inferenceConfig"]
    assert original["toolConfig"] == correction["toolConfig"]
    assert [message["role"] for message in correction["messages"]] == ["user"]
    first_input = json.loads(original["messages"][0]["content"][0]["text"])
    second_input = json.loads(correction["messages"][0]["content"][0]["text"])
    assert second_input.pop("validation_feedback") == [asdict(issue) for issue in failure.issues]
    assert first_input == second_input
    for value in (correction, result.interpretation):
        assert all(secret not in str(value) for secret in (
            "private provider content", "test.tool:1", "model-generated secret key", "private model-generated value",
            "private extra field", "private top-level value",
        ))
    assert result.interpretation.reported_tokens == result.interpretation.charged_tokens == 250
    assert result.usage.provider_attempts == 2 and result.usage.reported_tokens == 250
    assert result.research.limits.max_provider_attempts == 7
    assert result.research.limits.max_reported_tokens == 29_750


def test_correction_clears_generated_ids_from_risk_default(corpus):
    # Representative malformed default. The live report did not record which IDs were submitted.
    first = _structured(section_ids=["item_7a"])
    transport = MockTransport([_interpretation(first), _interpretation(_structured())])
    result = _run(corpus, transport, max_batches=0,
                  question="What foreign exchange risks did Apple disclose in its FY2024 annual report?")
    report = result.interpretation
    assert report.correction_attempt == 2
    failure, = report.validation_failures
    assert failure.issues == (
        ValidationIssue("requirements.section_ids", "inconsistent_requirements", "default_section_ids_must_be_empty"),
    )
    system = transport.requests[1][0]["system"][0]["text"]
    assert 'use "risk_factors_default" with section_ids []' in system
    assert 'For expressly named sections, use "explicit" with every requested ID' in system
    assert "Leave section_ids out of inferred_fields" in system
    assert result.plan.intent.section_ids == ("item_1a",)
    assert result.plan.intent.inferred_fields == ("section_ids",)
    assert result.plan.missing_requirements == ()
    assert result.usage.provider_attempts == 2 and result.usage.reported_tokens == 250


def test_redundant_section_metadata_keeps_a_valid_plan(corpus):
    data = _structured(inferred_fields=["section_ids", "Section IDs", "sections"], scope_notes="Risk factors.")
    result = _run(corpus, MockTransport([_interpretation(data)]), max_batches=0,
                  question="What foreign exchange risks did Apple disclose in its FY2024 annual report?")
    assert result.interpretation.outcome.status == "ready"
    assert result.interpretation.validation_failures == ()
    assert result.plan.intent.section_ids == ("item_1a",)
    assert result.plan.intent.inferred_fields == ("section_ids",)


def test_literal_error_feedback_lists_permitted_values(corpus):
    bad = _structured(inferred_fields=["section_ids", "segments"], scope_notes="Risk factors.")
    transport = MockTransport([_interpretation(bad), _interpretation(_structured())])
    result = _run(corpus, transport, max_batches=0,
                  question="What foreign exchange risks did Apple disclose in its FY2024 annual report?")
    issue, = result.interpretation.validation_failures[0].issues
    assert issue == ValidationIssue(
        "requirements.inferred_fields[0]", "schema", "literal_error",
        "company_selection, fiscal_years, objective, topic, comprehensive",
    )
    feedback = json.loads(transport.requests[1][0]["messages"][0]["content"][0]["text"])["validation_feedback"]
    assert feedback[0]["permitted"] == issue.permitted
    assert "Leave section_ids out of inferred_fields" in transport.requests[1][0]["system"][0]["text"]
    assert result.plan.intent.section_ids == ("item_1a",)


def test_invalid_correction_stops_after_one_correction_even_with_a_spare_attempt(corpus):
    transport = MockTransport([_interpretation(_structured(phrases=["..."])), _interpretation({"mode": "review"})])
    result = _run(corpus, transport, interpretation_limits=InterpretationLimits(max_provider_attempts=3))
    report = result.interpretation
    assert report.outcome.stop_reason == "invalid_interpretation"
    assert report.correction_attempt == 2
    assert [item.attempt for item in report.validation_failures] == [1, 2]
    assert report.validation_failures[0].issues[0].code == "non_lexical_phrase"
    assert report.validation_failures[1].issues[0].code == "missing"
    assert len(transport.requests) == result.usage.provider_attempts == 2 and result.plan is result.research is None


@pytest.mark.parametrize("steps", [2, 3])
def test_initial_transport_retry_and_correction_share_attempt_limits(corpus, steps):
    retry = ProviderTransportError("aws_unavailable", retryable=True, usage={"inputTokens": 21, "outputTokens": 0})
    transport = MockTransport([retry, _interpretation(_structured(phrases=["!"])), _interpretation(_structured())])
    result = _run(corpus, transport, max_batches=0, interpretation_limits=InterpretationLimits(max_provider_attempts=steps))
    report = result.interpretation
    assert report.validation_failures[0].attempt == 2
    assert report.attempts[0].error == "aws_unavailable"
    assert len(transport.requests) == result.usage.provider_attempts == steps
    if steps == 2:
        assert report.outcome.stop_reason == "provider_attempt_limit"
        assert report.correction_attempt is None and result.research is None
    else:
        assert report.outcome.status == "ready"
        assert report.correction_attempt == 3 and report.charged_tokens == 271
        assert result.research.limits.max_provider_attempts == 6


def test_correction_does_not_retry_transport_failure(corpus):
    retryable = ProviderTransportError("aws_throttled", retryable=True)
    transport = MockTransport([_interpretation(_structured(phrases=["!"])), retryable, _interpretation(_structured())])
    result = _run(corpus, transport, interpretation_limits=InterpretationLimits(max_provider_attempts=3))
    report = result.interpretation
    assert report.outcome.stop_reason == "aws_throttled"
    assert report.correction_attempt == 2 and len(transport.requests) == 2
    assert report.attempts_with_missing_usage == 1
    assert report.charged_tokens == 125 + report.attempts[1].estimated_tokens
    assert result.research is None


def test_correction_is_blocked_when_remaining_tokens_cannot_cover_its_estimate(corpus):
    invalid = _interpretation(_structured(phrases=["..."]))
    invalid["usage"]["inputTokens"] = 4_500
    transport = MockTransport([invalid])
    result = _run(corpus, transport, interpretation_limits=InterpretationLimits(max_tokens=7_000))
    report = result.interpretation
    assert report.outcome.stop_reason == "interpretation_token_limit"
    assert report.validation_failures[0].issues[0].code == "non_lexical_phrase"
    assert report.charged_tokens == 4_525 and report.correction_attempt is None
    assert len(transport.requests) == 1 and result.research is None


def test_correction_shares_deadline_and_rejects_late_response(corpus):
    clock = Clock()
    transport = MockTransport([_interpretation(_structured(phrases=["!"])), _interpretation(_structured())],
                              clock=clock, latency=0.6)
    result = _run(corpus, transport, clock=clock, deadline_seconds=1.0)
    report = result.interpretation
    assert report.outcome.stop_reason == "deadline"
    assert report.correction_attempt == 2 and report.attempts[1].error == "deadline"
    assert [timeout for _, timeout in transport.requests] == pytest.approx([1.0, 0.4])
    assert result.usage.reported_tokens == 250 and result.usage.elapsed_seconds == 1.2
    assert result.research is None


@pytest.mark.parametrize(("field", "change"), [
    ("companies", ["AMZN"]),
    ("fiscal_years", [2022, 2024]),
    ("section_ids", ["item_7"]),
    ("objective", "review"),
])
def test_correction_cannot_narrow_or_substitute_previously_valid_explicit_scope(corpus, field, change):
    question = "Did Amazon and Unlisted Corp. disclose a new AI infrastructure risk in FY2024 versus FY2023 Item 1A?"
    scope = {"companies": ["AMZN", "Unlisted Corp."], "fiscal_years": [2023, 2024],
             "section_selection": "explicit", "section_ids": ["item_1a"],
             "objective": "year_over_year", "inferred_fields": [], "scope_notes": None}
    first = _structured(mode="ai_infrastructure_comparison", **scope)
    mode = "review" if field == "objective" else "ai_infrastructure_comparison"
    corrected = _structured(mode=mode, comprehensive=True, **(scope | {field: change}))
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(corrected)]), question=question)
    report = result.interpretation
    assert report.outcome.stop_reason == "invalid_interpretation"
    assert report.validation_failures[0].issues[0].code == "comprehensive_required_for_comparison"
    assert report.validation_failures[1].issues == (
        ValidationIssue(f"requirements.{field}", "inconsistent_requirements", "correction_changed_scope"),
    )
    assert result.plan is result.research is None


@pytest.mark.parametrize("first_selection", ["explicit", None])
@pytest.mark.parametrize("section_ids", [["item_1a"], ["item_7a"]])
def test_correction_cannot_replace_explicit_section_with_default(corpus, first_selection, section_ids):
    first = _structured(section_selection="explicit", section_ids=section_ids, phrases=[])
    if first_selection is None:
        first["requirements"].pop("section_selection")
    transport = MockTransport([_interpretation(first), _interpretation(_structured())])
    result = _run(corpus, transport, question=f"What foreign exchange risks did Apple disclose in FY2024 {section_ids[0]}?")
    report = result.interpretation
    assert report.correction_attempt == 2
    assert report.validation_failures[1].issues == (
        ValidationIssue("requirements.section_selection", "inconsistent_requirements", "correction_changed_scope"),
    )
    assert result.plan is result.research is None


def test_inconsistent_inferred_default_can_be_repaired(corpus):
    first = _structured(fiscal_years=[2023], inferred_fields=["fiscal_years"], scope_notes="Use the latest snapshot year.")
    corrected = _structured(fiscal_years=[2024], inferred_fields=["fiscal_years"],
                            scope_notes="Use the latest snapshot year.")
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(corrected)]), max_batches=0,
                  question="What foreign exchange risks does Apple disclose?")
    assert result.plan.intent.fiscal_years == (2024,)
    assert result.interpretation.validation_failures[0].issues[0].code == "invalid_inferred_years"


def test_correction_cannot_downgrade_an_incomplete_comparison_envelope(corpus):
    first = {"mode": "ai_infrastructure_comparison"}
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(_structured())]),
                  question="Did Apple introduce any new AI infrastructure risks in FY2024 compared with FY2023?")
    assert result.plan is result.research is None
    assert result.interpretation.validation_failures[1].issues == (
        ValidationIssue("requirements.objective", "inconsistent_requirements", "correction_changed_scope"),
    )


@pytest.mark.parametrize(("reply", "failure"), [
    (_interpretation(_structured(), stop="max_tokens"), "truncated_response"),
    (_reply("unexpected_tool", _structured()), "invalid_interpretation_tool_use"),
    (_interpretation(_structured(), stop="end_turn"), "missing_structured_interpretation"),
    (ProviderTransportError("aws_access_denied"), "aws_access_denied"),
])
def test_protocol_and_provider_failures_are_metered(corpus, reply, failure):
    result = _run(corpus, MockTransport([reply]))
    assert result.interpretation.outcome.stop_reason == failure
    assert result.interpretation.attempts[0].error == failure
    assert result.usage.provider_attempts == 1 and result.research is None


def test_unavailable_provider_setup_and_zero_allowance_do_not_call_provider(corpus):
    def unavailable():
        raise ValueError("private configuration")

    result = _run(corpus, None, interpreter_factory=unavailable)
    assert result.interpretation.outcome.stop_reason == "interpretation_unavailable"
    assert result.usage.provider_attempts == 0
    exhausted = _run(corpus, None, max_provider_attempts=0)
    assert exhausted.interpretation.outcome.stop_reason == "provider_attempt_limit"
    assert not exhausted.interpretation.attempts
    low_tokens = _run(corpus, MockTransport([]), max_reported_tokens=10)
    assert low_tokens.interpretation.outcome.stop_reason == "interpretation_token_limit"
    assert not low_tokens.interpretation.attempts


def test_retry_and_missing_usage_charges_are_settled_before_tree_admission(corpus):
    clock = Clock()
    retry = ProviderTransportError("aws_throttled", retryable=True, usage={"inputTokens": 21, "outputTokens": 0})
    transport = MockTransport([retry, _interpretation(_structured(), usage=False)], clock=clock, latency=0.2)
    result = _run(corpus, transport, clock=clock, max_provider_attempts=10, max_reported_tokens=10_500,
                  deadline_seconds=4.0, interpretation_limits=InterpretationLimits(max_output_tokens=512))
    report = result.interpretation
    first, second = report.attempts
    assert first.error == "aws_throttled" and first.provider_stop_reason is None
    assert second.error is None and second.provider_stop_reason == "tool_use"
    assert first.latency_seconds == second.latency_seconds == pytest.approx(0.2)
    assert report.elapsed_seconds == pytest.approx(0.65)
    assert second.provider_latency_seconds == 0.08
    assert first.output_cap == second.output_cap == 512
    assert transport.requests[0][1] == 4.0 and transport.requests[1][1] == pytest.approx(3.55)
    assert report.reported_tokens == result.usage.reported_tokens == 21
    assert report.charged_tokens == 21 + second.estimated_tokens
    assert result.usage.estimated_tokens == first.estimated_tokens + second.estimated_tokens
    assert result.usage.attempts_with_missing_usage == 1
    assert result.usage.provider_attempts == 2
    assert result.research.limits.deadline_seconds == pytest.approx(3.35)
    assert result.research.limits.max_reported_tokens == 10_500 - report.charged_tokens
    assert result.research.stop_reason == "insufficient_reported_tokens" and not result.research.batches


def test_retry_is_rejected_by_attempt_and_token_caps(corpus):
    unavailable = ProviderTransportError("aws_unavailable", retryable=True)
    result = _run(corpus, MockTransport([unavailable]), max_provider_attempts=1)
    assert result.interpretation.outcome.stop_reason == "provider_attempt_limit"
    assert result.interpretation.attempts[0].error == "aws_unavailable"
    assert result.usage.provider_attempts == 1
    cost = result.interpretation.attempts[0].estimated_tokens
    result = _run(corpus, MockTransport([unavailable]), interpretation_limits=InterpretationLimits(max_tokens=cost + 1))
    assert result.interpretation.outcome.stop_reason == "interpretation_token_limit"
    assert result.interpretation.charged_tokens == cost
    assert result.usage.provider_attempts == 1


def test_late_and_over_budget_replies_cannot_start_research(corpus):
    clock = Clock()
    transport = MockTransport([_interpretation(_structured())], clock=clock, latency=1.2)
    late = _run(corpus, transport, clock=clock, deadline_seconds=1.0)
    assert late.interpretation.outcome.stop_reason == late.interpretation.attempts[0].error == "deadline"
    assert late.interpretation.attempts[0].provider_stop_reason == "tool_use"
    assert late.usage.elapsed_seconds == 1.2 and late.research is None
    reply = _interpretation(_structured())
    reply["usage"]["inputTokens"] = 17_000
    over = _run(corpus, MockTransport([reply]))
    assert over.interpretation.outcome.stop_reason == "interpretation_token_limit"
    assert over.usage.reported_tokens == 17_025 and over.research is None


def test_interpretation_consumes_tree_attempt_reservation(corpus):
    result = _run(corpus, MockTransport([_interpretation(_structured())]), max_provider_attempts=8)
    assert result.research.stop_reason == "insufficient_provider_attempts"
    assert result.research.limits.max_provider_attempts == 7
    assert result.usage.provider_attempts == 1 and not result.research.batches


@pytest.mark.parametrize("needs_correction", [False, True])
def test_review_plan_executes_with_scoped_discovery_and_original_evidence(corpus, needs_correction):
    clock = Clock()
    chunk = corpus.get_chunk(CHUNK_ID)
    outside = corpus.get_chunk("4d1474550060251037c5f981")
    transports = {}

    def factory(invocation):
        finish = {"answer": "One scoped finding.", "analysis_status": "incomplete", "cited_chunk_ids": [CHUNK_ID]}
        if invocation.depth:
            outside_candidates = _reply("list_candidates", {"document_id": outside.document_id,
                                                            "section_id": outside.section_id})
            candidates = _reply("list_candidates", {"document_id": chunk.document_id, "section_id": chunk.section_id})
            early_finish = _reply("finish_research", {"answer": "Only candidate IDs.", "cited_chunk_ids": [CHUNK_ID]})
            read = _reply("get_chunk", {"chunk_id": CHUNK_ID})
            finish["findings"] = [{"claim": "A stronger dollar affects translated sales and earnings.",
                                   "analysis_status": "analyzed", "evidence": [{"chunk_id": CHUNK_ID, "quote": QUOTE}]}]
            replies = [outside_candidates, candidates, early_finish, read, _reply("finish_research", finish)]
        else:
            assert QUESTION in invocation.question
            assert "Foreign exchange exposure" in invocation.question
            assign = _reply("delegate_research", {"question": "Examine exchange rates.", "goal": "Review risk.",
                                                  "partitions": [asdict(pair) for pair in invocation.partitions]})
            finish["used_finding_ids"] = [f"{invocation.invocation_id.removesuffix('root')}child-1:finding-1"]
            replies = [assign, _reply("finish_research", finish)]
        for number, response in enumerate(replies, 1):
            response["output"]["message"]["content"][-1]["toolUse"]["toolUseId"] += f":{number}"
        transport = transports[invocation.depth] = MockTransport(replies, clock=clock, latency=0.1)
        return BedrockRootClient(transport, config=CONFIG, clock=clock)

    replies = [_interpretation(_structured())]
    if needs_correction:
        replies.insert(0, _interpretation(_structured(section_ids=["item_7a"])))
    result = _run(corpus, MockTransport(replies), research_factory=factory, clock=clock, max_provider_attempts=10)
    child_first = transports[1].requests[0][0]
    child_prompt = child_first["messages"][0]["content"][0]["text"]
    assert QUESTION in child_prompt and "Foreign exchange exposure" in child_prompt
    assert CHUNK_ID in child_prompt and "AI infrastructure" not in child_prompt
    specs = {tool["toolSpec"]["name"]: tool["toolSpec"] for tool in child_first["toolConfig"]["tools"]}
    assert {"list_candidates", "get_spans", "submit_findings"} <= specs.keys()
    assert "comparisons" not in specs["finish_research"]["inputSchema"]["json"]["$defs"]["_ReferenceSynthesis"]["properties"]
    research = result.research
    assert research.execution_status == "finished" and research.research_status == "incomplete"
    assert research.plan == result.plan
    assert len(research.scan.sections) == 1 and research.scan.sections[0].ticker == "AAPL"
    assert research.citations[0].chunk_id == CHUNK_ID
    assert research.evidence[0].quote == QUOTE and research.evidence[0].source.source_url == chunk.source_url
    assert research.root_selected_inherited_findings == research.acknowledged_child_findings
    assert {code for _, code in research.errors} == {"unknown_id", "unseen_citation"}
    assert "A stronger dollar" in research.answer
    assert research.publication_state == "selected_evidence" and research.answer_citations == research.citations
    assert research.root_comparisons == ()
    assert result.interpretation.correction_attempt == (2 if needs_correction else None)
    assert result.usage.provider_attempts == 7 + len(replies)
    assert result.usage.reported_tokens == 875 + 125 * len(replies)
    assert research.limits.max_reported_tokens == 30_000 - 125 * len(replies)
    assert result.usage.estimated_tokens == research.usage.estimated_tokens + result.interpretation.estimated_tokens
    assert result.usage.corpus_attempts == 3


def test_two_year_ai_comparison_uses_existing_comparative_execution(corpus):
    data = _structured(mode="ai_infrastructure_comparison", objective="year_over_year", fiscal_years=[2023, 2024],
                       companies=["AMZN"], topic="AI infrastructure supply risk", phrases=["semiconductor products"])
    corrected = {
        **data, "requirements": {**data["requirements"], "comprehensive": True,
                                 "inferred_fields": ["comprehensive"],
                                 "scope_notes": "Review both years to evaluate novelty."},
    }

    def factory(invocation):
        if invocation.depth:
            assert len(invocation.partitions) == 2
            return ScriptedModelClient([{"type": "finish", "answer": "Scripted abstention."}])
        assign = {"type": "tool", "name": "delegate_research", "arguments": {
            "question": "Compare the risk.", "goal": "Review both years.",
            "partitions": [asdict(pair) for pair in invocation.partitions],
        }}
        return ScriptedModelClient([assign, {"type": "synthesis", "synthesis": {}}])

    result = _run(
        corpus, MockTransport([_interpretation(data), _interpretation(corrected)]), research_factory=factory,
        question="Did Amazon add an AI infrastructure supply risk in FY2024 compared with FY2023?",
        max_provider_attempts=10,
    )
    assert result.research.execution_status == "finished" and result.research.research_status == "incomplete"
    assert result.plan.intent.objective == "year_over_year"
    assert result.plan.intent.comprehensive and "comprehensive" in result.plan.intent.inferred_fields
    assert result.interpretation.correction_attempt == 2
    assert result.interpretation.validation_failures[0].issues[0].code == "comprehensive_required_for_comparison"
    assert result.plan.requested_companies == ("AMZN",)
    assert result.research.scan.years == (2023, 2024)
    assert "novelty" in result.research.answer


@pytest.mark.parametrize("root_finishes", [True, False])
def test_apple_review_saves_partial_findings_and_sanitized_validation_diagnostics(corpus, root_finishes):
    from evidenceroute.research.rlm.spans import passage_spans
    from evidenceroute.research.rlm.tools import observation_json

    chunk = corpus.get_chunk(CHUNK_ID)
    spans = ["200:256", "257:290", "291:489"]
    draft = {
        "claim": "A stronger dollar can reduce translated sales and earnings.",
        "analysis_status": "analyzed",
        "evidence": [{"chunk_id": CHUNK_ID, "span_id": span} for span in spans],
    }
    secret = "private provider value"

    def tool(name, **arguments):
        return {"type": "tool", "name": name, "arguments": arguments}

    def factory(invocation):
        if invocation.depth:
            return ScriptedModelClient([
                tool("list_candidates", document_id=chunk.document_id, section_id=chunk.section_id),
                tool("submit_findings", findings=[draft]),
                tool("get_spans", chunk_id=CHUNK_ID, span_ids=spans),
                tool("submit_findings", findings=[draft, {**draft, secret: secret}]),
                {"type": "synthesis", "synthesis": {"cite_refs": [secret]}},
            ])
        return ScriptedModelClient([
            tool("delegate_research", question="Review foreign exchange.", goal="Find exact passages.",
                 partitions=[asdict(pair) for pair in invocation.partitions]),
            {"type": "synthesis", "synthesis": {"cite_refs": ["d1" if root_finishes else "d2"]}},
        ])

    result = _run(corpus, MockTransport([_interpretation(_structured())]), research_factory=factory).research
    assert result.execution_status == result.research_status == "incomplete"
    assert result.groups[0].outcome.assignment_status == "failed"
    assert result.groups[0].outcome.findings_acknowledged_by_root
    assert result.usage.lexical_chunks_scanned == 53 and result.usage.finding_submission_attempts == 2
    assert result.usage.corpus_attempts == 4 and result.usage.provider_attempts == 0
    assert len(result.acknowledged_child_findings) == 1
    evidence = result.acknowledged_child_findings[0].evidence
    canonical = passage_spans(chunk.text)
    assert [item.span_id for item in evidence] == spans
    assert [item.quote for item in evidence] == [chunk.text[slice(*canonical[span])] for span in spans]
    assert "A stronger dollar" in result.answer
    assert len(result.citations) == len(result.root_selected_inherited_findings) == int(root_finishes)
    assert ("Partial findings from unfinished execution." in result.answer) != root_finishes
    assert result.publication_state == ("selected_evidence" if root_finishes else "partial_evidence")
    assert result.answer_citations == (evidence[0].source,)

    root, child = result.diagnostics
    assert child["stop_reason"] == "model_error"
    edge = root["decisions"][0]["operations"][0]
    assert edge["child_status"] == "failed"
    assert edge["retained_finding_ids"] == list(edge["offered_finding_ids"]) == list(edge["submitted_finding_ids"])
    assert edge["submitted_finding_ids"] == edge["acknowledged_finding_ids"]
    assert bool(edge["used_finding_ids"]) == root_finishes
    assert child["decisions"][1]["operations"][0]["finding_validation"] == [
        {"path": "findings[0].evidence[0].span_id", "category": "provenance", "code": "unacknowledged_span"},
    ]
    assert child["decisions"][3]["operations"][0]["finding_validation"] == [
        {"path": "findings[1]", "category": "schema", "code": "extra_forbidden"},
    ]
    assert child["decisions"][4]["finding_validation"] == [
        {"path": "synthesis.cite_refs[0]", "category": "schema", "code": "string_pattern_mismatch"},
    ]
    assert secret not in observation_json(result)


def _operand(ticker, year, section):
    return {"ticker": ticker, "filing_year": year, "section_id": section, "period_text": None}


def _comparison(**overrides):
    return _structured(
        mode="generic_synthesis", comprehensive=True, inferred_fields=["comprehensive"],
        scope_notes="Review every named filing source.", **overrides,
    )


class RecordingClient(ScriptedModelClient):
    def __init__(self, decisions):
        super().__init__(decisions)
        self.prompts = []

    def next_action(self, question, tools, observation, *, remaining_seconds):
        self.prompts.append(question)
        return super().next_action(question, tools, observation, remaining_seconds=remaining_seconds)


def _tool(name, **arguments):
    return {"type": "tool", "name": name, "arguments": arguments}


def _scripted_factory(corpus, prompts, *, conclude=True, statement="Scripted cross-source relationship."):
    from evidenceroute.research.rlm.spans import passage_spans

    def factory(invocation):
        if invocation.depth:
            actions = []
            drafts = []
            for pair in invocation.partitions:
                source = corpus.read_chunks(pair.document_id, pair.section_id, limit=1).items[0]
                span = next(iter(passage_spans(source.text)))
                actions.append(_tool("get_spans", chunk_id=source.chunk_id, span_ids=[span]))
                drafts.append({
                    "claim": f"Scripted disclosure in {pair.section_id}.", "analysis_status": "analyzed",
                    "evidence": [{"chunk_id": source.chunk_id, "span_id": span}],
                })
            actions.append({"type": "finish", "answer": "Child evidence.", "findings": drafts})
        else:
            grouped = defaultdict(list)
            order = []
            for pair in invocation.partitions:
                filing = corpus.get_filing(pair.document_id)
                if filing.ticker not in grouped:
                    order.append(filing.ticker)
                grouped[filing.ticker].append((filing, pair))
            actions = []
            bindings = []
            for ticker in order:
                actions.append(_tool(
                    "delegate_research", question="Inspect the assigned filing.", goal="Record evidence.",
                    partitions=[asdict(pair) for _, pair in grouped[ticker]],
                ))
            ref = 1
            for ticker in order:
                for filing, pair in grouped[ticker]:
                    bindings.append({
                        "ref": f"d{ref}", "ticker": filing.ticker, "filing_year": filing.fiscal_year,
                        "section_id": pair.section_id,
                    })
                    ref += 1
            synthesis = {"cite_refs": [item["ref"] for item in bindings], "analysis_status": "analyzed"}
            if conclude:
                synthesis["conclusions"] = [{
                    "statement": statement, "disposition": "answered",
                    "qualification": "The relationship is limited to the cited passages.", "bindings": bindings,
                }]
            else:
                synthesis["conclusions"] = []
            actions.append({"type": "synthesis", "synthesis": synthesis})
        client = RecordingClient(actions)
        prompts.append(client)
        return client

    return factory


def _assert_generic_publication(result, statement):
    research = result.research
    assert research.comparative_capture is False and research.root_comparisons == ()
    assert research.scan is not None and research.scan.discovery is not None
    assert "Batch-root comparisons" not in research.answer
    assert "No filing-wide novelty conclusion is established" not in research.answer
    assert "Captured findings and lexical scan results cannot establish absence" in research.answer
    assert "unverified" in research.answer and statement in research.answer
    assert research.published_conclusions
    assert all(item.conclusion.semantic_status == "unverified" for item in research.published_conclusions)
    return research


WIDER = SharedLimits(max_provider_attempts=8, max_reported_tokens=10_000, max_corpus_attempts=8,
                     max_delegations=2, max_partitions=4, max_depth=1)


def test_descriptive_research_infers_a_processed_section(corpus):
    data = _structured(
        section_selection="inferred", section_ids=["item_1"], companies=["Apple Inc."],
        topic="Apple business description", phrases=["products", "services"],
        scope_notes="The question asks for the business description and does not name a section.",
    )
    prompts = []
    result = _run(
        corpus, MockTransport([_interpretation(data)]), research_factory=_scripted_factory(corpus, prompts, conclude=False),
        question="What business does Apple describe in its FY2024 annual report?",
        max_batches=1, max_provider_attempts=10,
    )
    intent = result.plan.intent
    assert intent.objective == "review" and intent.required_operands == ()
    assert intent.section_ids == ("item_1",) and "section_ids" in intent.inferred_fields
    assert "Inferred filing sections from the corpus inventory: item_1." in intent.scope_notes
    assert intent.scope_notes.startswith("The question asks for the business description")
    assert {target.section_id for target in result.plan.resolved_targets} == {"item_1"}
    assert "Scripted disclosure in item_1." in result.research.answer
    assert result.research.root_comparisons == () and result.research.answer_disposition is None
    assert "cannot establish absence" in result.research.answer
    assert "AI infrastructure" not in prompts[1].prompts[0]


def test_single_source_response_uses_the_shared_contract(corpus):
    question = "What does Apple's FY2024 MD&A disclose?"
    statement = "Apple describes its financial results in MD&A."
    result = _run(
        corpus, MockTransport([_interpretation(_structured(section_selection="explicit", section_ids=["item_7"]))]),
        research_factory=_scripted_factory(corpus, [], statement=statement), question=question,
    )
    response = result.response
    assert response.question == question and response.strategy == "rlm"
    assert response.verification_status == "not_checked"
    published, = result.research.published_conclusions
    assert len(published.conclusion.bindings) == 1
    ref, = published.citations
    assert response.citations == [corpus.get_chunk(ref.chunk_id)]
    assert response.answer.startswith(statement) and f"[chunk:{ref.chunk_id}]" in response.answer
    assert "Scripted disclosure" not in response.answer
    assert result.research.diagnostics


@pytest.mark.parametrize("cap", [2, 4])
def test_inferred_sections_are_alternatives_within_the_partition_allowance(corpus, cap):
    data = _comparison(
        companies=["AAPL"], fiscal_years=[2023, 2024], section_selection="inferred",
        section_ids=["item_7", "item_8"], topic="Net sales", phrases=["net sales"],
        required_operands=[_operand("AAPL", year, "item_8") for year in (2023, 2024)],
    )
    result = _run(
        corpus, MockTransport([_interpretation(data)]), max_partitions=cap,
        shared_limits=SHARED.model_copy(update={"max_partitions": cap}),
        research_factory=_scripted_factory(corpus, [], statement="Both filings discuss net sales."),
        question="How did Apple discuss net sales in its FY2023 and FY2024 filings?",
    )
    assert result.plan.intent.section_ids == (("item_7",) if cap == 2 else ("item_7", "item_8"))
    assert result.interpretation.outcome.intent.section_ids == ("item_7", "item_8")
    assert {op.section_id for op in result.plan.intent.required_operands} == {None}
    assert result.research.answer_disposition == "answered"
    assert result.usage.partition_assignments == cap


def test_non_ai_cross_year_comparison_uses_the_generic_executor(corpus):
    question = "Compare Apple's foreign-exchange risk in FY2023 and FY2024."
    data = _comparison(
        companies=["Apple Inc."], fiscal_years=[2023, 2024], topic="Foreign exchange exposure",
        phrases=["foreign currency", "exchange rates"],
        required_operands=[_operand("AAPL", 2023, "item_1a"), _operand("Apple Inc.", 2024, "item_1a")],
    )
    prompts = []
    statement = "Apple's FY2023 and FY2024 risk factors both discuss foreign currency."
    result = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, statement=statement),
        question=question, max_provider_attempts=10, max_corpus_attempts=16,
        shared_limits=SharedLimits(max_provider_attempts=8, max_reported_tokens=10_000, max_corpus_attempts=16,
                                   max_delegations=1, max_partitions=4, max_depth=1),
    )
    intent = result.plan.intent
    assert intent.objective == "review" and intent.fiscal_years == (2023, 2024)
    assert [operand.model_dump() for operand in intent.required_operands] == [
        {"ticker": "AAPL", "filing_year": 2023, "section_id": "item_1a", "period_text": None},
        {"ticker": "AAPL", "filing_year": 2024, "section_id": "item_1a", "period_text": None},
    ]
    assert intent.section_ids == ("item_1a",) and "section_ids" in intent.inferred_fields
    _assert_generic_publication(result, statement)
    assert result.research.answer_disposition == "answered"
    assert result.research.research_status == "incomplete"
    root, child = prompts
    assert question in root.prompts[0] and "AAPL" in root.prompts[0] and "Each conclusion requires these source operands" in root.prompts[0]
    assert "Required source operands" in root.prompts[0]
    assert "Accelerator demand" not in root.prompts[0] and "synthesis.comparisons" not in root.prompts[0]
    assert "No lexical miss proves absence" in child.prompts[0]
    assert "infrastructure_mechanism_hint" not in child.prompts[0] and "Accelerator demand" not in child.prompts[0]
    for row in result.research.scan.sections:
        for hit in row.candidates:
            assert "artificial intelligence" not in hit.terms and "data center" not in hit.terms


def test_three_fiscal_year_comparison(corpus):
    question = "Compare Apple's business description across FY2022, FY2023, and FY2024."
    data = _comparison(
        companies=["AAPL"], fiscal_years=[2022, 2023, 2024], section_selection="explicit", section_ids=["item_1"],
        topic="Business description", phrases=["products", "services"],
        required_operands=[_operand("AAPL", year, "item_1") for year in (2022, 2023, 2024)],
    )
    # The named years make the comparison explicit, so no other requirement is inferred.
    data["requirements"]["inferred_fields"] = []
    data["requirements"]["scope_notes"] = None
    prompts = []
    statement = "Apple describes its business in each of the three filings."
    result = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, statement=statement),
        question=question, max_provider_attempts=10,
    )
    assert result.plan.intent.fiscal_years == (2022, 2023, 2024)
    assert len(result.plan.intent.required_operands) == 3
    assert len(result.plan.resolved_targets) == 3 and result.plan.missing_requirements == ()
    _assert_generic_publication(result, statement)
    assert {binding.filing_year for binding in result.research.published_conclusions[0].conclusion.bindings} == {2022, 2023, 2024}


def test_cross_company_synthesis_round_trips_through_a_checkpoint(corpus):
    question = "Compare the FY2024 business descriptions of Apple and Amazon."
    data = _comparison(
        companies=["AAPL", "AMZN"], fiscal_years=[2024], section_selection="explicit",
        section_ids=["item_1"], topic="Business description", phrases=["products", "services"],
        required_operands=[_operand("AAPL", 2024, "item_1"), _operand("AMZN", 2024, "item_1")],
    )
    prompts = []
    statement = "Apple and Amazon describe different businesses in their FY2024 filings."
    result = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, statement=statement),
        question=question, shared_limits=WIDER, max_delegations=2, max_provider_attempts=10,
    )
    assert result.plan.requested_companies == ("AAPL", "AMZN")
    assert {operand.ticker for operand in result.plan.intent.required_operands} == {"AAPL", "AMZN"}
    assert len(result.plan.resolved_targets) == 2
    _assert_generic_publication(result, statement)
    assert question in prompts[0].prompts[0]

    from evidenceroute.research.rlm.checkpoint import create_checkpoint, load_checkpoint, save_checkpoint

    path = Path(__file__).resolve().parents[2] / f"rlm-m3-generic-{uuid4().hex}.json"
    try:
        checkpoint = create_checkpoint(corpus, result)
        save_checkpoint(path, checkpoint)
        loaded = load_checkpoint(path, corpus)
        assert loaded.version == 4
        assert loaded.plan.intent.required_operands == result.plan.intent.required_operands
        assert loaded.plan.intent.question == question
        published, = loaded.publication.published_conclusions
        assert published.conclusion.semantic_status == "unverified" and statement in loaded.publication.answer
        assert loaded.findings and loaded.publication.answer_citations
        assert "cannot establish absence" in loaded.publication.answer
    finally:
        path.unlink(missing_ok=True)


def test_cross_section_synthesis_does_not_add_an_operand_per_partition(corpus):
    question = "Compare Apple's FY2024 business description with its MD&A."
    data = _comparison(
        companies=["AAPL"], fiscal_years=[2024], section_selection="explicit", section_ids=["item_1", "item_7"],
        topic="Business description and MD&A", phrases=["products", "net sales"],
        required_operands=[_operand("AAPL", 2024, "item_1"), _operand("AAPL", 2024, "item_7")],
    )
    prompts = []
    statement = "The business description and MD&A discuss Apple's products from different perspectives."
    result = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, statement=statement),
        question=question, max_provider_attempts=10,
    )
    assert {target.section_id for target in result.plan.resolved_targets} == {"item_1", "item_7"}
    assert [(item.section_id, item.filing_year) for item in result.plan.intent.required_operands] == [
        ("item_1", 2024), ("item_7", 2024),
    ]
    _assert_generic_publication(result, statement)
    assert {binding.section_id for binding in result.research.published_conclusions[0].conclusion.bindings} == {"item_1", "item_7"}


def test_comparative_figures_inside_one_filing_stay_a_single_review(corpus):
    question = "What does Apple's FY2024 MD&A disclose about its 2022 and 2023 net sales?"
    data = _structured(
        section_selection="explicit", section_ids=["item_7"], companies=["AAPL"], topic="Net sales",
        phrases=["net sales"],
    )
    prompts = []
    result = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, conclude=False),
        question=question, max_provider_attempts=10,
    )
    assert result.plan.intent.objective == "review"
    assert result.plan.intent.fiscal_years == (2024,) and result.plan.intent.required_operands == ()
    assert len(result.plan.resolved_targets) == 1
    assert result.research.root_comparisons == () and result.research.published_conclusions == ()
    assert result.research.answer.count("Scripted disclosure in item_7.") == 1
    assert "Each conclusion requires these source operands" not in prompts[0].prompts[0]
    assert "cannot establish absence" in result.research.answer


def test_unavailable_required_operands_remain_in_scope(corpus):
    question = "Compare Apple and Unlisted Corp. Item 9A disclosures for FY2024 and FY2025."
    data = _comparison(
        companies=["Apple Inc.", "Unlisted Corp."], fiscal_years=[2024, 2025], section_selection="explicit",
        section_ids=["item_9a"], topic="Item 9A controls", phrases=["disclosure controls"],
        required_operands=[
            _operand("AAPL", 2024, "item_9a"), _operand("AAPL", 2025, "item_9a"),
            _operand("Unlisted Corp.", 2024, "item_9a"),
        ],
    )
    result = _run(corpus, MockTransport([_interpretation(data)]), question=question, max_batches=0)
    plan = result.plan
    assert [(item.ticker, item.filing_year, item.section_id) for item in plan.intent.required_operands] == [
        ("AAPL", 2024, "item_9a"), ("AAPL", 2025, "item_9a"), ("Unlisted Corp.", 2024, "item_9a"),
    ]
    assert plan.resolved_targets == ()
    assert {entry.reason for entry in plan.missing_requirements} == {"missing_section", "missing_filing", "unknown_company"}
    assert plan.intent.question == question
    assert result.research.answer_disposition == "insufficient_evidence"
    assert "AAPL FY2025 item_9a" in result.research.answer and "Unlisted Corp. FY2024 item_9a" in result.research.answer
    assert "cannot establish absence" in result.research.answer


@pytest.mark.parametrize(("data", "question", "reason"), [
    (_blocked("unsupported", "Ranking issuers requires an ordering the filings do not supply."),
     "Rank the companies by foreign-exchange exposure.", "unsupported"),
    (_blocked("unsupported", "The question asks for a forecast beyond the filed text."),
     "What will Apple's foreign-exchange risk be next year?", "unsupported"),
    (_blocked("unsupported", "The question needs market data from outside these filings."),
     "How does Apple's FY2024 revenue compare with its stock price?", "unsupported"),
    (_blocked("unsupported", "The question asks for a calculation the filings do not state."),
     "Calculate Apple's FY2024 free cash flow from its filings.", "unsupported"),
    (_blocked("ambiguous", "Which company and filing years should be compared?"),
     "How did the risk change recently?", "ambiguous"),
])
def test_ambiguous_and_unsupported_objectives_stay_explicit(corpus, data, question, reason):
    result = _run(corpus, MockTransport([_interpretation(data)]), question=question)
    assert result.interpretation.outcome.status == reason
    assert result.interpretation.outcome.explanation == data["explanation"]
    assert result.plan is result.research is None


def test_over_capacity_comparison_is_qualified_instead_of_truncated(corpus):
    data = _comparison(
        companies=[f"Company {index}" for index in range(9)], fiscal_years=[2024],
        required_operands=[_operand(f"Company {index}", 2024, "item_1a") for index in range(8)],
    )
    result = _run(corpus, MockTransport([_interpretation(data)]), question="Compare all nine companies' FY2024 risk factors.")
    outcome = result.interpretation.outcome
    assert outcome.status == "unsupported" and outcome.stop_reason == "unsupported_operand_capacity"
    assert "eight" in outcome.explanation and "not truncated" in outcome.explanation
    assert result.plan is None and len(result.interpretation.attempts) == 1


def test_correction_cannot_drop_explicit_comparative_requirements(corpus):
    question = "Compare Apple's FY2023 and FY2024 business descriptions."
    scope = dict(
        companies=["AAPL"], fiscal_years=[2023, 2024], section_selection="explicit", section_ids=["item_1"],
        topic="Business description", required_operands=[_operand("AAPL", 2023, "item_1"), _operand("AAPL", 2024, "item_1")],
    )
    first = _comparison(phrases=["..."], **scope)
    narrowed = _comparison(phrases=["products"], **(scope | {"fiscal_years": [2024]}))
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(narrowed)]), question=question)
    assert result.plan is result.research is None
    assert result.interpretation.validation_failures[1].issues == (
        ValidationIssue("requirements.fiscal_years", "inconsistent_requirements", "correction_changed_scope"),
    )
    dropped = _comparison(phrases=["products"], **(scope | {
        "required_operands": [_operand("AAPL", 2024, "item_1"), _operand("AAPL", 2023, "item_7")],
    }))
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(dropped)]), question=question)
    assert result.interpretation.validation_failures[1].issues == (
        ValidationIssue("requirements.required_operands", "inconsistent_requirements", "correction_changed_scope"),
    )
    review = _structured(companies=["AAPL"], fiscal_years=[2023, 2024], section_selection="explicit", section_ids=["item_1"])
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(review)]), question=question)
    assert result.interpretation.validation_failures[1].issues == (
        ValidationIssue("mode", "inconsistent_requirements", "correction_changed_scope"),
    )


def test_correction_cannot_replace_an_inferred_section(corpus):
    first = _structured(
        section_selection="inferred", section_ids=["item_1"], scope_notes="The business section matches the question.",
        phrases=["..."],
    )
    replacement = _structured(section_selection="inferred", section_ids=["item_7"],
                              scope_notes="The MD&A matches the question.")
    result = _run(corpus, MockTransport([_interpretation(first), _interpretation(replacement)]),
                  question="What business does Apple describe in its FY2024 annual report?")
    assert result.plan is None
    assert result.interpretation.validation_failures[1].issues[0].code == "correction_changed_scope"


def test_budget_limit_and_insufficient_evidence(corpus):
    question = "Compare Apple's FY2023 and FY2024 business descriptions."
    data = _comparison(
        companies=["AAPL"], fiscal_years=[2023, 2024], section_selection="explicit", section_ids=["item_1"],
        topic="Business description", phrases=["products"],
        required_operands=[_operand("AAPL", 2023, "item_1"), _operand("AAPL", 2024, "item_1")],
    )
    limited = _run(
        corpus, MockTransport([_interpretation(data)]), question=question, max_reported_tokens=12_000,
        shared_limits=SharedLimits(max_provider_attempts=8, max_reported_tokens=12_000, max_corpus_attempts=8,
                                   max_delegations=1, max_partitions=4, max_depth=1),
    )
    assert limited.plan.intent.required_operands and limited.research.stop_reason == "insufficient_reported_tokens"
    assert limited.research.batches == () and limited.research.published_conclusions == ()
    prompts = []
    withheld = _run(
        corpus, MockTransport([_interpretation(data)]),
        research_factory=_scripted_factory(corpus, prompts, conclude=False),
        question=question, max_provider_attempts=10,
    )
    assert withheld.research.answer_disposition == "insufficient_evidence"
    assert "Evidence bindings are missing or invalid" in withheld.research.answer
    assert "AAPL FY2023 item_1" in withheld.research.answer and "AAPL FY2024 item_1" in withheld.research.answer
    assert withheld.research.published_conclusions == ()
    assert "No filing-wide novelty conclusion is established" not in withheld.research.answer
    assert "cannot establish absence" in withheld.research.answer


def test_fabricated_period_anchor_is_rejected(corpus):
    data = _comparison(
        companies=["AAPL"], fiscal_years=[2024], section_selection="explicit", section_ids=["item_1", "item_7"],
        required_operands=[
            {**_operand("AAPL", 2024, "item_1"), "period_text": "FY2022"},
            _operand("AAPL", 2024, "item_7"),
        ],
    )
    result = _run(corpus, MockTransport([_interpretation(data)]), max_provider_attempts=1,
                  question="Compare Apple's FY2024 business description with its MD&A.")
    assert result.interpretation.validation_failures[0].issues[0].code == "fabricated_period_text"
    assert result.plan is None
