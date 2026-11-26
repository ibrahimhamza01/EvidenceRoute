"""Verify the question entry point through mocked Bedrock Converse responses."""

import json
from collections import deque
from dataclasses import asdict

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
         interpreter_factory=None, **limits):
    def unexpected(_):
        pytest.fail("Research must be blocked by scope, status, or overall limits.")

    clock = clock if clock is not None else Clock()
    runner = RecursiveResearch(corpus, research_factory or unexpected, shared_limits=SHARED, clock=clock)
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
    assert requirements["properties"]["section_selection"]["enum"] == ["explicit", "risk_factors_default"]
    assert "section_ids" not in requirements["properties"]["inferred_fields"]["items"]["enum"]
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
    (_structured(inferred_fields=["section_ids"], scope_notes="Inferred section."),
     ValidationIssue("requirements.inferred_fields[0]", "schema", "literal_error")),
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
    assert "list_candidates" in specs and "get_spans" not in specs
    assert "comparisons" not in specs["finish_research"]["inputSchema"]["json"]["properties"]
    research = result.research
    assert research.execution_status == "finished" and research.research_status == "incomplete"
    assert research.plan == result.plan
    assert len(research.scan.sections) == 1 and research.scan.sections[0].ticker == "AAPL"
    assert research.citations[0].chunk_id == CHUNK_ID
    assert research.evidence[0].quote == QUOTE and research.evidence[0].source.source_url == chunk.source_url
    assert research.root_selected_inherited_findings == research.acknowledged_child_findings
    assert {code for _, code in research.errors} == {"unknown_id", "unseen_citation"}
    assert "A stronger dollar" in research.answer
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
