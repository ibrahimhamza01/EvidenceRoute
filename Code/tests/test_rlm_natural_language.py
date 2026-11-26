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
from evidenceroute.research.rlm.interpretation import BedrockIntentInterpreter, InterpretationLimits
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
            "section_ids": ["item_1a"], "objective": "review", "topic": "Foreign exchange exposure",
            "comprehensive": False, "inferred_fields": ["section_ids"], "scope_notes": "Use the risk factors section.",
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
    assert outcome.intent.inferred_fields == ("section_ids",)
    assert outcome.intent.scope_notes == "Use the risk factors section."
    assert outcome.discovery.phrases == ("data center", "technical infrastructure")
    assert len(transport.requests) == 1
    payload, _ = transport.requests[0]
    assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "interpret_research_question"}}
    assert len(payload["toolConfig"]["tools"]) == 1
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
    inferred = ["fiscal_years", "section_ids"]
    if not explicit:
        inferred.extend(["company_selection", "comprehensive"])
    data = _structured(company_selection="all", companies=[], comprehensive=True,
                       inferred_fields=inferred, scope_notes="Use the latest snapshot year and Item 1A.")
    question = "List every company in this corpus with foreign exchange risk." if explicit else (
        "Which companies discuss foreign exchange risk?"
    )
    result = _run(corpus, MockTransport([_interpretation(data)]),
                  question=question, max_batches=0)
    assert result.plan.intent.company_selection == "all"
    assert result.plan.intent.comprehensive
    assert result.plan.intent.fiscal_years == (2024,)
    assert ("company_selection" in result.plan.intent.inferred_fields) is not explicit
    assert len(result.plan.requested_companies) == result.plan.required_target_count == 25
    assert result.research.research_status == "incomplete"


def test_explicit_unavailable_scope_remains_in_plan(corpus):
    data = _structured(companies=["Apple Inc.", "Unlisted Corp."], fiscal_years=[2024, 2025],
                       section_ids=["item_1a", "item_9a"], inferred_fields=[], scope_notes=None)
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


@pytest.mark.parametrize(("data", "question", "status", "reason"), [
    (_blocked("unsupported", "The comparison concerns foreign exchange risk."),
     "Compare Apple's FY2023 and FY2024 foreign exchange risk.", "unsupported", "unsupported"),
    (_blocked("ambiguous", "Which company and fiscal years should be compared?"),
     "How did the company's risk change recently?", "ambiguous", "ambiguous"),
    (_structured(mode="ai_infrastructure_comparison", objective="year_over_year", fiscal_years=[2022, 2023, 2024]),
     "Compare AI infrastructure risks in 2022, 2023 and 2024.", "unsupported", "unsupported_comparison_years"),
    (_structured(objective="year_over_year", fiscal_years=[2023, 2024]), "Compare disclosure.", "failed", "invalid_interpretation"),
])
def test_blocked_questions_have_explicit_outcomes(corpus, data, question, status, reason):
    result = _run(corpus, MockTransport([_interpretation(data)]), question=question)
    assert (result.interpretation.outcome.status, result.interpretation.outcome.stop_reason) == (status, reason)
    assert result.plan is result.research is None
    assert result.usage.provider_attempts == 1


@pytest.mark.parametrize("data", [
    _structured(fiscal_years=["2024"]),
    _structured(comprehensive=1),
    _structured(extra_field="private provider content"),
    _structured(fiscal_years=[2023], inferred_fields=["fiscal_years"]),
    _structured(inferred_fields=["section_ids"], scope_notes=None),
    _structured(phrases=[]),
    _structured(phrases=["..."]),
    _structured(phrases=["risk"] * 33),
    _structured(phrases=["x" * 201]),
])
def test_invalid_structured_requirements_and_phrases_fail_closed(corpus, data):
    result = _run(corpus, MockTransport([_interpretation(data)]))
    assert result.interpretation.outcome.stop_reason == "invalid_interpretation"
    assert result.interpretation.attempts[0].error == "invalid_interpretation"
    assert result.usage.provider_attempts == 1
    assert result.research is None
    assert "private provider content" not in str(result.interpretation)


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


def test_review_plan_executes_with_scoped_discovery_and_original_evidence(corpus):
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

    result = _run(corpus, MockTransport([_interpretation(_structured())]), research_factory=factory, clock=clock)
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
    assert result.usage.provider_attempts == 8 and result.usage.reported_tokens == 1_000
    assert result.usage.estimated_tokens == research.usage.estimated_tokens + result.interpretation.estimated_tokens
    assert result.usage.corpus_attempts == 3


def test_two_year_ai_comparison_uses_existing_comparative_execution(corpus):
    data = _structured(mode="ai_infrastructure_comparison", objective="year_over_year", fiscal_years=[2023, 2024],
                       companies=["AMZN"], topic="AI infrastructure supply risk", phrases=["semiconductor products"])

    def factory(invocation):
        if invocation.depth:
            assert len(invocation.partitions) == 2
            return ScriptedModelClient([{"type": "finish", "answer": "Scripted abstention."}])
        assign = {"type": "tool", "name": "delegate_research", "arguments": {
            "question": "Compare the risk.", "goal": "Review both years.",
            "partitions": [asdict(pair) for pair in invocation.partitions],
        }}
        return ScriptedModelClient([assign, {"type": "synthesis", "synthesis": {}}])

    result = _run(corpus, MockTransport([_interpretation(data)]), research_factory=factory,
                  question="Did Amazon add an AI infrastructure supply risk in FY2024 compared with FY2023?")
    assert result.research.execution_status == "finished" and result.research.research_status == "incomplete"
    assert result.plan.intent.objective == "year_over_year"
    assert result.research.scan.years == (2023, 2024)
    assert "novelty" in result.research.answer
