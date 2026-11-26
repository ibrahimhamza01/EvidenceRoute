"""Offline comparison and source-span regressions on canonical metadata."""

import json
import runpy
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.comparison import ComparisonDraft, resolve_comparison, scan_research_plan
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.findings import Finding, FindingEvidence
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import EvidenceRef


@pytest.fixture(scope="module")
def corpus():
    try:
        return CorpusEnvironment()
    except FileNotFoundError:
        pytest.skip("Canonical local corpus unavailable")


@pytest.fixture(scope="module")
def examples():
    return Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def plan(corpus, examples):
    example = runpy.run_path(str(examples / "demo_rlm_plan.py"))
    return resolve_research_plan(interpret_intent(example["QUESTION"], example["scripted_interpretation"]), corpus)


def test_scan_reads_complete_inventory_and_is_deterministic(corpus, plan):
    terms = ("data centers depend", "water shortages")
    scan = scan_research_plan(corpus, plan, terms)
    assert scan.stop_reason is None and len(scan.sections) == 50
    assert sum(row.scanned_chunks for row in scan.sections) == 3038 and scan.pages_read == 328
    assert {row.pair for row in scan.sections} == set(plan.resolved_targets)
    assert scan.sections == scan_research_plan(corpus, plan, terms[::-1]).sections
    bounded = scan_research_plan(corpus, plan, terms, max_chunks=11)
    assert bounded.stop_reason == "chunk_limit"
    assert sum(row.scanned_chunks for row in bounded.sections) == 11
    assert sum(row.inventory_chunks for row in bounded.sections) == 3038


@pytest.fixture(scope="module")
def amd(corpus):
    findings = []
    for index, key in enumerate(("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf"), 1):
        chunk = corpus.get_chunk(key)
        span, text = next((span, chunk.text[start:end]) for span, (start, end) in passage_spans(chunk.text).items()
                          if chunk.text[start:end].startswith("Data centers depend"))
        assert "clean water and reliable energy" in text and "data center capacity" in text
        findings.append(Finding(str(index), "AMD", "Capacity constraint",
                                (FindingEvidence(EvidenceRef.from_chunk(chunk), text, span),),
                                "analyzed", "partial", None, None))
    assert "customer’s ability" in findings[0].evidence[0].quote
    assert "customers’ ability" in findings[1].evidence[0].quote
    assert "generate revenue" in findings[1].evidence[0].quote
    return findings


def _assessment(index, relevance="infrastructure", entailment="supports"):
    return {"finding_index": index, "relevance": relevance, "entailment": entailment, "rationale": "Scoped judgment."}


@pytest.mark.parametrize(("assessments", "same_risk", "expected"), [
    ([_assessment(1), _assessment(2)], "yes", "comparable_both_years"),
    ([_assessment(2)], "yes", "insufficient"),
    ([_assessment(1, "other"), _assessment(2)], "uncertain", "possible_new_needs_review"),
    ([_assessment(1, entailment="contradicts"), _assessment(2)], "yes", "conflicting"),
    ([_assessment(1), _assessment(2, "other")], "no", "insufficient"),
    ([_assessment(1), _assessment(2)], "uncertain", "insufficient"),
])
def test_comparison_requires_relevant_support_and_handles_counterevidence(amd, assessments, same_risk, expected):
    data = {"risk": "Data-center capacity", "same_risk": same_risk, "assessments": assessments}
    result = resolve_comparison(ComparisonDraft.model_validate(data), amd, (2023, 2024))
    assert result.outcome == expected and result.ticker == "AMD"
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate({**data, "outcome": "verified_new"})


def _finish(item, used=()):
    return {"type": "finish", "answer": "Sampled excerpt", "used_finding_ids": list(used),
            "findings": [{"claim": "Risk excerpt", "analysis_status": "incomplete", "evidence": [item]}]}


def test_span_ids_require_delivery_and_parent_cannot_expand(corpus, plan, amd):
    older, newer = [finding.evidence[0] for finding in amd]
    pair = SectionTarget(older.source.document_id, older.source.section_id)
    good = {"chunk_id": older.source.chunk_id, "span_id": older.span_id}
    outside = {"chunk_id": newer.source.chunk_id, "span_id": newer.span_id}
    other_span = next(key for key in passage_spans(corpus.get_chunk(good["chunk_id"]).text) if key != older.span_id)
    siblings = {**good, "span_id": other_span}
    child_used = ("child-1:finding-1",)
    root_actions = [
        _finish(good),  # Knowing a canonical ID does not make its text accessible.
        {"type": "tool", "name": "delegate_research", "arguments": {
            "question": "Find the scoped passage.", "goal": "Return one span.",
            "partitions": [{"document_id": pair.document_id, "section_id": pair.section_id}],
        }},
        _finish(siblings, child_used), _finish(good, child_used),
    ]
    child = [
        {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": outside["chunk_id"]}},
        _finish(outside),
        {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": good["chunk_id"]}},
        _finish({**good, "span_id": "0:9999"}), _finish(good),
    ]
    scan = scan_research_plan(corpus, plan, ("clean water",), max_chunks=1)
    trace = RecursiveResearch(corpus, lambda inv: ScriptedModelClient(root_actions if inv.depth == 0 else child),
                              comparison_scan=scan).run("Collect evidence.", partitions=(pair,))
    assert trace.steps[0].error["code"] == trace.steps[2].error["code"] == "invalid_findings"
    child_trace = trace.steps[1].delegation.child
    assert [child_trace.steps[i].error["code"] for i in (0, 1, 3)] == ["unknown_id", "invalid_findings", "invalid_findings"]
    body = child_trace.steps[2].observation
    assert older.span_id in body["span_ids"][older.source.chunk_id] and body["result"]["text"].count(older.quote) == 1
    assert trace.findings[0].evidence[0] == older


def test_oversized_chunk_observation_does_not_unlock_span(corpus, plan, amd):
    item = amd[0].evidence[0]
    selected = {"chunk_id": item.source.chunk_id, "span_id": item.span_id}
    read = {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": item.source.chunk_id}}
    trace = RecursiveResearch(
        corpus, lambda _: ScriptedModelClient([read, _finish(selected), {"type": "finish", "answer": "Abstain."}]),
        comparison_scan=scan_research_plan(corpus, plan, ("clean water",), max_chunks=1),
        local_limits=RootLimits(max_response_bytes=256),
    ).run("Check the quote.", partitions=(SectionTarget(item.source.document_id, item.source.section_id),))
    assert [step.error["code"] for step in trace.steps[:2]] == ["response_too_large", "invalid_findings"]
    assert not trace.findings and not trace.coverage.visible_chunks


def test_canonical_demo_reinvestigates_prior_year_under_corpus_budget(corpus, examples, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    monkeypatch.setattr("evidenceroute.research.rlm.bedrock.BedrockRootClient.from_env",
                        lambda **_: pytest.fail("Offline demo must not make provider requests"))
    demo = runpy.run_path(str(examples / "demo_rlm_comparative.py"))
    assert demo["main"]([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["research_status"] == "incomplete" and not report["independent_semantic_verification"]
    assert [item["outcome"] for item in report["root_comparisons"]] == ["comparable_both_years"]
    assert [item["outcome"] for item in report["acknowledged_comparisons"]] == ["comparable_both_years"]
    rows = report["scan"]["by_company_year"]
    assert [sum(row[k] for row in rows) for k in ("scanned_chunks", "candidate_chunks", "delivered_chunks",
                                               "assessed_chunks")] == [3038, 87, 2, 2]
    assert report["shared_usage"]["corpus_attempts"] == 2 and report["shared_usage"]["provider_attempts"] == 0

    # The same engine cannot do the prior-year lookup when only one corpus attempt is available.
    from evidenceroute.research.rlm.budget import SharedLimits
    from evidenceroute.research.rlm.planned_research import execute_research_plan

    plan = resolve_research_plan(interpret_intent(demo["QUESTION"], demo["scripted_interpretation"]), corpus)
    scan = scan_research_plan(corpus, plan, ("data centers depend",), max_chunks=1)
    result = execute_research_plan(plan, RecursiveResearch(
        corpus, demo["_offline_factory"](corpus, plan), comparison_scan=scan,
        shared_limits=SharedLimits(max_corpus_attempts=1), local_limits=RootLimits(max_iterations=6),
    ))
    assert result.trace.shared_budget.corpus_attempts == 1
    assert not result.trace.comparisons and not result.citations and result.answer_status == "incomplete"
