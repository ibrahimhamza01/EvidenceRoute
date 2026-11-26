"""Offline comparison and source-span regressions on canonical metadata."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.comparison import (
    ComparisonDraft, ReferenceComparisonDraft, render_comparisons, resolve_comparison, resolve_reference_comparison,
    scan_research_plan,
)
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.findings import Finding, FindingEvidence
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES, ComparativeProfile
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import EvidenceRef, observation_json


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
    return {"finding_index": index, "relevance": relevance, "entailment": entailment, "rationale": "Scoped judgment.",
            "infrastructure_connection": "Water and energy shortages could constrain data-center capacity."}


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
    assert result.both_years_contributed == (len(assessments) == 2)
    assert result.evidence_years == ((2023, 2024) if len(assessments) == 2 else (2024,))
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate({**data, "outcome": "verified_new"})


def test_infrastructure_support_requires_a_connection(amd):
    draft = ComparisonDraft.model_validate({"risk": "Data-center capacity", "same_risk": "yes", "assessments": [
        _assessment(1, "other"),
        {**_assessment(2), "infrastructure_connection": None, "rationale": "Accelerator demand increased."},
    ]})
    result = resolve_comparison(draft, amd, (2023, 2024))
    assert result.outcome == "insufficient" and result.both_years_contributed
    assert result.assessments[1].relevance == "uncertain"


@pytest.mark.parametrize(("category", "context_index", "explicit", "contextual"), [
    (None, None, (), ()), ("explicit", None, (), ()), ("contextual", 2, (), (2024,)),
])
def test_water_risk_requires_separate_support_for_contextual_ai(corpus, category, context_index, explicit, contextual):
    chunk = corpus.get_chunk("4f9a71476fee75c3bb848dd2")
    findings = []
    for i, phrase in enumerate(("unavailable power or water supply", "demand for AI services"), 1):
        span, quote = next((span, chunk.text[start:end]) for span, (start, end) in passage_spans(chunk.text).items()
                           if phrase in chunk.text[start:end])
        findings.append(Finding(f"test-{i}", "Microsoft", "Scoped excerpt", (FindingEvidence(EvidenceRef.from_chunk(chunk),
                                      quote, span),), "incomplete", "partial", None, None))
    draft = ComparisonDraft.model_validate({"risk": "Datacenter water supply", "same_risk": "uncertain", "assessments": [
        {**_assessment(1), "ai_specific_connection": "The reported expansion supports AI workloads.",
         "ai_specificity": category, "ai_context_finding_index": context_index},
    ]})
    result = resolve_comparison(draft, findings, (2023, 2024))
    assert result.ai_specific_years == explicit and result.ai_contextual_years == contextual
    assert result.finding_ids == (("test-1", "test-2") if contextual else ("test-1",))
    answer = render_comparisons([result])
    assert "Datacenter water supply" in answer and chunk.chunk_id in answer and "insufficient" in answer


def test_explicit_ai_supply_risk_uses_exact_span(corpus):
    chunk = corpus.get_chunk("4d1474550060251037c5f981")
    span, quote = next((span, chunk.text[start:end]) for span, (start, end) in passage_spans(chunk.text).items()
                       if "artificial intelligence infrastructure" in chunk.text[start:end])
    finding = Finding("test-1", "Amazon", "Supplier concentration", (FindingEvidence(EvidenceRef.from_chunk(chunk), quote, span),),
                      "incomplete", "partial", None, None)
    draft = ComparisonDraft.model_validate({"risk": "Semiconductor supply", "same_risk": "uncertain", "assessments": [
        {**_assessment(1), "ai_specific_connection": "The passage ties limited suppliers to AI infrastructure products.",
         "ai_specificity": "explicit"},
    ]})
    result = resolve_comparison(draft, [finding], (2023, 2024))
    assert result.ai_specific_years == (2024,) and not result.ai_contextual_years
    assert result.outcome == "insufficient"


def _finish(item, used=()):
    return {"type": "finish", "answer": "Sampled excerpt", "used_finding_ids": list(used),
            "findings": [{"claim": "Risk excerpt", "analysis_status": "incomplete", "evidence": [item]}]}


def _span_fixture(corpus, key, phrase, finding_id):
    chunk = corpus.get_chunk(key)
    span, quote = next((span, chunk.text[start:end]) for span, (start, end) in passage_spans(chunk.text).items()
                       if phrase in chunk.text[start:end])
    return Finding(finding_id, chunk.company_name, "Scoped SEC excerpt", (FindingEvidence(EvidenceRef.from_chunk(chunk), quote, span),),
                   "incomplete", "partial", None, None)


def _draft(finding):
    return {"claim": finding.claim, "analysis_status": finding.analysis_status,
            "evidence": [{"chunk_id": item.source.chunk_id, "span_id": item.span_id} for item in finding.evidence]}


def _reference_assessment(ref, **overrides):
    return {"ref": ref, "relevance": "infrastructure", "entailment": "supports", "rationale": "Scoped judgment.", **overrides}


def test_root_combines_direct_and_inherited_spans_with_same_year_ai_context(corpus, plan):
    water = _span_fixture(corpus, "4f9a71476fee75c3bb848dd2", "unavailable power or water supply", "water")
    context = _span_fixture(corpus, "4f9a71476fee75c3bb848dd2", "demand for AI services", "context")
    prior = _span_fixture(corpus, "4815881d3fdefcf0a17e31d7", "unavailable power supply", "prior")
    old, new = (SectionTarget(f.evidence[0].source.document_id, "item_1a") for f in (prior, water))
    delegate = {"type": "tool", "name": "delegate_research", "arguments": {
        "question": "Find water and AI context.", "goal": "Select exact spans.", "partitions": [vars(new)],
    }}

    def read(*findings):
        evidence = [f.evidence[0] for f in findings]
        return {"type": "tool", "name": "get_spans", "arguments": {
            "chunk_id": evidence[0].source.chunk_id, "span_ids": [item.span_id for item in evidence],
        }}

    def synthesize(context_ref):
        return {"type": "synthesis", "synthesis": {"new_findings": [_draft(prior)], "comparisons": [
            {"risk": "Datacenter water supply", "same_risk": "uncertain", "assessments": [
                _reference_assessment("n1", relevance="other", entailment="uncertain"),
                _reference_assessment("d1", ai_specificity="contextual", context_ref=context_ref),
            ]}]
        }}

    root = [delegate, read(prior), {"type": "synthesis", "synthesis": {"new_findings": [_draft(water)]}},
            synthesize("n1"), synthesize("d9"), synthesize("d2")]
    child = [read(water, context), {"type": "finish", "answer": "Scoped excerpts", "findings": [_draft(water), _draft(context)]}]
    trace = RecursiveResearch(corpus, lambda inv: ScriptedModelClient(child if inv.depth else root),
        comparison_scan=scan_research_plan(corpus, plan, ("power",), max_chunks=1),
    ).run("Compare water risk.", partitions=(old, new))
    assert trace.stop_reason == "finished"
    assert all(step.error["code"] == "invalid_findings" for step in trace.steps[2:5])
    assert "same-company, same-year" in trace.steps[3].error["message"]
    assert "d9" in trace.steps[4].error["message"]
    assert len(trace.findings) == 1 and trace.findings[0].evidence == prior.evidence
    assert [f.evidence for f in trace.selected_inherited_findings] == [water.evidence, context.evidence]
    comparison = trace.comparisons[0]
    assert comparison.outcome == "possible_new_needs_review" and comparison.ai_contextual_years == (2024,)
    assert comparison.assessments[1].context_finding_id == "child-1:finding-2"
    assert comparison.finding_ids == ("root:finding-1", "child-1:finding-1", "child-1:finding-2")
    assert trace.cited_chunk_ids == ("4815881d3fdefcf0a17e31d7", "4f9a71476fee75c3bb848dd2")


def test_reference_comparison_rejects_cross_company_and_out_of_year_evidence(corpus, amd):
    amazon = _span_fixture(corpus, "4d1474550060251037c5f981", "artificial intelligence infrastructure", "amazon")
    draft = ReferenceComparisonDraft(risk="Supply", same_risk="uncertain",
                                     assessments=[_reference_assessment("d1"), _reference_assessment("d2")])
    with pytest.raises(ValueError, match="one company and the requested years"):
        resolve_reference_comparison(draft, {"d1": amd[0], "d2": amazon}, (2023, 2024))
    with pytest.raises(ValueError, match="one company and the requested years"):
        resolve_reference_comparison(draft, {"d1": amd[0], "d2": amd[1]}, (2024, 2025))


def test_reference_finish_does_not_change_noncomparative_finishes(corpus):
    scripted = [{"type": "synthesis", "synthesis": {}}, {"type": "finish", "answer": "No assessment."}]
    trace = RecursiveResearch(corpus, lambda _: ScriptedModelClient(scripted)).run("Review risk.")
    assert trace.steps[0].error["code"] == "invalid_action"
    assert trace.stop_reason == "finished" and trace.answer == "No assessment."


def test_serialized_compact_finish_fixture(corpus, amd):
    # Reconstruct the two Batch 1 assessments from validation run c00c4eda…, using canonical spans.
    apple = [_span_fixture(corpus, key, "machine learning and artificial intelligence", f"aapl-{i}") for i, key in enumerate(
        ("e4bf3d03444b5f8b4e1614e3", "1b31d472397f437ab1975947"), 1)]
    findings = [*apple, *amd]
    specs = [
        ("Regulatory compliance risks related to machine learning and artificial intelligence laws and regulations", "other",
         ("FY2023 discusses machine learning and AI regulation.", "FY2024 discusses the same regulatory exposure.")),
        ("Power or water shortages constraining data center capacity expansion and product consumption", "infrastructure",
         ("The earlier passage links shortages to capacity expansion.", "The later passage supports the same capacity constraint.")),
    ]
    full, compact = [], []
    available = {f"d{i}": finding for i, finding in enumerate(findings, 1)}
    for offset, (risk, relevance, reasons) in enumerate(specs):
        assessments = [_reference_assessment(f"d{offset * 2 + i}", relevance=relevance, rationale=reason)
                       for i, reason in enumerate(reasons, 1)]
        indexed = [{**{key: value for key, value in assessment.items() if key != "ref"},
                    "finding_index": int(assessment["ref"][1:]),
                    **({"infrastructure_connection": assessment["rationale"]} if relevance == "infrastructure" else {})}
                   for assessment in assessments]
        new = {"risk": risk, "same_risk": "yes", "assessments": assessments}
        old = {**new, "assessments": indexed}
        prior = resolve_comparison(ComparisonDraft.model_validate(old), findings, (2023, 2024))
        current = resolve_reference_comparison(ReferenceComparisonDraft.model_validate(new), available, (2023, 2024))
        assert (prior.outcome, prior.finding_ids, prior.assessed_chunk_ids) == (
            current.outcome, current.finding_ids, current.assessed_chunk_ids)
        full.append(old)
        compact.append(new)
    old_payload = {"answer": "AMD has a comparable capacity risk. Apple excerpts discuss regulation.",
                   "findings": [_draft(f) for f in findings], "comparisons": full, "analysis_status": "incomplete",
                   "used_finding_ids": [f"c00c4eda1b6640719a18e4ee87790f4d.batch-1.child-{i // 2 + 1}:finding-{i % 2 + 1}"
                                        for i in range(4)],
                   "cited_chunk_ids": [f.evidence[0].source.chunk_id for f in findings]}
    new_payload = {"synthesis": {"comparisons": compact, "analysis_status": "incomplete"}}
    before, after = (len(observation_json(payload).encode("utf-8")) for payload in (old_payload, new_payload))
    print(f"Serialized finish input: {before} → {after} UTF-8 bytes ({(before - after) / before:.1%} smaller)")
    assert after < before


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


def test_scoped_candidate_pagination_does_not_grant_evidence(corpus, plan):
    scan = scan_research_plan(corpus, plan, ("business",))
    row = next(row for row in scan.sections if len(row.candidates) >= 5)
    other = next(item.pair for item in scan.sections if item.pair != row.pair)

    def page(pair, **kwargs):
        return {"type": "tool", "name": "list_candidates", "arguments": {**vars(pair), **kwargs}}

    initial = scan.shortlist((row.pair,))[0]
    assert initial["next_offset"] == len(initial["chunks"]) == 3
    hit = row.candidates[3]
    chunk = corpus.get_chunk(hit.chunk_id)
    hit_finish = _finish({"chunk_id": hit.chunk_id, "span_id": next(iter(passage_spans(chunk.text)))})
    read = {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": hit.chunk_id}}
    trace = RecursiveResearch(corpus, lambda _: ScriptedModelClient([
        page(other, offset=3), page(row.pair, limit=11), page(row.pair, offset=3, limit=1), hit_finish,
        page(row.pair, offset=4, limit=1), page(row.pair, offset=len(row.candidates)), read, hit_finish,
    ]), comparison_scan=scan).run("Sample lexical candidates.", partitions=(row.pair,))
    assert [trace.steps[i].error["code"] for i in (0, 1, 3)] == ["unknown_id", "invalid_arguments", "invalid_findings"]
    first, second, end = (trace.steps[i].observation["result"] for i in (2, 4, 5))
    assert first["chunks"] == [hit.chunk_id] and first["next_offset"] == second["offset"] == 4
    assert second["chunks"] == [row.candidates[4].chunk_id]
    assert end["chunks"] == [] and end["next_offset"] is None
    assert all(step.observation_sent and not step.evidence and not step.search_discovered for step in trace.steps[2:6])
    assert [ref.chunk_id for ref in trace.coverage.visible_chunks] == [hit.chunk_id]
    assert trace.shared_budget.corpus_attempts == 6 and trace.stop_reason == "finished"


def test_comparative_profiles_preserve_diagnostic_and_expand_without_changing_protocol():
    from evidenceroute.research.rlm.tools import MAX_PROTOCOL_TOOL_BATCH, MAX_TOOL_BATCH

    diagnostic, expanded = (COMPARATIVE_PROFILES[key] for key in ("diagnostic", "expanded"))
    assert diagnostic.shared_limits.max_provider_attempts == 18 and diagnostic.local_limits.max_response_bytes == 6_000
    assert expanded.model_dump() == {
        "max_groups": 2,
        "max_root_reinvestigation_calls": 2,
        "shared_limits": {"max_depth": 1, "max_delegations": 2, "max_provider_attempts": 36,
                          "max_corpus_attempts": 30, "max_partitions": 4, "max_reported_tokens": 200_000,
                          "deadline_seconds": 300.0},
        "local_limits": {"max_iterations": 12, "max_tool_calls": 12, "max_response_bytes": 16_000,
                         "deadline_seconds": 300.0},
        "finalization": {"child_provider_attempts": 12, "finalization_attempts": 2,
                         "finalization_max_output_tokens": 2048, "finalization_tool_choice": "required",
                         "root_token_headroom": 24_000, "child_token_headroom": 24_000,
                         "finalization_timeout_seconds": 60.0, "finalization_timeout_retries": 1},
    }
    assert (MAX_TOOL_BATCH, MAX_PROTOCOL_TOOL_BATCH) == (4, 16)
    larger = expanded.model_dump()
    larger["max_groups"] = 3
    larger["shared_limits"].update(max_delegations=3, max_partitions=6, max_provider_attempts=42)
    assert ComparativeProfile.model_validate_json(json.dumps(larger)).max_groups == 3


@pytest.mark.parametrize(("section", "key", "value", "message"), [
    ("shared_limits", "max_provider_attempts", 26, "allocation must leave"),
    ("shared_limits", "max_provider_attempts", 27, "Root provider allowance"),
    ("shared_limits", "max_corpus_attempts", 3, "corpus allowance"),
    ("shared_limits", "max_partitions", 3, "partition limits"),
    ("shared_limits", "max_delegations", 1, "partition limits"),
    ("shared_limits", "max_depth", 0, "partition limits"),
    ("local_limits", "max_iterations", 3, "Local limits"),
    ("local_limits", "max_tool_calls", 1, "Local limits"),
    ("finalization", "child_provider_attempts", 2, "allocation must leave"),
    ("finalization", "finalization_tool_choice", "forced", "finalization_tool_choice"),
    ("finalization", "root_token_headroom", 152_000, "Token headroom must leave"),
    ("finalization", "finalization_timeout_retries", 2, "Timeout retries must fit"),
])
def test_comparative_profiles_reject_infeasible_capacity(section, key, value, message):
    data = COMPARATIVE_PROFILES["expanded"].model_dump()
    data[section][key] = value
    with pytest.raises(ValidationError, match=message):
        ComparativeProfile.model_validate(data)


def test_compact_search_spans_and_duplicate_whole_chunk(corpus, plan, amd):
    evidence = amd[0].evidence[0]
    chunk = corpus.get_chunk(evidence.source.chunk_id)
    pair = SectionTarget(chunk.document_id, chunk.section_id)
    other_span = next(span for span in passage_spans(chunk.text) if span != evidence.span_id)
    selected = {"chunk_id": chunk.chunk_id, "span_id": evidence.span_id}
    lookup = {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": chunk.chunk_id}}
    page = {"type": "tool", "name": "read_chunks", "arguments": {
        **vars(pair), "offset": chunk.chunk_index, "limit": 1,
    }}
    trace = RecursiveResearch(corpus, lambda _: ScriptedModelClient([
        {"type": "tool", "name": "search_chunks", "arguments": {"query": "clean water reliable energy"}},
        _finish(selected), {"type": "tool", "name": "get_spans", "arguments": {
            "chunk_id": chunk.chunk_id, "span_ids": [evidence.span_id],
        }}, _finish({**selected, "span_id": other_span}), lookup, lookup, page, _finish(selected),
    ]), comparison_scan=scan_research_plan(corpus, plan, ("clean water",))).run("Check an exact passage.", partitions=(pair,))
    search, hinted, span, unread_span, whole, duplicate, duplicate_page, _ = trace.steps
    assert search.search_discovered and not search.evidence and "text" not in search.observation["result"]["items"][0]
    assert hinted.error["code"] == unread_span.error["code"] == "invalid_findings"
    assert span.observation["result"]["spans"] == {evidence.span_id: evidence.quote}
    assert span.observation_bytes < whole.observation_bytes and span.evidence == (evidence.source,)
    assert duplicate.observation["result"] == {"chunk_id": chunk.chunk_id, "already_delivered": True}
    assert duplicate_page.observation["result"]["items"] == [duplicate.observation["result"]]
    assert not duplicate.evidence and not duplicate_page.evidence
    assert duplicate.observation_bytes < whole.observation_bytes
    assert trace.stop_reason == "finished" and trace.findings[0].evidence == (evidence,)


def test_canonical_demo_reinvestigates_prior_year_under_corpus_budget(corpus, examples, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    monkeypatch.setattr("evidenceroute.research.rlm.bedrock.BedrockRootClient.from_env",
                        lambda **_: pytest.fail("Offline demo must not make provider requests"))
    demo = runpy.run_path(str(examples / "demo_rlm_comparative.py"))
    assert demo["main"](["--fixed-terms"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["research_status"] == "incomplete" and not report["independent_semantic_verification"]
    assert report["execution_status"] == report["recursive_comparative_status"] == "success"
    assert report["group_counts"]["attempted"] == 2
    assert report["root_cited_years_by_company"] == {"AAPL": [], "AMD": [2023, 2024]}
    assert report["answer"].startswith("Research is incomplete.")
    assert "Supported novelty. The validated comparisons establish no novelty conclusion." in report["answer"]
    assert "Comparable existing risks. AMD" in report["answer"]
    assert "Admitted companies with unresolved assessments. AAPL" in report["answer"]
    pending = ", ".join(group["ticker"] for group in report["groups"] if not group["admitted"])
    assert f"Companies outside the admitted scope (23). {pending}" in report["answer"]
    assert "cannot establish absence of a risk or answer for the full requested scope" in report["answer"]
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
    report = demo["_report"](result, scan, False)
    assert report["answer"] is None and report["execution_status"] == "incomplete"


def test_possible_novelty_remains_unresolved_in_public_answer(amd, examples, monkeypatch):
    from evidenceroute.research.rlm.comparison import render_comparisons

    monkeypatch.syspath_prepend(str(examples))
    demo = runpy.run_path(str(examples / "demo_rlm_comparative.py"))
    comparison = resolve_comparison(ComparisonDraft.model_validate({
        "risk": "Capacity constraint", "same_risk": "uncertain",
        "assessments": [_assessment(1, "other"), _assessment(2)],
    }), amd, (2023, 2024))
    assert comparison.outcome == "possible_new_needs_review"
    result = SimpleNamespace(answer_status="incomplete", groups=[SimpleNamespace(ticker="AMD", admitted=True)])
    answer = demo["_scope_answer"](result, (comparison,), render_comparisons([comparison]))
    assert "The assessed spans support a risk in FY2024. Whether it is new remains unresolved." in answer
    assert "Supported novelty. The validated comparisons establish no novelty conclusion." in answer
    assert "Admitted companies with unresolved assessments. AMD" in answer
