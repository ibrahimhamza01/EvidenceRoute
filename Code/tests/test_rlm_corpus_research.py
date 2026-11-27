"""Offline checks of whole-scope scheduling, accounting, and evidence consolidation."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.checkpoint import create_checkpoint
from evidenceroute.research.rlm.controller import ModelAttempt, RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import EvidenceRef

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


def resolve(corpus, companies):
    return resolve_research_plan(interpret_intent("Compare the selected disclosures.", lambda *_: {
        "company_selection": "explicit", "companies": companies, "fiscal_years": [2023, 2024],
        "section_ids": ["item_1a"], "objective": "year_over_year", "topic": "Infrastructure risks",
        "comprehensive": True, "inferred_fields": [],
    }), corpus)


def overall(**overrides):
    return CorpusLimits(**({"max_batches": 10, "max_provider_attempts": 100, "max_corpus_attempts": 100,
                            "max_reported_tokens": 100_000, "max_delegations": 10, "max_partitions": 20,
                            "deadline_seconds": 60.0} | overrides))


class CountedClient(ScriptedModelClient):
    def bind_execution_budget(self, budget):
        super().bind_execution_budget(budget)
        self.budget = budget

    def next_action(self, *args, **kwargs):
        self.budget.start_provider_attempt(estimated_tokens=20)
        attempt = ModelAttempt("fixture", 0, input_tokens=8, output_tokens=2 if self.observations else None,
                               estimated_tokens=20, response_received=True)
        self.last_attempts = (attempt,)
        self.budget.record_provider_usage(attempt)
        return super().next_action(*args, **kwargs)


def runner_for(corpus, plan, *, client=ScriptedModelClient, broken_root=None, broken_child=None,
               unacknowledged_root=None, invalid_synthesis_root=None, **options):
    def factory(inv):
        groups = [group for group in plan.groups if set(group.targets) <= set(inv.partitions)]
        if inv.depth == 0 and any(group.ticker == broken_root for group in groups):
            raise RuntimeError("host setup error")
        if inv.depth and groups[0].ticker == broken_child:
            return client([])
        chunks = [corpus.read_chunks(pair.document_id, pair.section_id, limit=1).items[0]
                  for group in groups for pair in group.targets]
        spans = [next(iter(passage_spans(chunk.text))) for chunk in chunks]
        findings = [{"claim": "Scripted excerpt.", "analysis_status": "incomplete",
                     "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span}]}
                    for chunk, span in zip(chunks, spans)]
        comparisons = [{"risk": "Fixture outcome", "same_risk": "yes", "assessments": [
            {"finding_index": i + index, "relevance": "infrastructure",
             "entailment": "contradicts" if inv.depth == 0 and index == 1 else "supports",
             "rationale": "Scripted assessment", "infrastructure_connection": "Fixture only"}
            for index in (1, 2)]} for i in range(0, len(chunks), 2)]
        finish = {"type": "finish", "answer": "RAW DRAFT FROM TEST", "findings": findings, "comparisons": comparisons}
        if inv.depth:
            reads = [{"type": "tool", "name": "get_spans", "arguments": {"chunk_id": chunk.chunk_id, "span_ids": [span]}}
                     for chunk, span in zip(chunks, spans)]
            return client([*reads, finish])
        delegates = [{"type": "tool", "name": "delegate_research", "arguments": {
            "question": f"Read {group.ticker}.", "goal": "Return excerpt.", "partitions": [asdict(pair) for pair in group.targets],
        }} for group in groups]
        reference_finish = {"type": "synthesis", "synthesis": {"comparisons": [
            {**comparison, "assessments": [
                {"ref": f"d{item['finding_index']}", "relevance": item["relevance"],
                 "entailment": item["entailment"], "rationale": item["rationale"]}
                for item in comparison["assessments"]]} for comparison in comparisons
        ]}}
        if any(group.ticker == unacknowledged_root for group in groups):
            model = client([*delegates, reference_finish])
            model.last_observation_delivered = False
            return model
        if any(group.ticker == invalid_synthesis_root for group in groups):
            return client([*delegates, {"type": "synthesis", "synthesis": {"cite_refs": ["d99"]}}])
        if any(group.ticker == broken_child for group in groups):
            return client([*delegates, {"type": "finish", "answer": "No evidence."}])
        return client([*delegates, reference_finish])

    return RecursiveResearch(corpus, factory,
                             shared_limits=options.pop("shared_limits", SharedLimits(
                                 max_depth=1, max_delegations=1, max_partitions=2, max_provider_attempts=8,
                                 max_reported_tokens=10_000, max_corpus_attempts=2)),
                             finalization=FinalizationPolicy(child_provider_attempts=3), **options)


@pytest.mark.parametrize("limited", [False, True])
def test_canonical_demo_batches_all_scope_or_keeps_pending(corpus, limited, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    main = runpy.run_path(str(EXAMPLES / "demo_rlm_corpus_research.py"))["main"]
    assert main(["--limited"] if limited else []) == int(limited)
    report = json.loads(capsys.readouterr().out)
    result = report["result"]
    counts = report["group_counts"]
    assert report["required_targets"] == report["resolved_targets"] == len(result["coverage"]["targets"]) == 50
    assert counts["requested"] == counts["scheduled"] == 25
    assert (counts["attempted"], counts["completed"], counts["pending"]) == ((2, 2, 23) if limited else (25, 25, 0))
    sizes = [len(batch["companies"]) for batch in result["batches"]]
    assert sizes == ([2] if limited else [2] * 12 + [1])
    assert result["usage"]["lexical_chunks_scanned"] == 3038 and result["usage"]["lexical_pages_read"] == 328
    assert result["usage"]["provider_attempts"] == 0 and len(result["citations"]) == 2
    assert result["research_status"] == "incomplete" and not report["independent_semantic_verification"]
    assert "Admitted companies with unresolved assessments. AAPL" in result["answer"]
    assert f"Companies outside the admitted scope ({counts['pending']})." in result["answer"]


@pytest.mark.parametrize(("cap", "ceiling", "stop"), [
    ("max_batches", 1, "batch_limit"),
    ("max_provider_attempts", 12, "insufficient_provider_attempts"),
    ("max_reported_tokens", 10_045, "insufficient_reported_tokens"),
    ("max_corpus_attempts", 3, "insufficient_corpus_attempts"),
    ("max_delegations", 1, "insufficient_delegations"),
    ("max_partitions", 3, "insufficient_partitions"),
])
def test_overall_allowances_settle_and_admit_against_remaining(corpus, cap, ceiling, stop):
    plan = resolve(corpus, ["AMD", "AAPL"])
    result = execute_corpus_research(plan, runner_for(corpus, plan, client=CountedClient), max_groups=1,
                                     limits=overall(**{cap: ceiling}), scan_terms=("unlikelyliteralfixture",))
    assert result.stop_reason == stop and len(result.batches) == 1
    assert result.batches[0].companies == ("AAPL",) and result.groups[1].pending_reason == stop
    assert result.usage.provider_attempts == 5 and result.usage.reported_tokens == 46
    assert result.usage.estimated_tokens == 100 and result.usage.attempts_with_missing_usage == 2
    assert result.usage.corpus_attempts == 2 and len(result.citations) == 2


def test_zero_candidates_and_unresolved_subset_keep_original_denominator(corpus):
    plan = resolve(corpus, ["NVDA", "NOTREAL", "AMD"])
    result = execute_corpus_research(plan, runner_for(corpus, plan), limits=overall(max_scan_chunks=3),
                                     discovery=DiscoverySpec.from_intent(plan.intent, ("unlikelyliteralfixture",)))
    assert result.plan is plan and result.coverage.required_target_count == 6
    assert result.coverage.missing_requirements == plan.missing_requirements and len(plan.resolved_targets) == 4
    assert [batch.companies for batch in result.batches] == [("AMD",), ("NVDA",)]
    assert result.groups[-1].pending_reason == "unresolved_requirements"
    assert result.scan.stop_reason == "chunk_limit" and result.usage.lexical_chunks_scanned == 3
    assert result.usage.lexical_metadata_reads == result.scan.metadata_reads
    assert all(not row.candidates for row in result.scan.sections)
    assert result.usage.corpus_attempts == 4
    assert {ref.ticker for ref in result.citations} == {"AMD", "NVDA"}
    assert len(result.evidence) == len(result.citations) == len(result.assessed_chunk_ids) == 4
    assert all(len(target.delivered_span_ids) == 1 and not target.whole_chunk_ids for target in result.coverage.targets)
    assert {item.outcome for item in result.root_comparisons} == {"conflicting"}
    assert {item.outcome for item in result.acknowledged_child_comparisons} == {"comparable_both_years"}
    assert "RAW DRAFT" not in result.answer and "cannot establish absence" in result.answer
    assert not result.root_findings
    assert result.root_selected_inherited_findings == result.acknowledged_child_findings
    assert len({finding.finding_id for finding in result.root_selected_inherited_findings}) == 4
    assert all(target.used_finding_ids == target.acknowledged_finding_ids and not target.whole_chunk_ids
               for target in result.coverage.targets)
    assert all(node.selected_inherited_finding_ids == node.used_finding_ids for node in result.coverage.invocations
               if node.invocation.depth == 0)
    assert result.publication_state == "comparison_evidence"
    assert tuple(ref.chunk_id for ref in result.answer_citations) == tuple(dict.fromkeys(
        chunk_id for comparison in (*result.root_comparisons, *result.acknowledged_child_comparisons)
        for chunk_id in comparison.assessed_chunk_ids
    ))
    assert len({node.invocation.invocation_id for node in result.coverage.invocations}) == 4
    for item in result.evidence:
        chunk = corpus.get_chunk(item.source.chunk_id)
        start, end = passage_spans(chunk.text)[item.span_id]
        assert item.quote == chunk.text[start:end] and item.source.source_url == chunk.source_url


@pytest.mark.parametrize("failure", ["broken_root", "broken_child", "unacknowledged_root", "invalid_synthesis_root"])
def test_failed_tree_preserves_prior_and_later_evidence(corpus, failure):
    plan = resolve(corpus, ["AAPL", "AMD", "NVDA"])
    result = execute_corpus_research(plan, runner_for(corpus, plan, **{failure: "AMD"}),
                                     limits=overall(), scan_terms=("unlikelyliteralfixture",))
    assert len(result.batches) == 3 and result.execution_status == result.research_status == "incomplete"
    expected_evidence = {"AAPL", "AMD", "NVDA"} if failure == "invalid_synthesis_root" else {"AAPL", "NVDA"}
    assert {item.source.ticker for item in result.evidence} == expected_evidence
    assert {ref.ticker for ref in result.citations} == {"AAPL", "NVDA"}
    assert {item.evidence[0].source.ticker for item in result.root_selected_inherited_findings} == {"AAPL", "NVDA"}
    assert result.groups[1].pending_reason and result.errors
    retained = result.retained_unacknowledged_child_findings
    if failure == "unacknowledged_root":
        assert any(code == "invalid_findings" for _, code in result.errors)
        assert len(retained) == 2 and all(item.evidence[0].source.ticker == "AMD" for item in retained)
        for finding in retained:
            evidence = finding.evidence[0]
            source = corpus.get_chunk(evidence.source.chunk_id)
            start, end = passage_spans(source.text)[evidence.span_id]
            assert evidence.quote == source.text[start:end] and evidence.source.source_url == source.source_url
    else:
        assert not retained


def test_company_too_large_preserves_pending_scope(corpus):
    plan = resolve(corpus, ["AMD"])
    runner = runner_for(corpus, plan, shared_limits=SharedLimits(max_delegations=1, max_partitions=1))
    result = execute_corpus_research(plan, runner, limits=overall())
    assert not result.batches and result.groups[0].pending_reason == "partition_limit"
    assert result.coverage.required_target_count == 2 and result.research_status == "incomplete"


def test_scheduling_without_a_scan_keeps_noncomparative_multi_section_scope(corpus):
    plan = resolve_research_plan(interpret_intent("Review governance and operations.", lambda *_: {
        "company_selection": "explicit", "companies": ["NVDA", "AMD"], "fiscal_years": [2023],
        "section_ids": ["item_1", "item_1a"], "objective": "review", "topic": "Operations",
        "comprehensive": True, "inferred_fields": [],
    }), corpus)
    invocations = []

    def factory(invocation):
        invocations.append(invocation)
        finish = {"type": "finish", "answer": "Scripted abstention."}
        if invocation.depth:
            return ScriptedModelClient([finish])
        return ScriptedModelClient([{"type": "tool", "name": "delegate_research", "arguments": {
            "question": "Review the scoped sections.", "goal": "Inspect operations.",
            "partitions": [asdict(pair) for pair in invocation.partitions],
        }}, finish])

    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(max_delegations=1, max_partitions=2),
                               finalization=FinalizationPolicy())
    result = execute_corpus_research(plan, runner, limits=overall())
    assert result.execution_status == "finished" and result.research_status == "incomplete"
    assert result.scan is None and result.coverage.required_target_count == 4
    assert [inv.partitions for inv in invocations if inv.depth] == [group.targets for group in plan.groups]
    assert result.publication_state == "insufficient_evidence" and not result.answer_citations
    assert "Insufficient captured evidence" in result.answer and "cannot establish absence" in result.answer
    assert "Scripted abstention" not in result.answer


@pytest.mark.parametrize("root_decision", ["selected", "unfinished", "declined", "unacknowledged"])
def test_descriptive_publication_uses_recorded_evidence_with_or_without_discovery(corpus, monkeypatch, root_decision):
    for module in ("corpus_research", "recursive"):
        monkeypatch.setattr(f"evidenceroute.research.rlm.{module}.uuid4", lambda: UUID(int=1))
    plan = resolve_research_plan(interpret_intent("Review the stated supply risks.", lambda *_: {
        "company_selection": "explicit", "companies": ["AMZN"], "fiscal_years": [2024],
        "section_ids": ["item_1a"], "objective": "review", "topic": "Supply risk",
        "comprehensive": False, "inferred_fields": [],
    }), corpus)
    pair, = plan.resolved_targets
    chunks = sorted(corpus.read_chunks(pair.document_id, pair.section_id, limit=3).items,
                    key=lambda chunk: chunk.chunk_id, reverse=True)
    first, second, unused = (chunk.chunk_id for chunk in chunks)
    spans = {chunk.chunk_id: next(iter(passage_spans(chunk.text))) for chunk in chunks}

    def tool(name, **arguments):
        return {"type": "tool", "name": name, "arguments": arguments}

    def draft(claim, *keys, **qualifications):
        return {"claim": claim, "analysis_status": "analyzed",
                "evidence": [{"chunk_id": key, "span_id": spans[key]} for key in keys], **qualifications}

    drafts = [
        draft("If supply tightens, expenses could rise by $5 million per quarter.", first, second,
              uncertainty="The estimate is conditional."),
        {**draft("Expenses might stay flat during FY2024.", first, conflict="The supplied estimates differ."),
         "analysis_status": "conflicting"},
        draft("A third retained claim.", unused),
    ]

    def factory(invocation):
        if invocation.depth:
            return ScriptedModelClient([
                *(tool("get_spans", chunk_id=key, span_ids=[span]) for key, span in spans.items()),
                {"type": "finish", "answer": "Child draft.", "findings": drafts},
            ])
        decisions = [tool("delegate_research", question="Read disclosures.", goal="Record exact evidence.",
                          partitions=[asdict(pair)])]
        if root_decision == "selected":
            prefix = invocation.invocation_id.removesuffix("root")
            decisions.extend([
                tool("get_chunk", chunk_id=unused),
                {"type": "finish", "answer": "Root draft.", "analysis_status": "analyzed",
                 "used_finding_ids": [f"{prefix}child-1:finding-{index}" for index in (1, 2)],
                 "cited_chunk_ids": [first, unused]},
            ])
        elif root_decision != "unfinished":
            decisions.append({"type": "finish", "answer": "Root draft."})
        client = ScriptedModelClient(decisions)
        if root_decision == "unacknowledged":
            client.last_observation_delivered = False
        return client

    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(max_delegations=1, max_partitions=1))
    discovery = DiscoverySpec.from_intent(plan.intent, ("unlikelyliteralfixture",))
    direct = execute_corpus_research(plan, runner, limits=overall())
    scanned = execute_corpus_research(plan, runner, limits=overall(), discovery=discovery)
    assert direct.answer == scanned.answer and direct.answer_citations == scanned.answer_citations
    assert direct.coverage == scanned.coverage and direct.citations == scanned.citations
    assert direct.root_selected_inherited_findings == scanned.root_selected_inherited_findings
    assert direct.usage.corpus_attempts == scanned.usage.corpus_attempts
    assert direct.usage.provider_attempts == scanned.usage.provider_attempts == 0
    assert direct.execution_status == scanned.execution_status
    assert direct.research_status == scanned.research_status == "incomplete"
    assert direct.stop_reason == scanned.stop_reason == "schedule_completed"
    publication = create_checkpoint(corpus, direct).publication
    assert direct.answer.endswith(publication.answer) and direct.answer_citations == publication.answer_citations
    assert "Research is incomplete" in direct.answer and "Root draft." not in direct.answer
    child = direct.coverage.invocations[1]
    assert len(child.findings) == 3
    for finding, original in zip(child.findings, drafts):
        assert finding.claim == original["claim"]
        for evidence in finding.evidence:
            chunk = corpus.get_chunk(evidence.source.chunk_id)
            start, end = passage_spans(chunk.text)[evidence.span_id]
            assert evidence.source == EvidenceRef.from_chunk(chunk) and evidence.quote == chunk.text[start:end]

    expected_state, keys = {
        "selected": ("selected_evidence", [first, second]),
        "unfinished": ("partial_evidence", [first, second, unused]),
        "declined": ("insufficient_evidence", []),
        "unacknowledged": ("insufficient_evidence", []),
    }[root_decision]
    assert direct.publication_state == scanned.publication_state == publication.publication_state == expected_state
    assert [ref.chunk_id for ref in direct.answer_citations] == keys
    cited_in_answer = [key for line in direct.answer.splitlines() if "Evidence chunks " in line
                       for key in line.split("Evidence chunks ")[1].split(" Uncertainty.")[0].split(" Conflict.")[0].split(", ")]
    assert list(dict.fromkeys(cited_in_answer)) == keys
    if root_decision == "selected":
        assert direct.execution_status == "finished" and direct.coverage.targets[0].status == "partial"
        assert [ref.chunk_id for ref in direct.citations] == sorted([first, unused])
        assert cited_in_answer.count(first) == 2
        assert "Findings selected by finished roots." in direct.answer
        assert "A third retained claim." not in direct.answer
    if root_decision == "unfinished":
        assert direct.execution_status == "incomplete" and not direct.citations
        assert "Partial findings from unfinished execution." in direct.answer
    if keys:
        assert "AMZN FY2024" in direct.answer and "Semantically unverified" in direct.answer
        assert all(draft["claim"] in direct.answer for draft in drafts[:2])
        assert "The estimate is conditional." in direct.answer and "The supplied estimates differ." in direct.answer
    if root_decision == "declined":
        assert child.acknowledged_findings == direct.acknowledged_child_findings == child.findings
    if root_decision == "unacknowledged":
        assert direct.retained_unacknowledged_child_findings == child.findings and not child.acknowledged_findings


def test_batch_exception_after_work_still_settles_provider_and_corpus_usage(corpus, monkeypatch):
    from evidenceroute.research.rlm import corpus_research

    execute = corpus_research.execute_research_plan

    def fail_after_execution(plan, runner, **options):
        result = execute(plan, runner, **options)
        if plan.requested_companies == ("AMD",):
            raise RuntimeError("fixture exception after charged work")
        return result

    monkeypatch.setattr(corpus_research, "execute_research_plan", fail_after_execution)
    plan = resolve(corpus, ["AAPL", "AMD", "NVDA"])
    result = execute_corpus_research(plan, runner_for(corpus, plan, client=CountedClient),
                                     limits=overall(max_corpus_attempts=5), scan_terms=("fixture",))
    assert result.stop_reason == "insufficient_corpus_attempts"
    assert result.usage.corpus_attempts == 4 and result.usage.provider_attempts == 10
    assert result.usage.reported_tokens == 92 and result.usage.attempts_with_missing_estimate == 5
    assert result.batches[1].setup_or_execution_error == "RuntimeError"
    assert {ref.ticker for ref in result.citations} == {"AAPL"}


def test_total_deadline_caps_all_trees(corpus):
    now = [0.0]
    remaining = []

    class Timed(ScriptedModelClient):
        def next_action(self, *args, remaining_seconds, **kwargs):
            remaining.append(remaining_seconds)
            now[0] += 1
            return super().next_action(*args, remaining_seconds=remaining_seconds, **kwargs)

    plan = resolve(corpus, ["AAPL", "AMD", "NVDA"])
    result = execute_corpus_research(plan, runner_for(corpus, plan, client=Timed, clock=lambda: now[0],
                                                      local_limits=RootLimits(deadline_seconds=20.0)),
                                     limits=overall(deadline_seconds=7.0), scan_terms=("fixture",))
    assert remaining == [7, 6, 5, 4, 3, 2, 1]
    assert result.stop_reason == "deadline" and len(result.batches) == 2
    assert result.batches[-1].stop_reason == "deadline" and result.groups[-1].pending_reason == "deadline"
    assert {ref.ticker for ref in result.citations} == {"AAPL"}
