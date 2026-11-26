"""Offline checks of whole-scope scheduling, accounting, and evidence consolidation."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.controller import ModelAttempt, RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans

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
               unacknowledged_root=None, **options):
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
        if any(group.ticker == unacknowledged_root for group in groups):
            model = client(delegates)
            model.last_observation_delivered = False
            return model
        if any(group.ticker == broken_child for group in groups):
            return client([*delegates, {"type": "finish", "answer": "No evidence."}])
        prefix = inv.invocation_id.removesuffix("root")
        finish.update(cited_chunk_ids=[chunk.chunk_id for chunk in chunks],
                      used_finding_ids=[f"{prefix}child-{i}:finding-{j}" for i in range(1, len(groups) + 1) for j in (1, 2)])
        return client([*delegates, finish])

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
                                     scan_terms=("unlikelyliteralfixture",))
    assert result.plan is plan and result.coverage.required_target_count == 6
    assert result.coverage.missing_requirements == plan.missing_requirements and len(plan.resolved_targets) == 4
    assert [batch.companies for batch in result.batches] == [("AMD",), ("NVDA",)]
    assert result.groups[-1].pending_reason == "unresolved_requirements"
    assert result.scan.stop_reason == "chunk_limit" and result.usage.lexical_chunks_scanned == 3
    assert all(not row.candidates for row in result.scan.sections)
    assert result.usage.corpus_attempts == 4
    assert {ref.ticker for ref in result.citations} == {"AMD", "NVDA"}
    assert len(result.evidence) == len(result.citations) == len(result.assessed_chunk_ids) == 4
    assert all(len(target.delivered_span_ids) == 1 and not target.whole_chunk_ids for target in result.coverage.targets)
    assert {item.outcome for item in result.root_comparisons} == {"conflicting"}
    assert {item.outcome for item in result.acknowledged_child_comparisons} == {"comparable_both_years"}
    assert "RAW DRAFT" not in result.answer and "cannot establish absence" in result.answer
    assert len({finding.finding_id for finding in (*result.root_findings, *result.acknowledged_child_findings)}) == 8
    assert len({node.invocation.invocation_id for node in result.coverage.invocations}) == 4
    for item in result.evidence:
        chunk = corpus.get_chunk(item.source.chunk_id)
        start, end = passage_spans(chunk.text)[item.span_id]
        assert item.quote == chunk.text[start:end] and item.source.source_url == chunk.source_url


@pytest.mark.parametrize("failure", ["broken_root", "broken_child", "unacknowledged_root"])
def test_failed_tree_preserves_prior_and_later_evidence(corpus, failure):
    plan = resolve(corpus, ["AAPL", "AMD", "NVDA"])
    result = execute_corpus_research(plan, runner_for(corpus, plan, **{failure: "AMD"}),
                                     limits=overall(), scan_terms=("unlikelyliteralfixture",))
    assert len(result.batches) == 3 and result.execution_status == result.research_status == "incomplete"
    assert {item.source.ticker for item in result.evidence} == {"AAPL", "NVDA"}
    assert {ref.ticker for ref in result.citations} == {"AAPL", "NVDA"}
    assert result.groups[1].pending_reason and result.errors
    retained = result.retained_unacknowledged_child_findings
    if failure == "unacknowledged_root":
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
