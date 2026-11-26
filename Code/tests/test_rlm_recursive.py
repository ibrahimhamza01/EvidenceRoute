"""Offline recursive workflows over canonical-shaped local fixtures."""

import csv
import runpy
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.exceptions import ReadTimeoutError

from evidenceroute.ingestion.processor import chunk_sections, write_parquet
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import BudgetExceeded, FinalizationPolicy, SharedBudget, SharedLimits
from evidenceroute.research.rlm.claude import ClaudeConfig, ClaudeRootClient, ProviderTransportError
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.planned_research import execute_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.research_coverage import aggregate_research_coverage
from evidenceroute.schemas import Filing, Section

RISK = "item_1a"
BUSINESS = "item_1"
QUOTE = "Supply interruptions affect operations."


@pytest.fixture
def snapshot(tmp_path):
    records, sections, chunks = [], [], []
    for offset, (ticker, year) in enumerate([("NVDA", 2023), ("NVDA", 2024), ("OTHER", 2024)], start=1):
        cik = "0001045810" if ticker == "NVDA" else "0000000002"
        accession = f"{cik}-{str(year)[2:]}-{offset:06d}"
        filing = Filing(
            document_id=f"{cik}_{accession}", company_name=f"{ticker} Fixture", ticker=ticker,
            cik=cik, accession_number=accession, form_type="10-K", fiscal_year=year,
            report_period_end=f"{year}-01-28", filing_date=f"{year}-02-28",
            primary_document="sample.htm",
            source_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/sample.htm",
        )
        records.append(filing.model_dump(mode="json", exclude_none=True))
        selected = [
            Section(document_id=filing.document_id, section_id=section_id, section_name=name,
                    source_url=filing.source_url, text=f"{QUOTE} There are other risk details.")
            for section_id, name in ((RISK, "Risk Factors"), (BUSINESS, "Business"))
        ]
        sections.extend(selected)
        chunks.extend(chunk_sections(selected, filing, max_chars=40, overlap_chars=0))
    manifest = tmp_path / "filings_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_parquet(sections, chunks, tmp_path)
    corpus = CorpusEnvironment(manifest_path=manifest, processed_dir=tmp_path)
    return corpus, tuple(record["document_id"] for record in records)


def tool(name, **args):
    return {"type": "tool", "name": name, "arguments": args}


def delegate(*partitions, question="Check one disclosure."):
    return tool("delegate_research", question=question, goal="Find and quote evidence.",
                partitions=[asdict(pair) for pair in partitions])


def finish(*chunk_ids, findings=(), used=(), status="analyzed"):
    return {
        "type": "finish", "answer": "Scripted answer.", "cited_chunk_ids": list(chunk_ids),
        "findings": list(findings), "used_finding_ids": list(used), "analysis_status": status,
    }


def finding(chunk, *, quote=QUOTE, claim="Supply risk discussed.", status="analyzed", **notes):
    return {"claim": claim, "evidence": [{"chunk_id": chunk.chunk_id, "quote": quote}],
            "analysis_status": status, **notes}


def _company_plan(corpus, years=(2023, 2024), sections=(RISK,)):
    intent = interpret_intent("Compare supply risks.", lambda *_: {
        "company_selection": "all", "fiscal_years": years, "section_ids": sections,
        "objective": "year_over_year" if len(years) > 1 else "review", "topic": "supply risks",
        "comprehensive": True, "inferred_fields": [],
    })
    return resolve_research_plan(intent, corpus)


def test_research_coverage_deduplicates_overlapping_reads_and_keeps_citation_boundaries(snapshot):
    corpus, (old, new, _) = snapshot
    plan = _company_plan(corpus, sections=(BUSINESS, RISK))
    old_risk, new_risk, new_business = SectionTarget(old, RISK), SectionTarget(new, RISK), SectionTarget(new, BUSINESS)
    risk, unread = corpus.read_chunks(new, RISK).items
    business = corpus.read_chunks(new, BUSINESS).items
    cutoff = 25

    class FailWithoutAck(ScriptedModelClient):
        def next_action(self, *args, **kwargs):
            self.last_observation_delivered = bool(self._decisions)
            return super().next_action(*args, **kwargs)

    actions = {
        "root": [delegate(old_risk, new_risk), delegate(new_business, new_risk),
                 finish(unread.chunk_id), finish(business[0].chunk_id),
                 finish(risk.chunk_id, used=["child-1:finding-1"], status="incomplete")],
        "child-1": [tool("read_chunks", document_id=new, limit=1), tool("get_chunk", chunk_id=risk.chunk_id),
                    tool("get_section", document_id=new, section_id=RISK, limit=cutoff),
                    tool("search_chunks", query="unmatchableterm"),
                    finish(findings=[finding(risk, status="conflicting", conflict="Comparison inconclusive.")])],
        "child-2": [tool("read_chunks", document_id=new, limit=1),
                    tool("read_chunks", document_id=new, offset=len(business), limit=1),
                    tool("get_section", document_id=new, section_id=RISK, offset=cutoff - 5),
                    tool("get_section", document_id=new, section_id=BUSINESS, offset=cutoff),
                    tool("read_chunks", document_id=new, section_id=BUSINESS, offset=1, limit=1),
                    tool("search_chunks", query="details", filters={"section_ids": [RISK]}, limit=1)],
    }

    def factory(invocation):
        model = FailWithoutAck if invocation.invocation_id == "child-2" else ScriptedModelClient
        return model(actions[invocation.invocation_id])

    trace = RecursiveResearch(corpus, factory).run("Collect excerpts.", partitions=(old_risk, new_risk, new_business))
    report = aggregate_research_coverage(corpus, plan, trace)
    by_pair = {target.pair: target for target in report.targets}
    assert report.required_target_count == 8 and len(report.missing_requirements) == 2
    assert by_pair[old_risk].status == "none" and by_pair[old_risk].assigned_child_ids == ("child-1",)
    seen = by_pair[new_risk]
    assert seen.assigned_child_ids == ("child-1", "child-2")
    assert seen.whole_chunk_ids == (risk.chunk_id,) and unread.chunk_id in seen.discovered_chunk_ids
    assert seen.chunk_inventory_count == 2 and not seen.chunk_inventory_complete
    assert seen.character_ranges == ((0, seen.section_character_count),) and seen.section_text_complete
    assert seen.status == "complete" and seen.failed_invocation_ids == ("child-2",)
    assert seen.delegated_excerpt_chunk_ids == (risk.chunk_id,)
    assert seen.generated_finding_ids == seen.acknowledged_finding_ids == seen.used_finding_ids == ("child-1:finding-1",)
    seen_business = by_pair[new_business]
    assert seen_business.chunk_inventory_complete and not seen_business.section_text_complete
    assert set(seen_business.whole_chunk_ids) == {chunk.chunk_id for chunk in business}
    assert seen_business.character_ranges == ((cutoff, seen_business.section_character_count),)
    root, first, failed = report.invocations
    assert root.whole_chunk_ids == () and root.cited_chunk_ids == (risk.chunk_id,)
    assert first.findings[0].analysis_status == "conflicting"
    assert [(w.pair, w.start, w.end) for w in first.section_windows] == [(new_risk, 0, cutoff)]
    assert failed.stop_reason == "model_error" and failed.findings_acknowledged_by_parent == ()
    assert [step.error["code"] for step in trace.steps[2:4]] == ["unseen_citation", "unseen_citation"]


def test_research_coverage_counts_only_deliverable_batch_items(snapshot):
    corpus, (_, new, _) = snapshot
    plan = _company_plan(corpus, years=(2024,))
    pair = SectionTarget(new, RISK)
    chunks = corpus.read_chunks(new, RISK).items
    batch = {"type": "tool_batch", "calls": [
        {"tool_use_id": f"read-{i}", "name": "get_chunk", "arguments": {"chunk_id": chunk.chunk_id}}
        for i, chunk in enumerate(chunks)
    ]}
    trace = RecursiveResearch(
        corpus, lambda _: ScriptedModelClient([batch, finish()]), local_limits=RootLimits(max_response_bytes=750),
    ).run("Check two chunks.", partitions=(pair,))
    first, second = trace.steps[0].batch_operations
    assert first.observation_sent and second.observation_sent
    assert not first.delivery_error and second.delivery_error == "batch_observation_limit"
    seen = next(t for t in aggregate_research_coverage(corpus, plan, trace).targets if t.pair == pair)
    assert seen.whole_chunk_ids == (chunks[0].chunk_id,) and seen.status == "partial"


@pytest.mark.parametrize("second_status", ["finished", "undelegated", "failed"])
def test_execute_planned_groups_preserves_scope_and_original_citations(snapshot, second_status):
    corpus, (_, new, other) = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    first, second = plan.groups
    chunks = [corpus.read_chunks(doc, RISK).items[0] for doc in (new, other)]
    count = 2 if second_status == "finished" else 1
    selected = chunks[:count]
    called = []

    def factory(invocation):
        called.append(invocation)
        if invocation.depth == 0:
            return ScriptedModelClient([
                delegate(first.targets[0]),  # A host-side guard rejects splitting an admitted group.
                delegate(*first.targets),
                *([delegate(*second.targets)] if second_status != "undelegated" else []),
                finish(*(chunk.chunk_id for chunk in selected),
                       used=[f"child-{i}:finding-1" for i in range(1, count + 1)]),
            ])
        chunk = chunks[invocation.partitions == second.targets]
        last = (tool("search_chunks", query="unmatchableterm")
                if invocation.partitions == second.targets and second_status == "failed"
                else finish(findings=[finding(chunk)]))
        return ScriptedModelClient([tool("get_chunk", chunk_id=chunk.chunk_id), last])

    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(max_delegations=2, max_partitions=4),
                               finalization=FinalizationPolicy())
    result = execute_research_plan(plan, runner, delegation_first=False)
    assert result.plan is plan and result.coverage.required_target_count == 4
    assert result.answer_status == "incomplete" and result.trace.stop_reason == "finished"
    assert set(called[0].partitions) == set(plan.resolved_targets)
    assert result.coverage.rejected_delegations == (("root", "group_violation"),)
    assert result.groups[0].findings_acknowledged_by_root == ("child-1:finding-1",)
    second_outcome = result.groups[1]
    assert second_outcome.pending_reason is None
    assert second_outcome.root_delegate_ids == (() if second_status == "undelegated" else ("child-2",))
    assert second_outcome.access_status == ("none" if second_status == "undelegated" else "partial")
    assert second_outcome.failed_invocation_ids == (("child-2",) if second_status == "failed" else ())
    assert {ref.chunk_id: ref.source_url for ref in result.citations} == {c.chunk_id: c.source_url for c in selected}
    assert result.trace.shared_budget.partition_assignments == (2 if second_status == "undelegated" else 4)


def test_plan_admission_is_deterministic_and_reports_pending_and_missing(snapshot):
    corpus, _ = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    runner = RecursiveResearch(corpus, lambda _: ScriptedModelClient([finish()]),
                               shared_limits=SharedLimits(max_delegations=2, max_partitions=2))
    result = execute_research_plan(plan, runner)
    assert [group.pending_reason for group in result.groups] == [None, "partition_limit"]
    assert result.trace.invocation.partitions == plan.groups[0].targets
    assert result.coverage.required_target_count == 4
    assert sum(not target.admitted for target in result.coverage.targets) == 2

    result = execute_research_plan(_company_plan(corpus), runner)
    assert result.groups[1].pending_reason == "unresolved_requirements"
    assert result.coverage.missing_requirements[0].reason == "missing_filing"
    assert result.coverage.required_target_count == 4


def test_plan_preflight_rejects_infeasible_provider_reservations_before_any_model(snapshot):
    corpus, _ = snapshot
    plan = _company_plan(corpus)
    runner = RecursiveResearch(corpus, lambda _: pytest.fail("No model should start"),
                               shared_limits=SharedLimits(max_delegations=2, max_provider_attempts=9),
                               finalization=FinalizationPolicy())
    with pytest.raises(ValueError, match="allocation must leave"):
        execute_research_plan(plan, runner)


def test_initial_assignment_enforced_for_whole_batch_and_failed_child_can_be_revisited(snapshot):
    corpus, (_, current, _) = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    first, second = (group.targets for group in plan.groups)
    read = tool("get_chunk", chunk_id=corpus.read_chunks(current, RISK).items[0].chunk_id)
    batch = {"type": "tool_batch", "calls": [
        {"tool_use_id": f"request:{index}", "name": action["name"], "arguments": action["arguments"]}
        for index, action in enumerate([delegate(*first), delegate(*first), delegate(*second), read])
    ]}

    class RecordingModel(ScriptedModelClient):
        def next_action(self, question, tools, observation, **kwargs):
            self.offered.append({tool["name"] for tool in tools})
            return super().next_action(question, tools, observation, **kwargs)

    root = RecordingModel([finish(), read, delegate(first[0]), batch, delegate(*first), read, finish()])
    root.offered = []

    def factory(invocation):
        if invocation.invocation_id == "child-1":
            raise ValueError("fixture failed to start")
        return root if invocation.depth == 0 else ScriptedModelClient([finish()])

    result = execute_research_plan(plan, RecursiveResearch(
        corpus, factory, shared_limits=SharedLimits(max_depth=1, max_delegations=3, max_partitions=6),
        finalization=FinalizationPolicy(),
    ))
    trace = result.trace
    assert trace.stop_reason == "finished" and trace.initial_assignment_stop is None
    assert [step.assignment_stage for step in trace.steps] == ["assignment"] * 4 + ["synthesis"] * 3
    assert root.offered[:4] == [{"delegate_research"}] * 4
    assert {"delegate_research", "get_chunk"} <= root.offered[4]
    assert [step.error["code"] for step in trace.steps[:3]] == [
        "initial_assignment_required", "initial_assignment_required", "group_violation",
    ]
    first_child, duplicate, second_child, early_read = trace.steps[3].operations
    assert duplicate.error["code"] == "duplicate_initial_assignment"
    assert early_read.error["code"] == "initial_assignment_required"
    assert first_child.delegation.initial_assignment and second_child.delegation.initial_assignment
    assert not trace.steps[4].delegation.initial_assignment
    assert trace.steps[5].evidence and trace.shared_budget.corpus_attempts == 1
    failed, completed = result.groups
    assert failed.admitted and failed.assignment_status == "failed" and not failed.initial_result_acknowledged
    assert failed.result_acknowledged and failed.initial_invocation_id == "child-1"
    assert failed.root_delegate_ids == ("child-1", "child-3")
    assert failed.failed_invocation_ids == ("child-1",)
    assert completed.assignment_status == "completed" and completed.initial_result_acknowledged
    assert completed.initial_invocation_id == "child-2" and not completed.findings_acknowledged_by_root


@pytest.mark.parametrize(("limits", "protected", "reason"), [
    ({"max_tool_calls": 2}, True, "tool_call_limit"),
    ({"max_iterations": 3}, True, "reserved_local_decisions"),
    ({"max_iterations": 3}, False, "reserved_local_decisions"),
])
def test_initial_assignment_exhaustion_allows_incomplete_finalization(snapshot, limits, protected, reason):
    corpus, _ = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    result = execute_research_plan(plan, RecursiveResearch(
        corpus, lambda inv: ScriptedModelClient([delegate(*plan.groups[0].targets), tool("list_companies"), finish()]
                                               if inv.depth == 0 else [finish()]),
        shared_limits=SharedLimits(max_delegations=2, max_partitions=4),
        local_limits=RootLimits(**limits), finalization=FinalizationPolicy() if protected else None,
    ))
    trace = result.trace
    assert trace.stop_reason == "finished" and trace.initial_assignment_stop == reason
    assert trace.steps[-1].assignment_stage == "incomplete_finalization"
    assert trace.analysis_status == result.answer_status == "incomplete"
    assert result.groups[0].assignment_status == "completed"
    pending = result.groups[1]
    assert pending.admitted and pending.assignment_status == "unassigned"
    assert pending.assignment_pending_reason == reason and pending.initial_invocation_id is None
    assert trace.shared_budget.delegated_invocations == 1 and trace.shared_budget.corpus_attempts == 0


def test_descendants_cannot_consume_later_groups_initial_assignments(snapshot):
    corpus, _ = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    first, second = (group.targets for group in plan.groups)
    scripts = {"root": [delegate(*first), delegate(*second), finish()],
               "child-1": [delegate(first[0]), finish()], "child-2": [finish()]}
    result = execute_research_plan(plan, RecursiveResearch(
        corpus, lambda inv: ScriptedModelClient(scripts[inv.invocation_id]),
        shared_limits=SharedLimits(max_depth=2, max_delegations=2, max_partitions=4),
    ))
    assert result.trace.stop_reason == "finished"
    assert result.coverage.rejected_delegations == (("child-1", "delegation_limit"),)
    assert all(group.initial_result_acknowledged for group in result.groups)
    assert result.trace.shared_budget.partition_assignments == 4


def test_root_failed_group_reinvestigation_is_bounded_across_a_batch(snapshot):
    corpus, _ = snapshot
    plan = _company_plan(corpus, years=(2024,), sections=(RISK, BUSINESS))
    lookups = {"type": "tool_batch", "calls": [
        {"tool_use_id": f"retry-{index}", "name": "search_chunks", "arguments": {"query": "Supply"}}
        for index in range(4)
    ]}
    scripts = {"root": [*(delegate(*group.targets) for group in plan.groups), lookups, finish()],
               "child-1": [], "child-2": [finish()]}
    result = execute_research_plan(plan, RecursiveResearch(
        corpus, lambda inv: ScriptedModelClient(scripts[inv.invocation_id]),
        shared_limits=SharedLimits(max_depth=1, max_delegations=2, max_partitions=4),
        finalization=FinalizationPolicy(), max_root_reinvestigation_calls=2,
    ))
    trace = result.trace
    assert trace.stop_reason == "finished" and result.groups[0].assignment_status == "failed"
    assert trace.tool_calls == 4 and trace.shared_budget.corpus_attempts == 2
    assert trace.steps[-1].finalization_reason == "root_reinvestigation_limit"
    ops = trace.steps[2].operations
    assert [op.outcome for op in ops] == ["executed", "executed", "skipped", "skipped"]
    assert all(op.observation_sent for op in ops)


def research(corpus, scripts, **options):
    clients = {}

    def factory(invocation):
        client = ScriptedModelClient(scripts[invocation.invocation_id])
        clients[invocation.invocation_id] = client
        return client

    return RecursiveResearch(corpus, factory, **options), clients


def test_two_years_with_grandchild_aggregate_original_evidence_and_isolate_histories(snapshot):
    corpus, (previous, current, _) = snapshot
    old, new, business = SectionTarget(previous, RISK), SectionTarget(current, RISK), SectionTarget(current, BUSINESS)
    quote_old = corpus.read_chunks(previous, RISK).items[0]
    quote_new = corpus.read_chunks(current, RISK).items[0]
    runner, clients = research(corpus, {
        "root": [
            delegate(old, question="Check FY2023."), delegate(new, business, question="Check FY2024."),
            finish(quote_old.chunk_id, quote_new.chunk_id, used=("child-1:finding-1", "child-2:finding-1")),
        ],
        "child-1": [tool("read_chunks", document_id=previous, section_id=RISK, limit=10),
                    finish(findings=[finding(quote_old)])],
        "child-2": [delegate(new), finish(
            quote_new.chunk_id, used=["child-3:finding-1"],
            findings=[finding(quote_new, status="conflicting", conflict="Compare wording across years.",
                              uncertainty="Only part of the assigned scope was read.")],
            status="incomplete",
        )],
        "child-3": [tool("get_chunk", chunk_id=quote_new.chunk_id), finish(findings=[finding(quote_new)])],
    })
    trace = runner.run("Compare FY2023 and FY2024.", partitions=(old, new, business), run_id="compare")
    first, second = (step.delegation for step in trace.steps[:2])
    nested = second.child.steps[0].delegation
    assert (first.status, second.status, nested.status) == ("finished", "incomplete", "incomplete")
    assert (nested.child.invocation.parent_id, nested.child.invocation.depth) == ("child-2", 2)
    assert nested.child.invocation.run_id == trace.invocation.run_id == "compare"
    assert all(client.observations[0] is None for client in clients.values())
    assert len({id(client) for client in clients.values()}) == 4
    assert [len(clients[key].observations) for key in ("root", "child-1", "child-2", "child-3")] == [3, 2, 2, 2]
    assert trace.stop_reason == "finished"
    assert trace.cited_chunk_ids == (quote_old.chunk_id, quote_new.chunk_id)
    assert trace.used_finding_ids == ("child-1:finding-1", "child-2:finding-1")
    assert first.delivered_finding_ids == ("child-1:finding-1",)
    assert nested.delivered_finding_ids == ("child-3:finding-1",)
    assert [(ref.fiscal_year, ref.source_url) for ref in trace.coverage.visible_chunks] == [
        (2023, quote_old.source_url), (2024, quote_new.source_url),
    ]
    conflict, = second.child.findings
    assert conflict.company_name == quote_new.company_name
    assert conflict.analysis_status == "conflicting"
    assert conflict.coverage_status == "partial"
    assert conflict.conflict and conflict.uncertainty
    assert conflict.evidence[0].quote == QUOTE
    assert [entry.status for entry in second.child.partition_coverage] == ["partial", "missing"]
    assert [entry.status for entry in trace.partition_coverage] == ["partial", "partial", "missing"]
    assert trace.coverage_status == "partial"
    usage = trace.shared_budget
    assert (usage.model_sessions, usage.model_decisions, usage.delegated_invocations) == (4, 9, 3)
    assert (usage.provider_attempts, usage.corpus_attempts, usage.partition_assignments) == (0, 2, 4)
    json.dumps(asdict(trace))


def test_scoped_metadata_reads_search_and_further_delegation(snapshot):
    corpus, (old, new, other) = snapshot
    old_risk, new_business = SectionTarget(old, RISK), SectionTarget(new, BUSINESS)
    unrelated_chunk = corpus.read_chunks(new, RISK).items[0]
    runner, _ = research(corpus, {
        "root": [delegate(old_risk, new_business), finish()],
        "child-1": [
            tool("list_companies"), tool("list_filings"),
            tool("list_sections", document_id=new),
            tool("get_filing", document_id=other),
            tool("get_section", document_id=new, section_id=RISK),
            tool("get_chunk", chunk_id=unrelated_chunk.chunk_id),
            tool("read_chunks", document_id=new, section_id=RISK),
            tool("read_chunks", document_id=new, limit=10),
            tool("search_chunks", query="supply", filters={"document_ids": [old, new, other],
                                                          "section_ids": [RISK, BUSINESS]}, limit=10),
            delegate(SectionTarget(new, RISK)), delegate(old_risk, new_business),
            delegate(old_risk, old_risk), finish(),
        ],
    }, local_limits=RootLimits(max_tool_calls=15))
    child = runner.run("Explore scoped filings.").steps[0].delegation.child
    results = [step.observation for step in child.steps[:-1]]
    assert results[0]["result"]["items"][0]["ticker"] == "NVDA"
    assert results[0]["result"]["total"] == 1
    assert {item["document_id"] for item in results[1]["result"]["items"]} == {old, new}
    assert [item["section_id"] for item in results[2]["result"]["items"]] == [BUSINESS]
    assert [result["error"]["code"] for result in results[3:7]] == ["unknown_id"] * 4
    assert {item["section_id"] for item in results[7]["result"]["items"]} == {BUSINESS}
    hits = results[8]["result"]
    assert hits["total"] == len(hits["items"]) == 2
    assert {(hit["chunk"]["document_id"], hit["chunk"]["section_id"]) for hit in hits["items"]} == {
        (old, RISK), (new, BUSINESS),
    }
    assert [result["error"]["code"] for result in results[9:12]] == [
        "scope_violation", "scope_violation", "invalid_partitions",
    ]


@pytest.mark.parametrize(("limits", "expected"), [
    (SharedLimits(max_depth=1), "depth_limit"),
    (SharedLimits(max_delegations=1), "delegation_limit"),
    (SharedLimits(max_partitions=2), "partition_limit"),
])
def test_shared_child_admission_rejects_deeper_work(snapshot, limits, expected):
    corpus, (_, new, _) = snapshot
    pair, business = SectionTarget(new, RISK), SectionTarget(new, BUSINESS)
    runner, clients = research(corpus, {
        "root": [delegate(pair, business), finish()],
        "child-1": [delegate(pair), finish()],
    }, shared_limits=limits)
    trace = runner.run("Check a scope.")
    assert set(clients) == {"root", "child-1"}
    nested = trace.steps[0].delegation.child.steps[0].delegation
    assert nested.error == expected
    assert trace.shared_budget.delegated_invocations == 1


def test_corpus_attempts_are_shared_and_budget_feedback_allows_finishing(snapshot):
    corpus, (_, new, _) = snapshot
    pair = SectionTarget(new, RISK)
    chunk = corpus.read_chunks(new, RISK).items[0]
    runner, _ = research(corpus, {
        "root": [delegate(pair), tool("get_chunk", chunk_id=chunk.chunk_id),
                 finish(chunk.chunk_id, used=["child-1:finding-1"])],
        "child-1": [tool("get_chunk", chunk_id=chunk.chunk_id), finish(findings=[finding(chunk)])],
    }, shared_limits=SharedLimits(max_corpus_attempts=1))
    trace = runner.run("Check a risk.")
    assert trace.stop_reason == "finished"
    assert trace.steps[1].error["code"] == "corpus_attempt_limit"
    assert trace.cited_chunk_ids == (chunk.chunk_id,)
    assert trace.shared_budget.corpus_attempts == 1
    assert trace.tool_calls == 2


@pytest.mark.parametrize("mode", ["failed", "oversized", "unacknowledged"])
def test_only_delivered_findings_unlock_citations(snapshot, mode):
    corpus, (_, new, _) = snapshot
    pair = SectionTarget(new, RISK)
    chunk = corpus.read_chunks(new, RISK).items[0]
    last = finish(findings=[finding(chunk, claim="x" * 700)])
    child_actions = [tool("get_chunk", chunk_id=chunk.chunk_id)]
    if mode != "failed":
        child_actions.append(last)

    class RootScript(ScriptedModelClient):
        last_observation_delivered = mode != "unacknowledged"

    scripts = {
        "root": RootScript([delegate(pair), finish(chunk.chunk_id, used=["child-1:finding-1"]), finish()]),
        "child-1": ScriptedModelClient(child_actions),
    }
    limits = RootLimits(max_response_bytes=850 if mode == "oversized" else 24_000)
    trace = RecursiveResearch(corpus, lambda invocation: scripts[invocation.invocation_id],
                              local_limits=limits).run("Check a risk.")
    step = trace.steps[0]
    assert step.delegation.status == {"failed": "failed", "oversized": "oversized",
                                      "unacknowledged": "incomplete"}[mode]
    assert step.delegation.delivered_finding_ids == ()
    assert bool(step.delegation.child.findings) == (mode != "failed")
    if mode == "failed":
        assert step.error["code"] == "child_not_finished"
        assert step.observation["result"]["coverage"][0]["status"] == "partial"
    assert trace.steps[1].error["code"] == "unseen_citation"
    assert trace.coverage.visible_chunks == () and trace.cited_chunk_ids == ()


def test_section_text_and_unseen_quotes_cannot_grant_chunk_attribution(snapshot):
    corpus, (_, new, _) = snapshot
    chunk = corpus.read_chunks(new, RISK).items[0]
    runner, _ = research(corpus, {"root": [
        tool("get_section", document_id=new, section_id=RISK),
        finish(findings=[finding(chunk)]),
        tool("get_chunk", chunk_id=chunk.chunk_id),
        finish(findings=[finding(chunk, quote="fabricated quote")]),
        finish(findings=[finding(chunk)]),
    ]})
    trace = runner.run("Check a risk.")
    assert trace.steps[1].error["code"] == trace.steps[3].error["code"] == "invalid_findings"
    assert trace.steps[1].error["message"].startswith("Finding 1, evidence 1: use an observed chunk")
    assert chunk.chunk_id not in trace.steps[1].error["message"]
    assert trace.stop_reason == "finished"
    assert trace.findings[0].evidence[0].quote == QUOTE


def test_parent_cannot_expand_a_delivered_quote_using_canonical_lookup(snapshot):
    corpus, (_, new, _) = snapshot
    chunk = corpus.read_chunks(new, RISK).items[0]
    runner, _ = research(corpus, {
        "root": [delegate(SectionTarget(new, RISK)),
                 finish(findings=[finding(chunk)], used=["child-1:finding-1"]),
                 finish(findings=[finding(chunk, quote="Supply interruptions")], used=["child-1:finding-1"])],
        "child-1": [tool("get_chunk", chunk_id=chunk.chunk_id),
                    finish(findings=[finding(chunk, quote="Supply interruptions")])],
    })
    trace = runner.run("Check a risk.")
    assert trace.steps[1].error["code"] == "invalid_findings"
    assert trace.steps[1].error["message"] == (
        f"Finding 1, evidence 1 (chunk {chunk.chunk_id}): copy a contiguous quote exactly from "
        "the delivered excerpt or omit it."
    )
    assert trace.findings[0].evidence[0].quote == "Supply interruptions"


def test_parent_must_name_the_delivered_finding_behind_a_delegated_citation(snapshot):
    corpus, (_, new, _) = snapshot
    chunk = corpus.read_chunks(new, RISK).items[0]
    runner, _ = research(corpus, {
        "root": [delegate(SectionTarget(new, RISK)), finish(chunk.chunk_id),
                 finish(chunk.chunk_id, used=["not-delivered"]),
                 finish(chunk.chunk_id, used=["child-1:finding-1"])],
        "child-1": [tool("get_chunk", chunk_id=chunk.chunk_id), finish(findings=[finding(chunk)])],
    })
    trace = runner.run("Cite a delegated finding.")
    assert [step.error["code"] for step in trace.steps[1:3]] == ["invalid_findings"] * 2
    assert trace.stop_reason == "finished"
    assert trace.used_finding_ids == ("child-1:finding-1",)


def test_no_search_hits_leave_scope_and_analysis_incomplete(snapshot):
    corpus, (_, new, _) = snapshot
    runner, _ = research(corpus, {
        "root": [tool("search_chunks", query="nonexistentterm"), finish(status="analyzed")],
    })
    trace = runner.run("Check for a term.", partitions=(SectionTarget(new, RISK),))
    assert trace.steps[0].observation["result"]["total"] == 0
    assert trace.coverage_status == "missing"
    assert trace.analysis_status == "incomplete" and trace.findings == ()


def message(name, args, *, ident="use-1", usage=None):
    return {"type": "message", "stop_reason": "tool_use", "usage": usage, "content": [
        {"type": "tool_use", "id": ident, "name": name, "input": args},
    ]}


class Transport:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []

    def send(self, payload, *, timeout_seconds):
        self.calls.append((payload, timeout_seconds))
        reply = self.responses.popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize("attempt_limit", [2, 3])
def test_retry_charges_shared_attempts_and_reports_failed_usage(snapshot, attempt_limit):
    corpus, (_, new, _) = snapshot
    action = delegate(SectionTarget(new, RISK))
    root = Transport(message(action["name"], action["arguments"], usage={"input_tokens": 5, "output_tokens": 2}))
    child = Transport(
        ProviderTransportError("retry", retryable=True, usage={"input_tokens": 4}),
        message("finish_research", {"answer": "No supported finding."}, usage={"output_tokens": 5}),
    )
    now = [0.0]

    def factory(invocation):
        def wait(seconds):
            now[0] += seconds
        return ClaudeRootClient(root if invocation.depth == 0 else child, clock=lambda: now[0], wait=wait)

    trace = RecursiveResearch(corpus, factory, clock=lambda: now[0],
                              shared_limits=SharedLimits(max_provider_attempts=attempt_limit)).run("Try a child.")
    assert trace.stop_reason == "budget_exhausted"
    assert trace.budget_stop == "provider_attempt_limit"
    assert len(root.calls) == 1
    assert len(child.calls) == attempt_limit - 1
    assert child.calls[0][0]["messages"][0]["role"] == "user"
    assert len(child.calls[0][0]["messages"]) == 1
    assert "Authorized partitions" in child.calls[0][0]["messages"][0]["content"]
    assert trace.shared_budget.provider_attempts == attempt_limit
    assert trace.shared_budget.attempts_with_missing_usage == attempt_limit - 1
    assert trace.shared_budget.reported_tokens == (11 if attempt_limit == 2 else 16)
    subtrace = trace.steps[0].delegation.child
    assert subtrace.model_attempts[0].error == "retry"
    assert subtrace.model_attempts[0].input_tokens == 4


def test_token_threshold_stops_additional_work_after_measured_overage(snapshot):
    corpus, (_, new, _) = snapshot
    agent = Transport(message("get_filing", {"document_id": new}, usage={
        "input_tokens": 5, "output_tokens": 4, "cache_read_input_tokens": 2,
    }))
    runner = RecursiveResearch(
        corpus, lambda _: ClaudeRootClient(agent, config=ClaudeConfig(max_retries=0)),
        shared_limits=SharedLimits(max_reported_tokens=6),
    )
    trace = runner.run("Check a filing.")
    assert trace.budget_stop == trace.steps[0].error["code"] == "reported_token_limit"
    assert trace.shared_budget.reported_tokens == 11
    assert trace.shared_budget.corpus_attempts == 0
    assert len(agent.calls) == 1


@pytest.mark.parametrize(("shared", "local"), [(2.0, 5.0), (5.0, 2.0)])
def test_descendants_inherit_parent_and_shared_deadlines(snapshot, shared, local):
    corpus, (_, new, _) = snapshot
    now = [0.0]
    remaining = []

    class Timed(ScriptedModelClient):
        def next_action(self, question, tools, observation, *, remaining_seconds):
            remaining.append(remaining_seconds)
            now[0] += 1
            return super().next_action(question, tools, observation, remaining_seconds=remaining_seconds)

    def factory(invocation):
        return Timed([delegate(SectionTarget(new, RISK))] if not invocation.depth else [finish()])

    trace = RecursiveResearch(corpus, factory, clock=lambda: now[0],
                              local_limits=RootLimits(deadline_seconds=local),
                              shared_limits=SharedLimits(deadline_seconds=shared)).run("Time a child.")
    assert remaining == [2.0, 1.0]
    assert trace.stop_reason == trace.steps[0].delegation.child.stop_reason == "deadline"
    assert trace.answer is None


def test_factory_cannot_reuse_a_model_session(snapshot):
    corpus, (_, new, _) = snapshot
    shared_client = ScriptedModelClient([delegate(SectionTarget(new, RISK)), finish()])
    trace = RecursiveResearch(corpus, lambda _: shared_client).run("Check a risk.")
    assert trace.steps[0].delegation.error == "child_execution_error"
    assert trace.shared_budget.model_sessions == trace.shared_budget.delegated_invocations == 1


@pytest.fixture
def recursive_demo(snapshot, monkeypatch):
    corpus, _ = snapshot
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    path = Path(__file__).resolve().parents[1] / "examples" / "demo_rlm_recursive.py"
    return runpy.run_path(str(path))["main"]


def test_recursive_demo_defaults_to_scripted_without_provider_setup(recursive_demo, monkeypatch, capsys):
    monkeypatch.setattr(boto3, "Session", lambda: pytest.fail("Offline mode must not construct a session"))
    assert recursive_demo([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "scripted offline"
    assert report["shared_usage"]["provider_attempts"] == 0
    assert report["trace"]["delegations"][1]["child"]["delegations"][0]["child"]["invocation"]["depth"] == 2


def converse_reply(action, tool_id, *, usage=None):
    finish_action = action["type"] == "finish"
    name = "finish_research" if finish_action else action["name"]
    arguments = {key: value for key, value in action.items() if key != "type"} if finish_action else action["arguments"]
    return {
        "stopReason": "tool_use", "output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": tool_id, "name": name, "input": arguments}},
        ]}},
        "usage": usage if usage is not None else {"inputTokens": 11, "outputTokens": 4, "cacheReadInputTokens": 0},
        "metrics": {"latencyMs": 250},
    }


def stub_bedrock(monkeypatch, conversations, *, model="fixture-sonnet"):
    available = deque(deque(replies) for replies in conversations)
    sessions = []
    monkeypatch.setenv("BEDROCK_MODEL_ID", model)
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    # The live demo disables retries even if other demos have them configured.
    monkeypatch.setenv("CLAUDE_MAX_RETRIES", "2")

    class Session:
        def __init__(self):
            self.replies = available.popleft()
            self.requests = []
            self.timeouts = []
            sessions.append(self)

        def client(self, service, *, region_name, config):
            assert service == "bedrock-runtime" and region_name == "us-east-1"
            assert config.retries["total_max_attempts"] == 1
            assert config.read_timeout > 0
            self.timeouts.append((config.connect_timeout, config.read_timeout))
            return StubClient(self.requests, self.replies)

    class StubClient:
        def __init__(self, requests, replies):
            self.requests = requests
            self.replies = replies

        def converse(self, **payload):
            self.requests.append(payload)
            result = self.replies.popleft()
            if isinstance(result, Exception):
                raise result
            return result

        def close(self):
            pass

    monkeypatch.setattr(boto3, "Session", Session)
    return sessions


@pytest.mark.parametrize("case", ["both", "one_cited", "missing_finding", "child_failed", "undelivered"])
def test_live_recursive_experiment_with_stubbed_converse(snapshot, recursive_demo, monkeypatch, capsys, case):
    corpus, (old, new, _) = snapshot
    old_chunk = corpus.read_chunks(old, RISK).items[0]
    new_chunk = corpus.read_chunks(new, RISK).items[0]
    first = delegate(SectionTarget(old, RISK))
    second = delegate(SectionTarget(new, RISK))
    bad_request = ClientError({"Error": {"Code": "ThrottlingException", "Message": "private payload"}}, "Converse")
    cite_both = case == "both"
    finish_action = finish(
        *([old_chunk.chunk_id, new_chunk.chunk_id] if cite_both else [old_chunk.chunk_id]),
        used=["child-1:finding-1", "child-2:finding-1"] if cite_both else ["child-1:finding-1"],
    )
    conversations = [
        [converse_reply(first, "old-delegate"), converse_reply(second, "new-delegate"),
         bad_request if case == "undelivered" else converse_reply(finish_action, "root-finish")],
        [converse_reply(tool("search_chunks", query="supply", limit=1), "old-search"),
         converse_reply(finish(findings=[finding(old_chunk)]), "old-finish")],
        [converse_reply(tool("get_chunk", chunk_id=new_chunk.chunk_id), "new-read"),
         bad_request if case == "child_failed" else converse_reply(
             finish(findings=[] if case == "missing_finding" else [finding(new_chunk, status="incomplete")]),
             "new-finish", usage={"inputTokens": 13, "cacheWriteInputTokens": 3},
         )],
    ]
    sessions = stub_bedrock(monkeypatch, conversations)
    exit_code = recursive_demo(["--live", "--trace"])
    out = capsys.readouterr().out
    assert "private payload" not in out
    report = json.loads(out)
    assert exit_code == (0 if cite_both else 1)
    assert report["status"] == ("success" if cite_both else "incomplete")
    assert not report["semantic_claims_verified"] and not report["exhaustive_descendant_coverage_rollup"]
    assert report["actual_maximum_depth"] == 1
    assert report["region"] == "us-east-1" and report["model_id"] == "fixture-sonnet"
    assert report["authorized_partitions"] == {
        "2023": {"document_id": old, "section_id": RISK}, "2024": {"document_id": new, "section_id": RISK},
    }
    assert report["limits"]["shared"] == {
        "max_depth": 1, "max_delegations": 2, "max_provider_attempts": 12, "max_corpus_attempts": 8,
        "max_partitions": 2, "max_reported_tokens": 60_000, "deadline_seconds": 120,
    }
    assert report["limits"]["local"]["deadline_seconds"] == 120
    assert report["limits"]["local"]["max_response_bytes"] == 6_000
    assert report["limits"]["provider"]["max_retries"] == 0
    assert [len(session.requests) for session in sessions] == [3, 2, 2]
    for index, session in enumerate(sessions):
        first_request = session.requests[0]
        assert len(first_request["messages"]) == 1
        names = {tool["toolSpec"]["name"] for tool in first_request["toolConfig"]["tools"]}
        assert {"search_chunks", "finish_research"} <= names
        assert ("delegate_research" in names) == (index == 0)
    assert "Make two separate delegate_research" in sessions[0].requests[0]["messages"][0]["content"][0]["text"]
    assert old in sessions[1].requests[0]["messages"][0]["content"][0]["text"]
    assert new not in sessions[1].requests[0]["messages"][0]["content"][0]["text"]
    first_result = sessions[0].requests[1]["messages"][-1]["content"][0]["toolResult"]
    assert first_result["toolUseId"] == "old-delegate"
    evidence = first_result["content"][0]["json"]["result"]["findings"][0]["evidence"][0]
    assert evidence["quote"] == QUOTE and evidence["source"]["chunk_id"] == old_chunk.chunk_id
    assert len(QUOTE) <= len(old_chunk.text)
    root_trace = report["trace"]
    first_child, second_child = root_trace["delegations"]
    assert first_child["child"]["invocation"]["parent_id"] == "root"
    assert first_child["child"]["evidence_delivered"]["direct_chunk_text_ids"] == [old_chunk.chunk_id]
    assert root_trace["evidence_delivered"]["direct_chunk_text_ids"] == []
    assert second_child["child"]["coverage_status"] == ("missing" if case == "child_failed" else "partial")
    assert report["shared_usage"]["model_sessions"] == 3
    assert report["shared_usage"]["model_decisions"] == report["provider_usage"]["provider_attempts"] == 7
    assert report["shared_usage"]["corpus_attempts"] == 2 and report["shared_usage"]["partition_assignments"] == 2
    assert report["provider_usage"]["provider_retries"] == 0
    assert report["provider_usage"]["attempts_with_missing_input_or_output"] == (2 if case == "undelivered" else 1)
    if cite_both:
        assert all(report["criteria"].values())
        assert report["shared_usage"]["reported_tokens"] == 106
        assert report["provider_usage"]["tokens"]["input_tokens"]["reported_total"] == 79
        assert report["provider_usage"]["tokens"]["output_tokens"]["reported_total"] == 24
        assert report["provider_usage"]["tokens"]["cache_creation_input_tokens"] == {
            "reported_total": 3, "missing_attempts": 6,
        }
        assert report["provider_usage"]["reported_provider_latency_seconds_sum"] == 1.75
        assert [ref["fiscal_year"] for ref in report["original_citations"]] == [2023, 2024]
        assert not any(ref["direct_chunk_text_delivered"] for ref in report["original_citations"])
        assert all(ref["delegated_excerpts"][0]["quote"] == QUOTE for ref in report["original_citations"])
    else:
        assert not report["criteria"]["original_citations_from_used_findings_for_both_years"]
    if case == "undelivered":
        assert not second_child["observation_acknowledged"]
        assert second_child["child"]["findings_generated"] and not second_child["delivered_finding_ids"]


def test_direct_both_year_citations_do_not_pass_the_recursive_experiment(
    snapshot, recursive_demo, monkeypatch, capsys,
):
    corpus, (old, new, _) = snapshot
    chunks = [corpus.read_chunks(doc, RISK).items[0].chunk_id for doc in (old, new)]
    stub_bedrock(monkeypatch, [[
        converse_reply(tool("get_chunk", chunk_id=chunks[0]), "read-old"),
        converse_reply(tool("get_chunk", chunk_id=chunks[1]), "read-new"),
        converse_reply(finish(*chunks), "finish"),
    ]], model="arn:aws:bedrock:us-east-1:123456789012:inference-profile/fixture")
    assert recursive_demo(["--live", "--max-provider-attempts", "3", "--deadline-seconds", "65",
                           "--finalization-attempts", "0"]) == 1
    output = capsys.readouterr().out
    report = json.loads(output)
    assert "123456789012" not in output
    assert report["stop_reason"] == "finished" and report["status"] == "incomplete"
    assert report["limits"]["shared"]["max_provider_attempts"] == 3
    assert report["limits"]["local"]["deadline_seconds"] == report["limits"]["shared"]["deadline_seconds"] == 65
    assert not report["criteria"]["separate_year_delegations"]
    assert len(report["original_citations"]) == 2
    assert all(ref["direct_chunk_text_delivered"] for ref in report["original_citations"])


def test_live_experiment_reports_budget_exhaustion(recursive_demo, snapshot, monkeypatch, capsys):
    _, (old, _, _) = snapshot
    sessions = stub_bedrock(monkeypatch, [[converse_reply(delegate(SectionTarget(old, RISK)), "only-call")], []])
    assert recursive_demo(["--live", "--max-provider-attempts", "1", "--finalization-attempts", "0"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "incomplete" and report["stop_reason"] == "budget_exhausted"
    assert report["trace"]["budget_stop"] == "provider_attempt_limit"
    assert report["provider_usage"]["provider_attempts"] == 1
    assert list(map(lambda session: len(session.requests), sessions)) == [1, 0]


def converse_batch(*actions):
    reply = converse_reply(actions[0], "batch-1")
    reply["output"]["message"]["content"].extend(
        converse_reply(action, f"batch-{index}")["output"]["message"]["content"][0]
        for index, action in enumerate(actions[1:], 2)
    )
    return reply


def _large_converse_batch(*actions, prefix="oversized"):
    response = converse_batch(*actions)
    for index, block in enumerate(response["output"]["message"]["content"], 1):
        block["toolUse"]["toolUseId"] = f"{prefix}:{index}.request"
    return response


def _batch_feedback(payload):
    return [(block["toolResult"]["toolUseId"], block["toolResult"]["content"][0]["json"])
            for block in payload["messages"][-1]["content"]]


@pytest.mark.parametrize("deadline", [45.0, 300.0])
def test_protected_bedrock_timeout_replays_one_request_and_counts_missing_usage(snapshot, monkeypatch, deadline):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient
    from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES

    corpus, (older, _, _) = snapshot
    chunk = corpus.read_chunks(older, RISK).items[0]
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "20")
    sessions = stub_bedrock(monkeypatch, [[
        converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "native.read:1"),
        ReadTimeoutError(endpoint_url="https://bedrock.fixture"),
        converse_reply(finish(chunk.chunk_id), "native.finish:1"),
    ]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        local_limits=RootLimits(max_tool_calls=1, deadline_seconds=deadline),
        shared_limits=SharedLimits(max_delegations=0, deadline_seconds=deadline),
        finalization=COMPARATIVE_PROFILES["expanded"].finalization,
    ).run("Read one risk disclosure.", partitions=(SectionTarget(older, RISK),))
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
    assert trace.shared_budget.provider_attempts == 3 and trace.shared_budget.finalization_provider_attempts == 2
    assert trace.shared_budget.reported_tokens == 30 and trace.shared_budget.attempts_with_missing_usage == 1
    timeout, retried = trace.steps[-1].model_attempts
    assert timeout.error == "aws_timeout" and timeout.input_tokens is timeout.output_tokens is None
    assert not timeout.response_received and retried.response_received
    assert timeout.request_bytes == retried.request_bytes and timeout.estimated_tokens == retried.estimated_tokens
    assert sessions[0].requests[-2] == sessions[0].requests[-1]
    assert [key for key, _ in _batch_feedback(sessions[0].requests[-1])] == ["native.read:1"]
    assert trace.steps[0].observation_sent
    first, protected, retried_timeouts = sessions[0].timeouts
    assert first == (20.0, 20.0)
    for values in (protected, retried_timeouts):
        assert values[0] == values[1] and 20 < values[0] <= min(60.0, deadline)
        if deadline == 300.0:
            assert values == (60.0, 60.0)


@pytest.mark.parametrize("final_usage", [1_500, 14_000])
def test_token_headroom_switches_before_sending_and_accepts_late_usage_overshoot(snapshot, monkeypatch, final_usage):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (older, _, _) = snapshot
    chunk = corpus.read_chunks(older, RISK).items[0]
    sessions = stub_bedrock(monkeypatch, [[
        converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "native.read:1",
                       usage={"inputTokens": 8_400, "outputTokens": 100}),
        converse_reply(finish(chunk.chunk_id), "native.finish:1",
                       usage={"inputTokens": final_usage - 100, "outputTokens": 100}),
    ]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        shared_limits=SharedLimits(max_delegations=0, max_reported_tokens=22_000),
        finalization=FinalizationPolicy(root_token_headroom=12_000, finalization_tool_choice="required"),
    ).run("Read one risk disclosure.", partitions=(SectionTarget(older, RISK),))
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
    read, blocked, finalized = trace.steps
    assert blocked.error == {"code": "investigation_token_headroom"} and not blocked.model_attempts
    assert finalized.phase == "finalization" and finalized.finalization_reason == "investigation_token_headroom"
    assert read.observation_sent and trace.shared_budget.reported_tokens == 8_500 + final_usage
    assert trace.shared_budget.provider_attempts == 2 and trace.shared_budget.finalization_provider_attempts == 1
    assert [key for key, _ in _batch_feedback(sessions[0].requests[-1])] == ["native.read:1"]


def test_child_finish_can_use_own_headroom_while_protecting_root_and_later_child():
    from evidenceroute.research.rlm.controller import ModelAttempt

    budget = SharedBudget(SharedLimits(max_delegations=2, max_partitions=4, max_reported_tokens=100),
                          finalization=FinalizationPolicy(root_token_headroom=20, child_token_headroom=15))
    budget.initial_delegations_left, budget.initial_partitions_left = 2, 4
    with budget.invocation_allocation(root=True):
        root = budget.active_allocation
        budget.reserve_child(1, 2, initial_assignment=True)
        with budget.invocation_allocation(root=False):
            budget.record_provider_usage(ModelAttempt("fixture", 0.0, 44, 1))
            with pytest.raises(BudgetExceeded, match="investigation_token_headroom"):
                budget.check_provider_attempt(estimated_tokens=6)
            budget.active_allocation.finalizing = True
            budget.check_provider_attempt(estimated_tokens=20)
            with pytest.raises(BudgetExceeded, match="finalization_token_headroom"):
                budget.check_provider_attempt(estimated_tokens=21)
        root.finalizing = True
        budget.check_provider_attempt(estimated_tokens=55)
    assert budget.reported_tokens == 45 and budget.provider_attempts == 0


@pytest.mark.parametrize(("compact", "count"), [(False, 5), (True, 5), (False, 16)])
def test_bedrock_oversized_batch_returns_every_id_and_executes_only_the_correction(snapshot, monkeypatch, compact, count):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (old, new, _) = snapshot
    newer, older = (corpus.read_chunks(doc, RISK).items[0] for doc in (new, old))
    new_read = tool("get_chunk", chunk_id=newer.chunk_id)
    old_read = tool("get_chunk", chunk_id=older.chunk_id)
    first = _large_converse_batch(delegate(SectionTarget(new, RISK)), *([new_read] * (count - 1)))
    # Bedrock can replay binary redacted reasoning. Recovery must not re-encode the full assistant message.
    first["output"]["message"]["content"].insert(0, {"reasoningContent": {"redactedContent": b"fixture"}})
    corrected = converse_batch(old_read, tool("list_companies", limit=1))
    sessions = stub_bedrock(monkeypatch, [[first, corrected, converse_reply(finish(older.chunk_id), "finish")]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        local_limits=RootLimits(max_response_bytes=620 if compact else 6_000),
    ).run("Inspect one older risk disclosure.")

    first_request, correction, _ = sessions[0].requests
    assert "Request at most 4 tool operations per model decision." in first_request["system"][0]["text"]
    assert correction["messages"][-2] == first["output"]["message"]
    feedback = _batch_feedback(correction)
    assert [tool_id for tool_id, _ in feedback] == [f"oversized:{i}.request" for i in range(1, count + 1)]
    for _, result in feedback:
        assert not result["ok"] and result["error"]["code"] == "tool_batch_limit"
        assert result["error"]["max_operations"] == 4
        assert ("message" not in result["error"]) is compact
    rejected = trace.steps[0]
    assert rejected.provider_tool_request_count == len(rejected.provider_tool_requests) == count
    assert rejected.error["requested_operations"] == count
    assert len(rejected.batch_operations) == count
    assert all(op.outcome == "rejected" and op.observation_sent and not op.evidence and op.delegation is None
               for op in rejected.batch_operations)
    usage = trace.shared_budget
    assert usage.delegated_invocations == 0 and usage.model_sessions == 1
    assert usage.provider_attempts == usage.model_decisions == 3 and usage.corpus_attempts == trace.tool_calls == 2
    assert usage.reported_tokens == 45
    assert {chunk.chunk_id for chunk in trace.coverage.visible_chunks} == {older.chunk_id}
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (older.chunk_id,)


@pytest.mark.parametrize("valid_finish", [True, False])
def test_repeated_bedrock_oversized_batches_preserve_finalization_and_remain_bounded(snapshot, monkeypatch, valid_finish):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, _ = snapshot
    read = tool("list_companies")
    batches = [_large_converse_batch(*([read] * 5), prefix=f"turn-{i}") for i in range(4)]
    responses = batches[:2] + [converse_reply(finish(status="incomplete"), "finish")] if valid_finish else batches
    sessions = stub_bedrock(monkeypatch, [responses])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        shared_limits=SharedLimits(max_delegations=0, max_provider_attempts=4),
        local_limits=RootLimits(max_iterations=6),
        finalization=FinalizationPolicy(finalization_attempts=2, finalization_tool_choice="required"),
    ).run("Look for a disclosure within the budget.")

    requests = sessions[0].requests
    assert len(requests) == len(responses)
    assert [req["toolConfig"]["toolChoice"] for req in requests] == (
        [{"auto": {}}] * 2 + [{"tool": {"name": "finish_research"}}] * (len(responses) - 2)
    )
    assert len(_batch_feedback(requests[2])) == 5
    if valid_finish:
        assert trace.stop_reason == "finished" and trace.analysis_status == "incomplete"
    else:
        assert trace.stop_reason == "budget_exhausted" and trace.budget_stop == "finalization_attempt_limit"
        assert all(result["error"]["finish_only"] for _, result in _batch_feedback(requests[3]))
    assert trace.shared_budget.finalization_provider_attempts == len(responses) - 2
    assert trace.shared_budget.provider_attempts == trace.shared_budget.model_decisions == len(responses)
    assert trace.tool_calls == trace.shared_budget.corpus_attempts == trace.shared_budget.delegated_invocations == 0
    assert not trace.coverage.visible_chunks and not trace.cited_chunk_ids


@pytest.mark.parametrize(("case", "expected"), [
    ("duplicate", "invalid_tool_use"), ("invalid_id", "invalid_tool_use"), ("invalid_input", "invalid_tool_use"),
    ("invalid_number", "invalid_tool_use"),
    ("protocol_limit", "tool_batch_protocol_limit"), ("byte_cap", None),
])
def test_bedrock_unrecoverable_large_batch_fails_closed(snapshot, monkeypatch, case, expected):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient
    from evidenceroute.research.rlm.tools import MAX_PROTOCOL_TOOL_BATCH

    corpus, _ = snapshot
    count = MAX_PROTOCOL_TOOL_BATCH + 1 if case == "protocol_limit" else 5
    response = _large_converse_batch(*([tool("list_companies")] * count))
    blocks = response["output"]["message"]["content"]
    if case == "duplicate":
        blocks[-1]["toolUse"]["toolUseId"] = blocks[0]["toolUse"]["toolUseId"]
    elif case == "invalid_id":
        blocks[-1]["toolUse"]["toolUseId"] = "invalid id"
    elif case == "invalid_input":
        blocks[-1]["toolUse"]["input"] = []
    elif case == "invalid_number":
        blocks[-1]["toolUse"]["input"] = {"limit": float("nan")}
    sessions = stub_bedrock(monkeypatch, [[response]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        local_limits=RootLimits(max_response_bytes=128 if case == "byte_cap" else 6_000),
    ).run("Check the catalog.")
    assert len(sessions[0].requests) == 1 and trace.steps[0].provider_tool_request_count == count
    assert trace.model_attempts[0].error == expected
    assert trace.stop_reason == ("batch_observation_limit" if case == "byte_cap" else "model_error")
    assert trace.tool_calls == trace.shared_budget.corpus_attempts == trace.shared_budget.delegated_invocations == 0
    assert not trace.coverage.visible_chunks


def test_released_child_slots_do_not_consume_protected_comparative_decisions(snapshot, monkeypatch):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient
    from evidenceroute.research.rlm.comparison import scan_research_plan
    from evidenceroute.research.rlm.spans import passage_spans

    corpus, (_, new, _) = snapshot
    pair = SectionTarget(new, RISK)
    chunk = corpus.read_chunks(new, RISK).items[0]
    span = next(iter(passage_spans(chunk.text)))
    draft = {**finding(chunk), "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span}]}
    invalid = {**draft, "evidence": [{"chunk_id": chunk.chunk_id, "span_id": "0:999"}]}
    sessions = stub_bedrock(monkeypatch, [
        [converse_reply(delegate(pair), "delegate"),
         converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "read"),
         converse_reply(tool("list_companies"), "companies"), converse_reply(tool("list_filings"), "filings"),
         converse_reply(finish(findings=[invalid]), "invalid-finish"),
         converse_reply(finish(chunk.chunk_id, findings=[draft]), "corrected-finish")],
        [converse_reply(finish(), "child-finish")],
    ])
    scan = scan_research_plan(corpus, _company_plan(corpus), ("supply",))
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True), comparison_scan=scan,
        shared_limits=SharedLimits(max_delegations=1, max_provider_attempts=12),
        local_limits=RootLimits(max_iterations=6),
        finalization=FinalizationPolicy(child_provider_attempts=6, finalization_attempts=2,
                                        finalization_tool_choice="required"),
    ).run("Compare evidence.", partitions=(pair,))
    requests = sessions[0].requests
    schema = next(tool["toolSpec"]["inputSchema"]["json"] for tool in requests[0]["toolConfig"]["tools"]
                  if tool["toolSpec"]["name"] == "finish_research")
    assert "comparisons" in schema["properties"] and "SpanEvidenceDraft" in schema["$defs"]
    assert trace.shared_budget.released_child_provider_slots == 5
    assert [step.phase for step in trace.steps] == ["investigation"] * 4 + ["finalization"] * 2
    assert trace.steps[4].finalization_reason == "reserved_local_decisions"
    feedback = requests[5]["messages"][-1]["content"][0]["toolResult"]
    assert feedback["toolUseId"] == "invalid-finish" and feedback["content"][0]["json"]["error"]["code"] == "invalid_findings"
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)


@pytest.fixture(scope="module")
def planned_corpus():
    try:
        return CorpusEnvironment()
    except FileNotFoundError:
        pytest.skip("Canonical local corpus unavailable")


def test_corpus_demo_propagates_protected_comparative_policy_across_bedrock_batches(planned_corpus, monkeypatch, capsys):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: planned_corpus)
    limits = {"max_batches": 2, "max_provider_attempts": 72, "max_reported_tokens": 400_000,
              "max_corpus_attempts": 60, "max_delegations": 4, "max_partitions": 8, "deadline_seconds": 600}
    monkeypatch.setattr(Path, "read_bytes", lambda self: json.dumps(limits).encode() if self.name == "fixture-limits.json"
                        else pytest.fail(f"Unexpected read of {self.name}"))
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", "1024")
    main = runpy.run_path(str(examples / "demo_rlm_corpus_research.py"))["main"]
    assignments = []
    for ticker in ("AAPL", "AMD", "NVDA"):
        pairs = [SectionTarget(planned_corpus.list_filings(ticker=ticker, fiscal_year=year).items[0].document_id, RISK)
                 for year in (2023, 2024)]
        assignments.append(delegate(*pairs))
    partial = converse_reply(finish(), "partial")
    partial["stopReason"] = "max_tokens"
    partial["output"]["message"]["content"].append({"reasoningContent": {"reasoningText": {
        "text": "PRIVATE_TRACE_MARKER", "signature": "PRIVATE_SIGNATURE"}}})
    close = converse_reply(finish(), "finish")
    sessions = stub_bedrock(monkeypatch, [
        [converse_batch(*assignments[:2]), partial, close], [partial, close], [partial, close],
        [converse_reply(assignments[2], "delegate"), partial, close], [partial, close],
    ])
    assert main(["--live", "--trace", "--profile", "expanded", "--limits-file", "fixture-limits.json",
                 "--company", "AAPL", "--company", "AMD", "--company", "NVDA"]) == 0
    raw = capsys.readouterr().out
    assert "PRIVATE_TRACE_MARKER" not in raw and "PRIVATE_SIGNATURE" not in raw
    report = json.loads(raw)
    result = report["result"]
    assert result["execution_status"] == "finished" and result["research_status"] == "incomplete"
    policy = report["tree_configuration"]["finalization"]
    assert policy["root_token_headroom"] == policy["child_token_headroom"] == 24000
    assert policy["finalization_timeout_retries"] == 1
    for session in (sessions[0], sessions[3]):
        assert session.requests[0]["toolConfig"]["toolChoice"] == {"tool": {"name": "delegate_research"}}
        assert "Reassess the strongest delivered spans" in str(session.requests[0]["messages"][0])
    for session in sessions:
        investigate, protected = session.requests[-2:]
        assert investigate["inferenceConfig"]["maxTokens"] == 1024
        assert "get_spans" in {tool["toolSpec"]["name"] for tool in investigate["toolConfig"]["tools"]}
        assert protected["inferenceConfig"]["maxTokens"] == 2048
        assert protected["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}
        assert session.timeouts[-2:] == [(20, 20), (60, 60)]
        assert investigate["messages"] == protected["messages"]
    for invocation in result["diagnostics"]:
        first, recovery = invocation["decisions"][-2:]
        assert first["attempts"][0]["stop_reason"] == "max_tokens" and first["attempts"][0]["output_cap"] == 1024
        assert recovery["phase"] == "finalization" and recovery["recovery"] == "truncated_response"
        assert recovery["attempts"][0]["output_cap"] == 2048


@pytest.fixture
def comparative_demo(planned_corpus, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: planned_corpus)
    return runpy.run_path(str(examples / "demo_rlm_comparative.py"))["main"]


def test_comparative_bedrock_native_delegation_batch_and_span_round_trip(
    comparative_demo, planned_corpus, monkeypatch, capsys,
):
    from evidenceroute.research.rlm.spans import passage_spans

    pairs = [tuple(SectionTarget(planned_corpus.list_filings(ticker=ticker, fiscal_year=year).items[0].document_id, RISK)
                   for year in (2023, 2024)) for ticker in ("AAPL", "AMD")]
    chunks = [planned_corpus.get_chunk(key) for key in ("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf")]
    findings = []
    for chunk in chunks:
        span_id = next(span_id for span_id, (start, end) in passage_spans(chunk.text).items()
                       if chunk.text[start:end].startswith("Data centers depend"))
        findings.append({"claim": "Water or power shortages can limit data-center expansion.",
                         "analysis_status": "incomplete", "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span_id}]})
    comparison = {"risk": "Power and water constraints", "same_risk": "yes", "assessments": [
        {"finding_index": i, "relevance": "infrastructure", "entailment": "supports", "rationale": "Capacity constraint.",
         "infrastructure_connection": "Potential power and water shortages could limit data-center capacity."}
        for i in (1, 2)
    ]}
    first = converse_batch(*(delegate(*group) for group in pairs))
    ids = ("delegate:aapl.1", "delegate:amd.2")
    for block, tool_id in zip(first["output"]["message"]["content"], ids):
        block["toolUse"]["toolUseId"] = tool_id
    private_marker = "fixture private reasoning must never appear in the report"
    first["output"]["message"]["content"].insert(0, {
        "reasoningContent": {"reasoningText": {"text": private_marker, "signature": "fixture-signature"}},
    })
    root_finish = finish(*(chunk.chunk_id for chunk in chunks), findings=findings,
                         used=["child-2:finding-1", "child-2:finding-2"], status="incomplete")
    child_finish = {**finish(findings=findings, status="incomplete"), "comparisons": [comparison]}
    bad_finding = {**findings[0], "evidence": [{"chunk_id": chunks[0].chunk_id, "span_id": "0:9999"}]}
    sessions = stub_bedrock(monkeypatch, [
        [first, converse_reply({**root_finish, "comparisons": [comparison]}, "finish:root.2")],
        [converse_reply(finish(status="incomplete"), "finish:aapl.1")],
        [converse_reply(tool("get_chunk", chunk_id=chunks[0].chunk_id), "read:amd.1"),
         converse_reply(tool("get_chunk", chunk_id=chunks[1].chunk_id), "read:amd.2"),
         converse_reply(finish(findings=[bad_finding], status="incomplete"), "finish:amd.3"),
         converse_reply(child_finish, "finish:amd.4")],
    ])
    assert comparative_demo(["--live", "--trace"]) == 0
    raw_report = capsys.readouterr().out
    assert private_marker not in raw_report and "fixture-signature" not in raw_report
    report = json.loads(raw_report)
    assert report["root_stop_reason"] == "finished" and report["research_status"] == "incomplete"
    assert report["shared_usage"]["delegated_invocations"] == 2
    assert report["shared_usage"]["provider_attempts"] == 7 and report["shared_usage"]["corpus_attempts"] == 2
    assert [item["outcome"] for item in report["root_comparisons"]] == ["comparable_both_years"]
    assert [item["outcome"] for item in report["acknowledged_comparisons"]] == ["comparable_both_years"]
    assert {item["chunk_id"] for item in report["original_citations"]} == {chunk.chunk_id for chunk in chunks}
    assert [(issue["invocation_id"], issue["error"]["code"]) for issue in report["diagnostics"]] == [
        ("child-2", "invalid_findings"),
    ]
    assert [request["tool_use_id"] for request in report["trace"]["steps"][0]["provider_tool_requests"]] == list(ids)
    messages = sessions[0].requests[-1]["messages"]
    assert messages[-2] == first["output"]["message"]
    assert [block["toolResult"]["toolUseId"] for block in messages[-1]["content"]] == list(ids)
    amd_correction = sessions[2].requests[-1]["messages"][-1]["content"][0]["toolResult"]
    assert amd_correction["toolUseId"] == "finish:amd.3"
    assert amd_correction["content"][0]["json"]["error"]["code"] == "invalid_findings"
    root_findings = report["trace"]["findings_generated"]
    for chunk, draft, actual in zip(chunks, findings, root_findings):
        item = actual["evidence"][0]
        start, end = passage_spans(chunk.text)[draft["evidence"][0]["span_id"]]
        assert item["quote"] == chunk.text[start:end] and "’" in item["quote"]


def test_expanded_comparative_bedrock_synthesizes_both_companies_and_years(
    comparative_demo, planned_corpus, monkeypatch, capsys,
):
    from evidenceroute.research.rlm.spans import passage_spans

    chunks = [[planned_corpus.get_chunk(key) for key in keys] for keys in (
        ("e4bf3d03444b5f8b4e1614e3", "1b31d472397f437ab1975947"),
        ("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf"),
    )]
    topics = ("AI regulation", "Power and water constraints")
    snippets = ("machine learning and artificial intelligence", "Data centers depend")
    pairs, findings, comparisons = [], [], []
    for index, group in enumerate(chunks):
        pairs.append(tuple(SectionTarget(chunk.document_id, chunk.section_id) for chunk in group))
        findings.append([{
            "claim": topics[index], "analysis_status": "incomplete", "evidence": [{
                "chunk_id": chunk.chunk_id, "span_id": next(span for span, (start, end) in passage_spans(chunk.text).items()
                                                         if snippets[index] in chunk.text[start:end]),
            }],
        } for chunk in group])
        assessments = [{"finding_index": i, "relevance": "infrastructure" if index else "other",
                        "entailment": "supports" if index else "uncertain",
                        "rationale": "Water and power shortages could curb expansion." if index else "Regulatory topic."}
                       for i in (1, 2)]
        if index:
            for item in assessments:
                item["infrastructure_connection"] = "Shortages could limit data-center expansion."
        comparisons.append({"risk": topics[index], "same_risk": "yes" if index else "uncertain", "assessments": assessments})

    used = [f"child-{i}:finding-{j}" for i in (1, 2) for j in (1, 2)]
    selected = [chunk.chunk_id for group in chunks for chunk in group]
    root_finish = finish(*selected, findings=[item for group in findings for item in group], used=used, status="incomplete")
    contradiction = "AMD introduced a new risk in FY2024. The same underlying risk appeared in FY2023."
    root_finish["answer"] = contradiction
    root_finish["comparisons"] = [comparisons[0], {**comparisons[1], "assessments": [
        {**item, "finding_index": item["finding_index"] + 2} for item in comparisons[1]["assessments"]
    ]}]
    truncated = converse_reply(tool("list_filings"), "truncated.reply")
    truncated["stopReason"] = "max_tokens"
    sessions = stub_bedrock(monkeypatch, [
        [converse_reply(delegate(*pairs[0]), "delegate:aapl.1"), truncated,
         converse_reply(delegate(*pairs[1]), "delegate:amd.2"), converse_reply(root_finish, "synthesis.1")],
        *[[converse_batch(*(tool("get_chunk", chunk_id=chunk.chunk_id) for chunk in group)),
           converse_reply({**finish(findings=drafts, status="incomplete"), "comparisons": [comparison],
                           "answer": contradiction}, "compare.1")]
          for group, drafts, comparison in zip(chunks, findings, comparisons)],
    ])
    assert comparative_demo(["--live", "--profile", "expanded", "--trace"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["root_stop_reason"] == "finished" and report["profile"] == "expanded"
    assert report["execution_status"] == report["recursive_comparative_status"] == "success"
    assert report["research_status"] == "incomplete" and report["semantic_review_status"] == "model_assessed"
    assert not report["independent_semantic_verification"]
    assert report["root_cited_years_by_company"] == {"AAPL": [2023, 2024], "AMD": [2023, 2024]}
    assert "introduced" not in report["answer"] and "same underlying infrastructure risk" in report["answer"]
    assert "AI-specific connection unestablished" in report["answer"]
    assert {item["chunk_id"] for item in report["original_citations"]} == set(selected)
    assert report["trace"]["declared_used_finding_ids"] == used
    for key in ("root_comparisons", "acknowledged_comparisons"):
        assert [(item["outcome"], item["both_years_contributed"]) for item in report[key]] == [
            ("insufficient", True), ("comparable_both_years", True),
        ]
    assert report["group_counts"] == {"admitted": 2, "attempted": 2, "completed": 2, "not_admitted": 23,
                                      "failed": 0, "unassigned": 0, "unresolved": 0}
    assert report["shared_usage"]["provider_attempts"] == 8 and report["shared_usage"]["corpus_attempts"] == 4
    assert [step["assignment_stage"] for step in report["trace"]["steps"]] == ["assignment"] * 3 + ["synthesis"]
    root_requests = sessions[0].requests
    for request in root_requests[:3]:
        assert [item["toolSpec"]["name"] for item in request["toolConfig"]["tools"]] == ["delegate_research"]
        assert request["toolConfig"]["toolChoice"] == {"tool": {"name": "delegate_research"}}
    assert root_requests[1]["messages"] == root_requests[2]["messages"]
    assert _batch_feedback(root_requests[2])[0][0] == "delegate:aapl.1"
    assert _batch_feedback(root_requests[3])[0][0] == "delegate:amd.2"
    assert {item["toolSpec"]["name"] for item in root_requests[3]["toolConfig"]["tools"]} >= {"get_chunk", "finish_research"}


@pytest.mark.parametrize("acknowledged", [True, False])
def test_expanded_failure_preserves_only_acknowledged_child_comparison(
    comparative_demo, planned_corpus, monkeypatch, capsys, tmp_path, acknowledged,
):
    from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
    from evidenceroute.research.rlm.spans import passage_spans

    apple_pairs = tuple(SectionTarget(planned_corpus.list_filings(ticker="AAPL", fiscal_year=year).items[0].document_id, RISK)
                        for year in (2023, 2024))
    amd_chunks = [planned_corpus.get_chunk(key) for key in ("043757f52188a8c6fb2a2857", "483314af3b949a29920abbbf")]
    amd_pairs = tuple(SectionTarget(chunk.document_id, chunk.section_id) for chunk in amd_chunks)
    spans = [next(span for span, (start, end) in passage_spans(chunk.text).items() if "Data centers depend" in chunk.text[start:end])
             for chunk in amd_chunks]
    drafts = [{"claim": "Power supply constraints on data-center capacity", "analysis_status": "incomplete",
               "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span}]} for chunk, span in zip(amd_chunks, spans)]
    child_finish = {**finish(findings=drafts, status="incomplete"), "answer": "AMD introduced a new risk.", "comparisons": [
        {"risk": "Customer data-center capacity", "same_risk": "yes", "assessments": [
            {"finding_index": index, "relevance": "infrastructure", "entailment": "supports", "rationale": "Capacity constraint.",
             "infrastructure_connection": "The cited power or water shortages could limit expansion."} for index in (1, 2)
        ]},
    ]}
    profile = COMPARATIVE_PROFILES["expanded"].model_dump()
    profile["local_limits"]["max_tool_calls"] = 2
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    last = (converse_reply(finish("unknown-id"), "root.overrun", usage={"inputTokens": 201_000, "outputTokens": 1})
            if acknowledged else ClientError({"Error": {"Code": "AccessDeniedException"}}, "Converse"))
    stub_bedrock(monkeypatch, [
        [converse_batch(delegate(*apple_pairs), delegate(*amd_pairs)), last],
        [converse_batch(*(tool("search_chunks", query="risk", filters={"document_ids": [pair.document_id]}) for pair in apple_pairs)),
         ReadTimeoutError(endpoint_url="https://bedrock.fixture"), ReadTimeoutError(endpoint_url="https://bedrock.fixture")],
        [converse_batch(*(tool("get_spans", chunk_id=chunk.chunk_id, span_ids=[span]) for chunk, span in zip(amd_chunks, spans))),
         converse_reply(child_finish, "amd.finish")],
    ])
    assert comparative_demo(["--live", "--trace", "--profile-file", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["answer"] is None and not report["original_citations"] and not report["root_comparisons"]
    assert report["recursive_comparative_status"] == report["execution_status"] == report["research_status"] == "incomplete"
    if not acknowledged:
        assert report["partial_result"] is None and report["root_stop_reason"] == "model_error"
        return
    assert report["root_budget_stop"] == "reported_token_limit"
    assert report["shared_usage"]["provider_attempts"] == 7
    assert report["shared_usage"]["attempts_with_missing_usage"] == 2
    partial = report["partial_result"]
    assert partial["status"] == "acknowledged_child_evidence" and partial["root_synthesis"] is False
    assert partial["failed_groups"] == ["AAPL"] and partial["targets_by_access"] == {"none": 48, "partial": 2, "complete": 0}
    assert partial["child_comparisons"][0]["outcome"] == "comparable_both_years"
    assert "introduced" not in partial["answer"] and "same underlying infrastructure risk" in partial["answer"]
    assert "Admitted companies with unresolved assessments. AAPL" in partial["answer"]
    assert "Companies outside the admitted scope (23)." in partial["answer"]
    for item, chunk, span in zip(partial["child_findings"], amd_chunks, spans):
        evidence = item["evidence"][0]
        start, end = passage_spans(chunk.text)[span]
        assert evidence["source"]["source_url"] == chunk.source_url and evidence["quote"] == chunk.text[start:end]


def test_initial_provider_exhaustion_preserves_finish_and_pending_groups(comparative_demo, monkeypatch, capsys, tmp_path):
    from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES

    profile = COMPARATIVE_PROFILES["expanded"].model_dump()
    profile["shared_limits"]["max_provider_attempts"] = 28  # Two root investigation attempts after reservations.
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    sessions = stub_bedrock(monkeypatch, [[
        converse_reply(finish(), "premature.finish"), converse_reply(tool("list_companies"), "premature.read"),
        converse_reply(finish(status="incomplete"), "protected.finish"),
    ]])
    assert comparative_demo(["--live", "--profile-file", str(path), "--trace"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["execution_status"] == "success" and report["recursive_comparative_status"] == "incomplete"
    assert report["initial_assignment_stop"] == "investigation_attempt_limit"
    assert report["group_counts"]["unassigned"] == 2 and report["group_counts"]["attempted"] == 0
    assert report["shared_usage"]["provider_attempts"] == 3 and report["shared_usage"]["corpus_attempts"] == 0
    assert report["shared_usage"]["finalization_provider_attempts"] == 1
    assert [(_batch_feedback(request)[0][0], _batch_feedback(request)[0][1]["error"]["code"])
            for request in sessions[0].requests[1:]] == [(request_id, "initial_assignment_required")
                                                         for request_id in ("premature.finish", "premature.read")]
    assert sessions[0].requests[-1]["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}


@pytest.mark.parametrize("detailed", [False, True])
def test_comparative_reports_provider_decode_failure_without_private_content(comparative_demo, monkeypatch, capsys, detailed):
    response = converse_reply(tool("list_companies"), "unused")
    response["stopReason"] = "malformed_tool_use"
    response["output"]["message"]["content"] = [{"text": "fixture private provider output"}]
    stub_bedrock(monkeypatch, [[response]])
    assert comparative_demo(["--live", *(["--trace"] if detailed else [])]) == 1
    raw_report = capsys.readouterr().out
    assert "fixture private provider output" not in raw_report
    report = json.loads(raw_report)
    assert report["root_stop_reason"] == "model_error" and not report["root_budget_stop"]
    assert report["diagnostics"][0]["provider_errors"] == [{
        "code": "unexpected_stop_reason", "response_received": True, "provider_stop_reason": "malformed_tool_use",
    }]
    assert report["shared_usage"]["delegated_invocations"] == report["shared_usage"]["corpus_attempts"] == 0
    assert ("trace" in report) is detailed


@pytest.mark.parametrize("missing_year", [False, True])
def test_live_planned_research_stubbed_bedrock(planned_corpus, monkeypatch, capsys, missing_year):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: planned_corpus)
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", "1024")
    demo = runpy.run_path(str(examples / "demo_rlm_planned_research.py"))
    pairs = [tuple(SectionTarget(planned_corpus.list_filings(ticker=ticker, fiscal_year=year).items[0].document_id, RISK)
                   for year in (2023, 2024)) for ticker in ("AAPL", "AMD")]
    chunks = [[planned_corpus.read_chunks(pair.document_id, pair.section_id, limit=1).items[0] for pair in group]
              for group in pairs]
    findings = [[finding(chunk, quote=chunk.text[:80].strip(), status="incomplete") for chunk in group]
                for group in chunks]
    selected = [chunk.chunk_id for group in chunks for chunk in group]
    used = [f"child-{i}:finding-{j}" for i in (1, 2) for j in (1, 2)]
    if missing_year:
        findings[1].pop()
        selected.pop()
        used.pop()
    sessions = stub_bedrock(monkeypatch, [
        [converse_batch(*(delegate(*group) for group in pairs)),
         converse_reply(finish(*selected, used=used, status="incomplete"), "root-finish")],
        *[[converse_batch(*(tool("get_chunk", chunk_id=chunk.chunk_id) for chunk in group)),
           converse_reply(finish(findings=items, status="incomplete"), "child-finish")]
          for group, items in zip(chunks, findings)],
    ])

    assert demo["main"](["--live", "--trace", "--max-tool-calls", "2"]) == int(missing_year)
    report = json.loads(capsys.readouterr().out)
    assert report["intent_interpretation"] == "scripted" and not report["semantic_claims_verified"]
    assert report["structural_status"] == ("incomplete" if missing_year else "success")
    assert report["structural_criteria"]["companies"]["AMD"]["both_years_acknowledged"] is not missing_year
    assert report["structural_criteria"]["companies"]["AMD"]["both_years_cited_from_declared_used_findings"] is not missing_year
    assert report["stop_reason"] == "finished" and report["answer_status"] == "incomplete"
    assert (report["requested_companies"], report["required_targets"], len(report["admitted_scope"])) == (25, 50, 4)
    assert report["access_counts_by_target"] == {"none": 46, "partial": 4}
    assert report["unique_whole_chunks_delivered"] == 4 and report["section_characters_delivered"] == 0
    assert report["trace"]["declared_used_finding_ids"] == used
    assert {ref["chunk_id"] for ref in report["original_citations"]} == set(selected)
    assert report["provider_usage"]["provider_attempts"] == report["shared_usage"]["provider_attempts"] == 6
    assert report["provider_usage"]["tokens"]["input_tokens"]["reported_total"] == 66
    assert report["provider_usage"]["reported_provider_latency_seconds_sum"] == 1.5
    assert report["provider"]["max_retries"] == report["provider_usage"]["provider_retries"] == 0
    assert report["limits"]["shared"] == {
        "max_depth": 1, "max_delegations": 2, "max_provider_attempts": 18, "max_corpus_attempts": 12,
        "max_partitions": 4, "max_reported_tokens": 100_000, "deadline_seconds": 180.0,
    }
    assert report["limits"]["local"]["max_iterations"] == 6
    assert report["limits"]["local"]["deadline_seconds"] == 180
    assert report["limits"]["local"]["max_response_bytes"] == 6_000
    assert report["limits"]["finalization"]["child_provider_attempts"] == 6
    assert report["limits"]["finalization"]["finalization_attempts"] == 2
    assert [result["toolResult"]["toolUseId"] for result in sessions[0].requests[1]["messages"][-1]["content"]] == [
        "batch-1", "batch-2",
    ]
    for index, session in enumerate(sessions):
        initial, final = session.requests
        expected = {"tool": {"name": "delegate_research"}} if index == 0 else {"auto": {}}
        assert initial["toolConfig"]["toolChoice"] == expected
        assert final["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}
        assert (initial["inferenceConfig"]["maxTokens"], final["inferenceConfig"]["maxTokens"]) == (1024, 2048)
    for session in sessions[1:]:
        prompt = session.requests[0]["messages"][0]["content"][0]["text"]
        assert "Choose your own focused queries" in prompt and "A search miss does not show" in prompt
        assert "Do not insert ellipses" in prompt
        assert "4 provider attempt(s) for this invocation" in session.requests[0]["system"][0]["text"]


@pytest.mark.parametrize(("bad_quote", "corrected"), [
    ("Supply interruptions...operations.", True),
    ("Supply interruptions affect Operations.", True),
    (" Supply interruptions affect operations.", True),
    ("Supply interruptions affect operations!", False),
])
def test_protected_finish_quote_feedback_and_bounded_correction(snapshot, monkeypatch, bad_quote, corrected):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (_, new, _) = snapshot
    pair = SectionTarget(new, RISK)
    chunk = corpus.read_chunks(new, RISK).items[0]
    invalid = finding(chunk)
    invalid["evidence"].append({"chunk_id": chunk.chunk_id, "quote": bad_quote})
    bad_finish = finish(chunk.chunk_id, findings=[finding(chunk), invalid], status="incomplete")
    good_finish = finish(chunk.chunk_id, findings=[finding(chunk)], status="incomplete")
    sessions = stub_bedrock(monkeypatch, [[
        converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "read-id"),
        converse_reply(bad_finish, "first-finish"),
        converse_reply(good_finish if corrected else bad_finish, "second-finish"),
    ]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        shared_limits=SharedLimits(max_delegations=0, max_provider_attempts=6),
        local_limits=RootLimits(max_tool_calls=1, max_iterations=6, max_response_bytes=6_000),
        finalization=FinalizationPolicy(finalization_attempts=2, finalization_tool_choice="required"),
    ).run("Collect one risk excerpt.", partitions=(pair,))

    first_finish, correction = sessions[0].requests[1:]
    for payload in (first_finish, correction):
        assert payload["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}
        assert [tool["toolSpec"]["name"] for tool in payload["toolConfig"]["tools"]] == ["finish_research"]
    assert correction["messages"][:-2] == first_finish["messages"]
    assert correction["messages"][-2]["content"][0]["toolUse"]["toolUseId"] == "first-finish"
    result = correction["messages"][-1]["content"][0]["toolResult"]
    assert result["toolUseId"] == "first-finish"
    body = result["content"][0]["json"]
    assert body == {"ok": False, "error": {"code": "invalid_findings", "message": (
        f"Finding 2, evidence 2 (chunk {chunk.chunk_id}): copy a contiguous quote exactly from "
        "the delivered chunk text, including punctuation and spacing, or omit it."
    )}}
    assert len(sessions[0].requests) == trace.shared_budget.provider_attempts == 3
    assert trace.shared_budget.finalization_provider_attempts == 2 and trace.shared_budget.corpus_attempts == 1
    assert trace.steps[1].observation_sent
    assert trace.analysis_status == "incomplete" and trace.coverage_status == "partial"
    if corrected:
        assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
        assert len(trace.findings) == 1
        item = trace.findings[0].evidence[0]
        assert (item.quote, item.source.document_id, item.source.source_url) == (QUOTE, new, chunk.source_url)
    else:
        assert trace.stop_reason == "budget_exhausted" and trace.budget_stop == "finalization_attempt_limit"
        assert trace.findings == () and trace.cited_chunk_ids == ()


@pytest.mark.parametrize("tool_limit", [1, 4])
def test_live_recursive_delegations_in_one_bedrock_response(snapshot, recursive_demo, monkeypatch, capsys, tool_limit):
    corpus, (old, new, _) = snapshot
    old_chunk = corpus.read_chunks(old, RISK).items[0]
    new_chunk = corpus.read_chunks(new, RISK).items[0]
    two_years = tool_limit != 1
    batch = converse_batch(delegate(SectionTarget(old, RISK)), delegate(SectionTarget(new, RISK)))
    actions = [
        [batch, converse_reply(finish(
            old_chunk.chunk_id, *([new_chunk.chunk_id] if two_years else []),
            used=["child-1:finding-1", *(["child-2:finding-1"] if two_years else [])],
        ), "finish")],
        [converse_reply(tool("get_chunk", chunk_id=old_chunk.chunk_id), "read-old"),
         converse_reply(finish(findings=[finding(old_chunk)]), "old-finish")],
    ]
    if two_years:
        actions.append([
            converse_reply(tool("get_chunk", chunk_id=new_chunk.chunk_id), "read-new"),
            converse_reply(finish(findings=[finding(new_chunk)]), "new-finish"),
        ])
    sessions = stub_bedrock(monkeypatch, actions)
    assert recursive_demo(["--live", "--trace", "--max-tool-calls", str(tool_limit)]) == (0 if two_years else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == ("success" if two_years else "incomplete")
    assert report["stop_reason"] == "finished"
    root_messages = sessions[0].requests[-1]["messages"]
    assert root_messages[-2] == batch["output"]["message"]
    assert len(root_messages) == 3
    results = [block["toolResult"] for block in root_messages[-1]["content"]]
    assert [item["toolUseId"] for item in results] == ["batch-1", "batch-2"]
    old_result, new_result = (item["content"][0]["json"] for item in results)
    assert old_result["result"]["child_invocation_id"] == "child-1"
    if two_years:
        assert new_result["result"]["child_invocation_id"] == "child-2"
        assert old in sessions[1].requests[0]["messages"][0]["content"][0]["text"]
        assert new in sessions[2].requests[0]["messages"][0]["content"][0]["text"]
        assert len(report["original_citations"]) == 2
    else:
        assert new_result["error"]["code"] == "tool_call_limit"
        assert report["trace"]["tool_operations"][1]["outcome"] == "skipped"
        assert len(sessions) == 2 and len(report["original_citations"]) == 1
    count = 2 if two_years else 1
    usage = report["shared_usage"]
    assert usage["corpus_attempts"] == usage["partition_assignments"] == usage["delegated_invocations"] == count
    assert usage["model_decisions"] == report["provider_usage"]["provider_attempts"] == 2 + count * 2
    assert usage["reported_tokens"] == (2 + count * 2) * 15


@pytest.mark.parametrize("case", ["invalid_peer", "shared_limit", "mixed_finish", "oversized", "envelope", "duplicate"])
def test_batch_is_bounded_and_only_delivered_evidence_is_citable(snapshot, monkeypatch, case):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (old, new, _) = snapshot
    first, second = [corpus.read_chunks(doc, RISK).items[0] for doc in (old, new)]
    first_action, second_action = [tool("get_chunk", chunk_id=chunk.chunk_id) for chunk in (first, second)]
    if case == "invalid_peer":
        second_action = tool("get_chunk", chunk_id=5)
    if case == "mixed_finish":
        second_action = finish()
    initial = converse_batch(first_action, second_action)
    if case == "duplicate":
        initial["output"]["message"]["content"][1]["toolUse"]["toolUseId"] = "batch-1"
    replies = [initial]
    if case not in {"envelope", "duplicate"}:
        if case == "oversized":
            replies.append(converse_reply(finish(second.chunk_id), "unseen"))
        ids = () if case == "mixed_finish" else (first.chunk_id,)
        replies.append(converse_reply(finish(*ids), "final"))
    sessions = stub_bedrock(monkeypatch, [replies])
    limits = RootLimits(max_response_bytes={"oversized": 750, "envelope": 128}.get(case, 6_000))
    shared = SharedLimits(max_corpus_attempts=1 if case == "shared_limit" else 8)
    trace = RecursiveResearch(corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
                              local_limits=limits, shared_limits=shared).run("Check a batch.")
    step = trace.steps[0]
    assert step.provider_tool_request_count == 2
    if case in {"duplicate", "envelope"}:
        assert trace.stop_reason == ("model_error" if case == "duplicate" else "batch_observation_limit")
        assert trace.shared_budget.corpus_attempts == 0 and len(sessions[0].requests) == 1
        return
    assert trace.stop_reason == "finished"
    outcomes = sessions[0].requests[1]["messages"][-1]["content"]
    bodies = [item["toolResult"]["content"][0]["json"] for item in outcomes]
    expected_error = {
        "invalid_peer": "invalid_arguments", "shared_limit": "corpus_attempt_limit",
        "mixed_finish": "mixed_finish_batch", "oversized": "batch_observation_limit",
    }[case]
    assert bodies[1]["error"]["code"] == expected_error
    assert bodies[0]["ok"] == (case != "mixed_finish")
    assert trace.shared_budget.corpus_attempts == (0 if case == "mixed_finish" else 1 if case == "shared_limit" else 2)
    assert {ref.chunk_id for ref in trace.coverage.visible_chunks} == (set() if case == "mixed_finish" else {first.chunk_id})
    if case == "oversized":
        assert step.batch_operations[1].delivery_error == "batch_observation_limit"
        assert step.batch_operations[1].evidence[0].chunk_id == second.chunk_id
        assert trace.steps[1].error["code"] == "unseen_citation"


def test_child_can_exhaust_provider_budget_before_second_batch_operation(
    snapshot, recursive_demo, monkeypatch, capsys,
):
    corpus, (old, new, _) = snapshot
    chunk = corpus.read_chunks(old, RISK).items[0]
    sessions = stub_bedrock(monkeypatch, [
        [converse_batch(delegate(SectionTarget(old, RISK)), delegate(SectionTarget(new, RISK)))],
        [converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "old-read"),
         converse_reply(finish(findings=[finding(chunk)]), "old-finish")],
    ])
    assert recursive_demo(["--live", "--max-provider-attempts", "3", "--finalization-attempts", "0"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["stop_reason"] == "budget_exhausted"
    assert report["trace"]["budget_stop"] == "provider_attempt_limit"
    assert [len(session.requests) for session in sessions] == [1, 2]
    first, second = report["trace"]["tool_operations"]
    assert first["outcome"] == "executed" and not first["observation_acknowledged"]
    assert second["outcome"] == "skipped" and second["error"]["code"] == "provider_attempt_limit"
    assert report["trace"]["evidence_delivered"]["delegated_excerpts"] == []
    assert report["shared_usage"]["delegated_invocations"] == 1


@pytest.mark.parametrize("protected", [False, True], ids=["reproduce-starvation", "protected-finalization"])
def test_recursive_demo_with_retrieval_pressure(snapshot, recursive_demo, monkeypatch, capsys, protected):
    corpus, (old, new, _) = snapshot
    old_chunk, new_chunk = (corpus.read_chunks(doc, RISK).items[0] for doc in (old, new))
    delegations = converse_batch(delegate(SectionTarget(old, RISK)), delegate(SectionTarget(new, RISK)))
    search = tool("search_chunks", query="supply", limit=1)
    if protected:
        root_finish = finish(old_chunk.chunk_id, new_chunk.chunk_id,
                             used=["child-1:finding-1", "child-2:finding-1"])
        root_replies = [delegations, converse_reply(root_finish, "root-finish")]

        def child_replies(chunk):
            return [converse_batch(search, search), converse_reply(search, "search-3"),
                    converse_reply(search, "search-4"),
                    converse_reply(finish(findings=[finding(chunk)]), "child-finish")]
        first, second = child_replies(old_chunk), child_replies(new_chunk)
    else:
        root_replies = [delegations, converse_reply(search, "blocked-root-search")]
        first = [converse_reply(search, f"search-{i}") for i in range(1, 5)]
        first.append(converse_reply(finish(findings=[finding(old_chunk)]), "child-finish"))
        second = [converse_reply(search, f"search-{i}") for i in range(1, 6)]
    sessions = stub_bedrock(monkeypatch, [root_replies, first, second])
    flags = ["--live", "--trace"] + ([] if protected else ["--finalization-attempts", "0"])
    assert recursive_demo(flags) == (0 if protected else 1)
    report = json.loads(capsys.readouterr().out)
    usage = report["shared_usage"]
    assert usage["corpus_attempts"] == 8 and usage["delegated_invocations"] == 2
    assert [len(session.requests) for session in sessions] == ([2, 4, 4] if protected else [2, 5, 5])
    assert usage["provider_attempts"] == (10 if protected else 12)
    assert usage["model_decisions"] == (10 if protected else 13)
    assert usage["finalization_provider_attempts"] == (3 if protected else 0)
    if not protected:
        assert report["stop_reason"] == "budget_exhausted"
        assert report["trace"]["budget_stop"] == "provider_attempt_limit"
        assert report["trace"]["delegations"][1]["child"]["stop_reason"] == "tool_call_limit"
        assert report["trace"]["tool_operations"][-1]["error"]["code"] == "corpus_attempt_limit"
        assert report["original_citations"] == []
        return

    assert report["status"] == "success"
    assert {item["chunk_id"] for item in report["original_citations"]} == {old_chunk.chunk_id, new_chunk.chunk_id}
    for session, expected_ids in zip(sessions, [("batch-1", "batch-2"), ("search-4",), ("search-4",)]):
        final_request = session.requests[-1]
        tools = [item["toolSpec"]["name"] for item in final_request["toolConfig"]["tools"]]
        assert tools == ["finish_research"]
        assert "Finalization only" in final_request["system"][0]["text"]
        assert [item["toolResult"]["toolUseId"]
                for item in final_request["messages"][-1]["content"]] == list(expected_ids)
        assert "local tool slot(s)" in session.requests[0]["system"][0]["text"]
        assert "shared corpus attempt(s)" in session.requests[0]["system"][0]["text"]


@pytest.mark.parametrize("mode", ["search", "batch", "invalid_quote", "unseen_citation", "provider_failure",
                                  "required_choice_rejected"])
def test_finalization_rejects_work_and_preserves_validation(snapshot, monkeypatch, mode):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (old, _, _) = snapshot
    chunk = corpus.read_chunks(old, RISK).items[0]
    read = tool("get_chunk", chunk_id=chunk.chunk_id)
    first = converse_reply(read, "first-read")
    if mode == "search":
        last = converse_reply(tool("search_chunks", query="supply"), "unexpected-search")
        expected = "finalization_only"
    elif mode == "batch":
        last = converse_batch(read, delegate(SectionTarget(old, RISK)))
        expected = "finalization_only"
    elif mode == "invalid_quote":
        last = converse_reply(finish(findings=[finding(chunk, quote="unseen quote")]), "invalid-quote")
        expected = "invalid_findings"
    elif mode == "unseen_citation":
        last = converse_reply(finish("unseen-id"), "unseen-citation")
        expected = "unseen_citation"
    elif mode == "provider_failure":
        last = ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse")
        expected = "model_error"
    else:
        last = ClientError({"Error": {"Code": "ValidationException", "Message": "private provider detail"}},
                           "Converse")
        expected = "model_error"
    sessions = stub_bedrock(monkeypatch, [[first, last]])
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", "1024")
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        local_limits=RootLimits(max_tool_calls=1), shared_limits=SharedLimits(max_delegations=0),
        finalization=FinalizationPolicy(finalization_tool_choice="required", finalization_max_output_tokens=1792),
    ).run("Check a risk.")
    step = trace.steps[-1]
    assert step.phase == "finalization" and step.finalization_reason == "tool_call_limit"
    if mode == "batch":
        assert [op.error["code"] for op in step.batch_operations] == [expected, expected]
    else:
        assert step.error["code"] == expected
    assert trace.answer is None and trace.findings == () and trace.cited_chunk_ids == ()
    assert trace.stop_reason == ("model_error" if expected == "model_error" else "budget_exhausted")
    assert trace.shared_budget.corpus_attempts == 1 and trace.shared_budget.delegated_invocations == 0
    assert trace.shared_budget.finalization_provider_attempts == 1
    investigation, finalization = sessions[0].requests
    assert investigation["toolConfig"]["toolChoice"] == {"auto": {}}
    assert investigation["inferenceConfig"]["maxTokens"] == 1024
    assert finalization["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}
    assert [tool["toolSpec"]["name"] for tool in finalization["toolConfig"]["tools"]] == ["finish_research"]
    assert finalization["inferenceConfig"]["maxTokens"] == 1792
    assert "Finalization only" in finalization["system"][0]["text"]
    assert "retry with that limit" not in finalization["system"][0]["text"]
    assert finalization["messages"][-1]["content"][0]["toolResult"]["toolUseId"] == "first-read"
    if mode == "required_choice_rejected":
        assert step.model_attempts[0].error == "aws_required_finish_validation"
        assert "private provider detail" not in repr(trace)


@pytest.mark.parametrize("batched", [False, True], ids=["single-result", "batched-results"])
def test_truncated_investigation_replays_only_valid_bedrock_history(snapshot, monkeypatch, batched):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (old, new, _) = snapshot
    chunk = corpus.read_chunks(old, RISK).items[0]
    other = corpus.read_chunks(new, RISK).items[0]
    read = tool("get_chunk", chunk_id=chunk.chunk_id)
    initial = converse_batch(read, tool("list_sections", document_id=old)) if batched else converse_reply(read, "read")
    truncated = converse_reply(tool("get_chunk", chunk_id=other.chunk_id), "truncated-read",
                               usage={"inputTokens": 20, "outputTokens": 1024})
    truncated["stopReason"] = "max_tokens"
    if batched:
        # With no assistant message, acknowledgement must wait for the replay response.
        truncated["output"]["message"]["content"] = []
    final = converse_reply(finish(chunk.chunk_id, findings=[finding(chunk)]), "truncated-read")
    sessions = stub_bedrock(monkeypatch, [[initial, truncated, final]])
    monkeypatch.setenv("CLAUDE_MAX_OUTPUT_TOKENS", "1024")
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        shared_limits=SharedLimits(max_delegations=0), finalization=FinalizationPolicy(),
    ).run("Check a risk.")

    _, investigation, recovery = sessions[0].requests
    assert recovery["messages"] == investigation["messages"]
    assert recovery["messages"][1] == initial["output"]["message"]
    expected_ids = ["batch-1", "batch-2"] if batched else ["read"]
    assert [block["toolResult"]["toolUseId"] for block in recovery["messages"][-1]["content"]] == expected_ids
    assert investigation["inferenceConfig"]["maxTokens"] == 1024
    assert recovery["inferenceConfig"]["maxTokens"] == 2048
    assert recovery["toolConfig"]["toolChoice"] == {"auto": {}}
    assert trace.steps[1].error["code"] == "truncated_response"
    assert trace.steps[1].provider_tool_requests == ()
    assert trace.steps[2].phase == "finalization" and trace.steps[2].finalization_reason == "truncated_response"
    assert trace.shared_budget.corpus_attempts == len(expected_ids)
    assert trace.shared_budget.provider_attempts == 3
    assert trace.shared_budget.investigation_provider_attempts == 2
    assert trace.shared_budget.finalization_provider_attempts == 1
    assert trace.shared_budget.reported_tokens == 1074
    assert trace.model_attempts[1].response_received and trace.model_attempts[1].output_tokens == 1024
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
    assert {ref.chunk_id for ref in trace.coverage.visible_chunks} == {chunk.chunk_id}


@pytest.mark.parametrize("nested", [False, True])
def test_truncated_first_protected_finish_reuses_second_slot(snapshot, monkeypatch, nested):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (doc, _, _) = snapshot
    pair = SectionTarget(doc, RISK)
    chunk = corpus.read_chunks(doc, RISK, limit=1).items[0]
    partial = converse_reply(finish("unseen"), "partial")
    partial["stopReason"] = "max_tokens"
    reader = [converse_reply(tool("get_chunk", chunk_id=chunk.chunk_id), "read"), partial,
              converse_reply(finish(chunk.chunk_id, findings=[finding(chunk)]), "reader-finish")]
    if nested:
        ancestor = [converse_reply(delegate(pair), "delegate"),
                    converse_reply(finish(chunk.chunk_id, used=["child-1:finding-1"]), "root-finish")]
    sessions = stub_bedrock(monkeypatch, [ancestor, reader] if nested else [reader])
    trace = RecursiveResearch(corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
                              local_limits=RootLimits(max_tool_calls=1), shared_limits=SharedLimits(max_delegations=1),
                              finalization=FinalizationPolicy(finalization_attempts=2)).run("Review risk.", partitions=(pair,))
    read_trace = trace.steps[0].delegation.child if nested else trace
    assert trace.stop_reason == read_trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
    assert read_trace.steps[-2].phase == read_trace.steps[-1].phase == "finalization"
    assert read_trace.steps[-1].recovery == "truncated_response"
    assert read_trace.steps[-1].model_attempts[0].output_cap == 2048
    assert sessions[-1].requests[1]["messages"] == sessions[-1].requests[2]["messages"]
    assert read_trace.steps[0].observation_sent and read_trace.steps[0].observation_attempted


def test_comparative_delegation_batch_compacts_results_without_losing_sibling(snapshot, monkeypatch):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient
    from evidenceroute.research.rlm.comparison import scan_research_plan
    from evidenceroute.research.rlm.spans import passage_spans

    corpus, (old, new, _) = snapshot
    pairs = [SectionTarget(doc, RISK) for doc in (old, new)]
    replies, keys = [], []
    for pair in pairs:
        chunk = corpus.read_chunks(pair.document_id, RISK, limit=1).items[0]
        keys.append(chunk.chunk_id)
        span = next(iter(passage_spans(chunk.text)))
        drafts = [{"claim": str(i) + "x" * 575, "analysis_status": "incomplete",
                   "evidence": [{"chunk_id": chunk.chunk_id, "span_id": span}]} for i in range(3)]
        replies.append([converse_reply(tool("get_spans", chunk_id=chunk.chunk_id, span_ids=[span]), "read"),
                        converse_reply(finish(findings=drafts), "finish")])
    sessions = stub_bedrock(monkeypatch, [[converse_batch(*(delegate(pair) for pair in pairs)),
        converse_reply(finish(*keys, used=["child-1:finding-1", "child-2:finding-1"]), "root-finish")], *replies])
    trace = RecursiveResearch(corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
                              comparison_scan=scan_research_plan(corpus, _company_plan(corpus), ("operations",)),
                              local_limits=RootLimits(max_response_bytes=4400)).run("Review risk.", partitions=tuple(pairs))
    assert trace.stop_reason == "finished" and set(trace.cited_chunk_ids) == set(keys)
    step = trace.steps[0]
    assert step.envelope_bytes <= 4400
    edges = [op.delegation for op in step.operations]
    assert sum(edge.full_observation_bytes for edge in edges) > 4400
    assert all(edge.full_observation_bytes <= 4400 for edge in edges)
    for index, ((_, response), edge) in enumerate(zip(_batch_feedback(sessions[0].requests[-1]), edges)):
        assert response["ok"] and response["result"]["omitted_findings"] > 0
        assert len(edge.child.findings) == 3 and len(edge.delivered_finding_ids) in (1, 2)
        excerpt = response["result"]["findings"][0]["evidence"][0]
        assert excerpt["quote"] == edge.child.findings[0].evidence[0].quote
        assert excerpt["source"]["chunk_id"] == keys[index]


@pytest.mark.parametrize("failure", ["truncated", "invalid_quote", "unseen_citation"])
def test_direct_messages_recovery_uses_named_finish_and_remains_bounded(snapshot, failure):
    corpus, (old, new, _) = snapshot
    chunk = corpus.read_chunks(old, RISK).items[0]
    other = corpus.read_chunks(new, RISK).items[0]
    partial = {**message("get_chunk", {"chunk_id": other.chunk_id}, ident="partial"), "stop_reason": "max_tokens"}
    if failure == "truncated":
        last = partial
    elif failure == "invalid_quote":
        last = message("finish_research", {"answer": "answer", "findings": [finding(chunk, quote="invented")]},
                       ident="finish")
    else:
        last = message("finish_research", {"answer": "answer", "cited_chunk_ids": [other.chunk_id]}, ident="finish")
    transport = Transport(message("get_chunk", {"chunk_id": chunk.chunk_id}), partial, last)
    trace = RecursiveResearch(
        corpus, lambda _: ClaudeRootClient(transport, config=ClaudeConfig(max_output_tokens=73)),
        shared_limits=SharedLimits(max_delegations=0),
        finalization=FinalizationPolicy(finalization_tool_choice="required", finalization_max_output_tokens=151),
    ).run("Check a risk.")

    initial, investigation, recovery = [payload for payload, _ in transport.calls]
    assert initial["tool_choice"] == investigation["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert recovery["tool_choice"] == {"type": "tool", "name": "finish_research", "disable_parallel_tool_use": True}
    assert investigation["max_tokens"] == 73 and recovery["max_tokens"] == 151
    assert investigation["messages"] == recovery["messages"]
    assert recovery["messages"][-1]["content"][0]["tool_use_id"] == "use-1"
    assert trace.shared_budget.corpus_attempts == 1 and trace.shared_budget.finalization_provider_attempts == 1
    assert trace.answer is None and trace.cited_chunk_ids == () and trace.findings == ()
    if failure == "truncated":
        assert trace.stop_reason == "model_error" and trace.model_attempts[-1].error == "truncated_response"
    else:
        assert trace.budget_stop == "finalization_attempt_limit"
        assert trace.steps[-1].error["code"] == {"invalid_quote": "invalid_findings",
                                                 "unseen_citation": "unseen_citation"}[failure]


@pytest.mark.parametrize("constraint", ["tokens", "deadline", "unreserved"])
def test_truncation_does_not_override_remaining_budget(snapshot, constraint):
    corpus, _ = snapshot
    now = [0.0]

    class TruncationTransport(Transport):
        def send(self, payload, *, timeout_seconds):
            if constraint == "deadline":
                now[0] = 1.0
            return super().send(payload, timeout_seconds=timeout_seconds)

    partial = {**message("list_companies", {}, usage={"input_tokens": 2, "output_tokens": 1}),
               "stop_reason": "max_tokens"}
    transport = TruncationTransport(partial)
    trace = RecursiveResearch(
        corpus, lambda _: ClaudeRootClient(transport, clock=lambda: now[0], config=ClaudeConfig(max_retries=0)),
        clock=lambda: now[0], shared_limits=SharedLimits(max_delegations=0, deadline_seconds=1.0,
                                                       max_reported_tokens=3 if constraint == "tokens" else 100),
        finalization=None if constraint == "unreserved" else FinalizationPolicy(),
    ).run("Check a risk.")
    assert trace.stop_reason == {"tokens": "budget_exhausted", "deadline": "deadline", "unreserved": "model_error"}[constraint]
    if constraint == "tokens":
        assert trace.budget_stop == "reported_token_limit"
    assert trace.answer is None and trace.shared_budget.reported_tokens == 3
    assert len(transport.calls) == 1 and trace.shared_budget.finalization_provider_attempts == 0
    assert trace.shared_budget.corpus_attempts == 0


def test_live_demo_recovers_root_but_reports_failed_child_as_incomplete(snapshot, recursive_demo, monkeypatch, capsys):
    corpus, (old, new, _) = snapshot
    old_chunk = corpus.read_chunks(old, RISK).items[0]
    new_chunk = corpus.read_chunks(new, RISK).items[0]
    search = tool("search_chunks", query="supply", limit=1)
    read_new = tool("get_chunk", chunk_id=new_chunk.chunk_id)
    partial = converse_reply(search, "partial", usage={"inputTokens": 11, "outputTokens": 1024})
    partial["stopReason"] = "max_tokens"
    sessions = stub_bedrock(monkeypatch, [
        [converse_batch(delegate(SectionTarget(old, RISK)), delegate(SectionTarget(new, RISK))),
         converse_reply(read_new, "root-read"), partial,
         converse_reply(finish(old_chunk.chunk_id, new_chunk.chunk_id, used=["child-1:finding-1"],
                               status="incomplete"), "root-finish")],
        [converse_reply(search, "old-search-1"), converse_reply(search, "old-search-2"),
         converse_reply(finish(findings=[finding(old_chunk)]), "old-finish")],
        [converse_reply(read_new, "new-read"), converse_reply(search, "new-search-1"),
         converse_reply(search, "new-search-2"), converse_reply(search, "new-prohibited")],
    ])
    assert recursive_demo(["--live", "--trace", "--finalization-tool-choice", "required"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["limits"]["finalization"]["finalization_tool_choice"] == "required"
    assert report["stop_reason"] == "finished" and report["status"] == "incomplete"
    assert report["trace"]["finalization_reason"] == "truncated_response"
    assert report["shared_usage"]["provider_attempts"] == 11
    assert report["shared_usage"]["corpus_attempts"] == 6
    assert report["shared_usage"]["finalization_provider_attempts"] == 2
    assert report["shared_usage"]["released_child_provider_slots"] == 1
    old_result, new_result = report["trace"]["delegations"]
    assert all(result["observation_acknowledged"] for result in (old_result, new_result))
    assert old_result["delivered_finding_ids"] == ["child-1:finding-1"]
    assert not new_result["delivered_finding_ids"]
    assert new_result["child"]["budget_stop"] == "finalization_attempt_limit"
    assert not report["criteria"]["original_citations_from_used_findings_for_both_years"]
    assert [ref["direct_chunk_text_delivered"] for ref in report["original_citations"]] == [False, True]
    assert all(session.requests[-1]["toolConfig"]["toolChoice"] == {"tool": {"name": "finish_research"}}
               for session in (sessions[0], sessions[2]))


def test_child_allocations_are_isolated_and_unused_slots_return_on_failure():
    budget = SharedBudget(SharedLimits(max_provider_attempts=12, max_delegations=2),
                          finalization=FinalizationPolicy())
    with budget.invocation_allocation(root=True):
        root = budget.active_allocation
        budget.start_provider_attempt()
        budget.reserve_child(1, 1)
        with pytest.raises(RuntimeError, match="setup failed"):
            with budget.invocation_allocation(root=False):
                budget.start_provider_attempt()
                raise RuntimeError("setup failed")
        assert budget.active_allocation is root and root.investigation_left == 5
        assert budget.released_child_provider_slots == 3
        budget.reserve_child(1, 1)
        with budget.invocation_allocation(root=False):
            for _ in range(3):
                budget.start_provider_attempt()
            with pytest.raises(BudgetExceeded, match="investigation_attempt_limit"):
                budget.start_provider_attempt()
            budget.active_allocation.finalizing = True
            budget.start_provider_attempt()
            with pytest.raises(BudgetExceeded, match="finalization_attempt_limit"):
                budget.start_provider_attempt()
        assert root.finalization_left == 1
        for _ in range(5):
            budget.start_provider_attempt()
        root.finalizing = True
        budget.start_provider_attempt()
        with pytest.raises(BudgetExceeded, match="provider_attempt_limit"):
            budget.start_provider_attempt()
    assert budget.active_allocation is None and budget.provider_attempts == 12


def test_retry_cannot_spend_reserved_finalization_slots(snapshot):
    corpus, _ = snapshot
    transport = Transport(ProviderTransportError("retry", retryable=True), ProviderTransportError("retry", retryable=True))
    now = [0.0]

    def wait(seconds):
        now[0] += seconds

    trace = RecursiveResearch(
        corpus, lambda _: ClaudeRootClient(transport, config=ClaudeConfig(max_retries=2),
                                           clock=lambda: now[0], wait=wait),
        clock=lambda: now[0], shared_limits=SharedLimits(max_provider_attempts=3, max_delegations=0),
        finalization=FinalizationPolicy(),
    ).run("Try retries.")
    assert trace.budget_stop == "investigation_attempt_limit"
    assert trace.shared_budget.investigation_provider_attempts == len(transport.calls) == 2
    assert trace.shared_budget.finalization_provider_attempts == 0


def test_remaining_finalization_slot_returns_rejections_for_every_pending_id(snapshot, monkeypatch):
    from evidenceroute.research.rlm.bedrock import BedrockRootClient

    corpus, (old, _, _) = snapshot
    chunk = corpus.read_chunks(old, RISK).items[0]
    read = tool("get_chunk", chunk_id=chunk.chunk_id)
    sessions = stub_bedrock(monkeypatch, [[
        converse_reply(read, "read"),
        converse_batch(read, delegate(SectionTarget(old, RISK))),
        converse_reply(finish(chunk.chunk_id, findings=[finding(chunk)]), "finish"),
    ]])
    trace = RecursiveResearch(
        corpus, lambda _: BedrockRootClient.from_env(disable_retries=True),
        shared_limits=SharedLimits(max_delegations=0), local_limits=RootLimits(max_tool_calls=1),
        finalization=FinalizationPolicy(finalization_attempts=2),
    ).run("Check a risk.")
    results = [block["toolResult"] for block in sessions[0].requests[-1]["messages"][-1]["content"]]
    assert [result["toolUseId"] for result in results] == ["batch-1", "batch-2"]
    assert all(result["content"][0]["json"]["error"]["code"] == "finalization_only" for result in results)
    assert trace.stop_reason == "finished" and trace.cited_chunk_ids == (chunk.chunk_id,)
    assert trace.shared_budget.corpus_attempts == 1 and trace.shared_budget.delegated_invocations == 0


@pytest.mark.parametrize("constraint", ["tokens", "deadline"])
def test_finalization_reservation_does_not_override_tokens_or_deadline(snapshot, constraint):
    corpus, (old, _, _) = snapshot
    now = [0.0]
    if constraint == "tokens":
        second = ProviderTransportError("retry", retryable=True, usage={"input_tokens": 5})
    else:
        second = message("finish_research", {"answer": "Too late."}, ident="finish")

    class FinalizationTransport(Transport):
        def send(self, payload, *, timeout_seconds):
            if self.calls and constraint == "deadline":
                assert timeout_seconds <= 1.0
                now[0] = 1.0
            return super().send(payload, timeout_seconds=timeout_seconds)

    transport = FinalizationTransport(
        message("list_sections", {"document_id": old}, usage={"input_tokens": 2, "output_tokens": 1}),
        second,
    )
    trace = RecursiveResearch(
        corpus, lambda _: ClaudeRootClient(transport, clock=lambda: now[0],
                                           wait=lambda _: pytest.fail("Retry must be blocked")),
        clock=lambda: now[0], local_limits=RootLimits(max_tool_calls=1),
        shared_limits=SharedLimits(max_delegations=0, max_reported_tokens=8, deadline_seconds=1.0),
        finalization=FinalizationPolicy(finalization_attempts=2),
    ).run("Check a risk.")
    assert trace.steps[-1].phase == "finalization" and trace.answer is None
    assert len(transport.calls) == 2
    if constraint == "tokens":
        assert trace.budget_stop == "reported_token_limit"
    else:
        assert trace.stop_reason == "deadline"
