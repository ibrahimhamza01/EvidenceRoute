"""Offline recursive workflows over canonical-shaped local fixtures."""

import csv
from collections import deque
from dataclasses import asdict
import json

import pytest

from evidenceroute.ingestion.processor import chunk_sections, write_parquet
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.claude import ClaudeConfig, ClaudeRootClient, ProviderTransportError
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.recursive import RecursiveResearch
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
