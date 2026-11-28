"""Offline structural tests for generic conclusions and their durable evidence bindings."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.checkpoint import CheckpointError, continue_research, create_checkpoint, load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.controller import ScriptedModelClient, finish_input_schema
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.coverage import CoverageReport, ResearchScope
from evidenceroute.research.rlm.planning import ResearchIntent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.synthesis import ABSTENTION, SourceOperand, resolve_conclusions, validate_conclusion
from evidenceroute.research.rlm.tools import EvidenceRef
from evidenceroute.research.rlm.verification import (
    BedrockSemanticVerifier, VerificationLimits, record_verification, verification_report, verify_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
STATEMENT = "Apple describes exchange-rate exposure in component costs, while Amazon describes reliance on a limited group of chip suppliers."
EXCERPTS = {
    ("AAPL", 2024, "item_1a"): ("cd7988623c9a3bf545f06045", "1099:1275", "Stronger foreign currencies may increase component costs."),
    ("AMZN", 2024, "item_1a"): ("4d1474550060251037c5f981", "16:200", "Amazon relies on a limited group of semiconductor suppliers."),
}


@pytest.fixture(scope="module")
def corpus():
    return CorpusEnvironment()


@pytest.fixture
def path():
    p = ROOT / f"rlm-synthesis-test-{uuid4().hex}.json"
    yield p
    p.unlink(missing_ok=True)


def tool(name, **args):
    return dict(type="tool", name=name, arguments=args)


def run(corpus, *, companies=("AAPL", "AMZN"), years=(2024,), sections=("item_1a",),
        mode="valid", batch_size=2, qualified=False, require_all=True):
    required = tuple(SourceOperand(ticker=ticker, filing_year=year, section_id=section)
                     for ticker in companies for year in years for section in sections)
    plan = resolve_research_plan(ResearchIntent(
        question="How do the selected disclosures compare?", companies=companies, company_selection="explicit",
        fiscal_years=years, section_ids=sections, topic="Disclosures", comprehensive=False, objective="review",
        inferred_fields=(), required_operands=required if require_all else (),
    ), corpus)
    clients = []

    def factory(inv):
        actions = []
        if inv.depth:
            drafts = []
            for pair in inv.partitions:
                filing = corpus.get_filing(pair.document_id)
                key = (filing.ticker, filing.fiscal_year, pair.section_id)
                if key in EXCERPTS:
                    chunk, span, claim = EXCERPTS[key]
                else:
                    source = corpus.read_chunks(pair.document_id, pair.section_id, limit=1).items[0]
                    chunk, span, claim = source.chunk_id, next(iter(passage_spans(source.text))), "Scripted passage assessment."
                actions.append(tool("get_spans", chunk_id=chunk, span_ids=[span]))
                draft = dict(claim=claim, analysis_status="analyzed", evidence=[dict(chunk_id=chunk, span_id=span)])
                if qualified:
                    draft.update(conflict="The estimates differ.", uncertainty="The effect is conditional.")
                drafts.append(draft)
            actions.append(dict(type="finish", answer="Child evidence.", findings=drafts))
        else:
            bindings = []
            for group in plan.groups:
                if not set(group.targets) <= set(inv.partitions):
                    continue
                actions.append(tool("delegate_research", question="Inspect disclosures.", goal="Find scoped evidence.",
                                    partitions=[asdict(p) for p in group.targets]))
                for pair in group.targets:
                    filing = corpus.get_filing(pair.document_id)
                    bindings.append(dict(ticker=filing.ticker, filing_year=filing.fiscal_year,
                                         section_id=pair.section_id, ref=f"d{len(bindings) + 1}"))
            refs = [b["ref"] for b in bindings]
            if mode == "wrong_ref":
                bindings[0]["ref"] = "d96"
            if mode == "wrong_company":
                bindings[0]["ticker"] = "MSFT"
            if mode == "wrong_year":
                bindings[0]["filing_year"] -= 1
            if mode == "wrong_section":
                bindings[0]["section_id"] = "item_7"
            if mode == "missing":
                bindings.pop()
            statement = STATEMENT if len({b["ticker"] for b in bindings}) == 2 else "Scripted cross-source relationship."
            proposal = dict(statement=statement, bindings=list(reversed(bindings)), disposition="answered",
                            qualification="The passages describe exposures rather than realized losses.")
            if mode == "abstain":
                proposal.update(statement=STATEMENT, bindings=bindings[:1], disposition="insufficient_evidence")
                refs = refs[:1]
            if mode != "unfinished":
                actions.append(dict(type="synthesis", synthesis=dict(
                    conclusions=[proposal], cite_refs=[] if mode == "undelivered" else refs, analysis_status="analyzed")))
        client = ScriptedModelClient(actions)
        if not inv.depth and mode == "undelivered":
            client.last_observation_delivered = False
        clients.append(client)
        return client

    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(max_delegations=2, max_partitions=8))
    limits = CorpusLimits(max_batches=2, max_provider_attempts=48, max_corpus_attempts=64, max_delegations=4,
                         max_partitions=16, max_reported_tokens=120_000, deadline_seconds=60.0)
    result = execute_corpus_research(plan, runner, limits=limits, max_groups=batch_size)
    assert bool(result.errors) == (mode == "unfinished")
    return result, runner, clients


def test_conclusion_schema_names_the_rejected_field_and_one_correction_keeps_findings(corpus):
    bindings = [
        {"ref": "d1", "ticker": "AAPL", "filing_year": 2024, "section_id": "item_1a"},
        {"ref": "d2", "ticker": "AMZN", "filing_year": 2024, "section_id": "item_1a"},
    ]
    _, issues = resolve_conclusions([{
        "statement": "x" * 1201, "disposition": "answered", "qualification": "y" * 601, "bindings": bindings,
    }], {}, (), None, [], type("Coverage", (), {"visible_chunks": ()})())
    assert {(issue.path, issue.code) for issue in issues} == {
        ("synthesis.conclusions[0].statement", "string_too_long"),
        ("synthesis.conclusions[0].qualification", "string_too_long"),
    }
    _, accepted = resolve_conclusions([{
        "statement": "x" * 701, "disposition": "answered", "bindings": bindings,
    }], {}, (), None, [], type("Coverage", (), {"visible_chunks": ()})())
    assert accepted[0].code == "unacknowledged_finding_reference"
    assert accepted[0].path == "synthesis.conclusions[0].bindings"

    required = tuple(SourceOperand(ticker=ticker, filing_year=2024, section_id="item_1a") for ticker in ("AAPL", "AMZN"))
    plan = resolve_research_plan(ResearchIntent(
        question="How do the selected disclosures compare?", companies=("AAPL", "AMZN"), company_selection="explicit",
        fiscal_years=(2024,), section_ids=("item_1a",), topic="Disclosures", comprehensive=False, objective="review",
        inferred_fields=(), required_operands=required,
    ), corpus)
    clients = []

    def factory(inv):
        actions = []
        if inv.depth:
            for pair in inv.partitions:
                filing = corpus.get_filing(pair.document_id)
                chunk, span, claim = EXCERPTS[filing.ticker, filing.fiscal_year, pair.section_id]
                actions.append(tool("get_spans", chunk_id=chunk, span_ids=[span]))
            actions.append(dict(type="finish", answer="Child evidence.", findings=[{
                "claim": claim, "analysis_status": "analyzed",
                "evidence": [{"chunk_id": chunk, "span_id": span}],
            }]))
        else:
            bindings = []
            for group in plan.groups:
                actions.append(tool("delegate_research", question="Inspect disclosures.", goal="Find scoped evidence.",
                                    partitions=[asdict(pair) for pair in group.targets]))
                for pair in group.targets:
                    filing = corpus.get_filing(pair.document_id)
                    bindings.append(dict(ticker=filing.ticker, filing_year=filing.fiscal_year,
                                         section_id=pair.section_id, ref=f"d{len(bindings) + 1}"))
            refs = [item["ref"] for item in bindings]
            actions.append(dict(type="synthesis", synthesis=dict(
                cite_refs=refs, analysis_status="analyzed", conclusions=[{
                    "statement": "x" * 1201, "disposition": "answered", "bindings": bindings,
                }])))
            actions.append(dict(type="synthesis", synthesis=dict(
                cite_refs=refs, analysis_status="analyzed", conclusions=[{
                    "statement": STATEMENT, "disposition": "answered", "bindings": bindings,
                }])))
        client = ScriptedModelClient(actions)
        clients.append(client)
        return client

    runner = RecursiveResearch(
        corpus, factory,
        shared_limits=SharedLimits(max_delegations=2, max_partitions=4, max_provider_attempts=16,
                                   max_corpus_attempts=16, max_reported_tokens=80_000),
        finalization=FinalizationPolicy(child_provider_attempts=4, finalization_attempts=2),
    )
    result = execute_corpus_research(plan, runner, limits=CorpusLimits(
        max_batches=1, max_provider_attempts=16, max_corpus_attempts=16, max_delegations=2,
        max_partitions=4, max_reported_tokens=80_000, deadline_seconds=60,
    ), max_groups=2)
    assert STATEMENT in result.answer and result.answer_disposition == "answered"
    assert "string_too_long" not in result.answer
    feedback = " ".join(item or "" for client in clients for item in client.observations)
    assert "conclusion_correction" in feedback and "synthesis.conclusions[0].statement string_too_long" in feedback


def test_canonical_conclusion_replay_and_continuation(corpus, path):
    result, runner, clients = run(corpus)
    checkpoint = create_checkpoint(corpus, result)
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint(path, corpus)
    assert loaded == checkpoint and loaded.version == 4
    root = loaded.runs[0].coverage.invocations[0]
    conclusion, = root.conclusions
    assert [b.finding_id for b in conclusion.bindings] == list(reversed(root.used_finding_ids))
    assert [r.ticker for r in result.answer_citations] == ["AMZN", "AAPL"]
    assert result.answer_citations == loaded.publication.answer_citations
    assert result.answer_disposition == "answered" and result.research_status == "incomplete"
    assert STATEMENT in result.answer and result.answer.index(STATEMENT) < result.answer.index("Findings selected")
    assert result.answer.endswith(loaded.publication.answer) and "unverified" in result.answer
    assert sum(len(c.observations) for c in clients) == 7
    updated = continue_research(loaded, runner, companies=("AAPL",), limits=result.limits)
    save_checkpoint(path, updated)
    restored = load_checkpoint(path, corpus)
    assert restored.version == 4 and restored.runs[0] == loaded.runs[0]
    assert STATEMENT in restored.publication.answer and restored.historical_usage == loaded.current_usage
    for finding in restored.findings:
        for evidence in finding.evidence:
            chunk = corpus.get_chunk(evidence.source.chunk_id)
            assert evidence.source == EvidenceRef.from_chunk(chunk)
            assert evidence.quote == chunk.text[slice(*passage_spans(chunk.text)[evidence.span_id])]

    for key, bad in (("finding_id", "fabricated"), ("filing_year", 2020), ("section_id", "item_7")):
        data = loaded.model_dump(mode="json")
        data["runs"][0]["coverage"]["invocations"][0]["conclusions"][0]["bindings"][0][key] = bad
        path.write_text(json.dumps(data))
        with pytest.raises(CheckpointError, match="conclusion"):
            load_checkpoint(path, corpus)



@pytest.mark.parametrize("mode", ["wrong_ref", "wrong_company", "wrong_year", "wrong_section", "missing",
                                  "undelivered", "unfinished"])
def test_bad_synthesis_keeps_eligible_atomic_evidence(corpus, mode):
    result, _, _ = run(corpus, mode=mode)
    root = result.coverage.invocations[0]
    assert not root.conclusions and STATEMENT not in result.answer
    assert result.answer_disposition == "insufficient_evidence"
    if mode == "undelivered":
        assert not result.answer_citations
    else:
        assert len(result.answer_citations) == 2
        assert all(e[2] in result.answer for e in EXCERPTS.values())
    assert result.publication_state == ("partial_evidence" if mode == "unfinished" else
                                       "insufficient_evidence" if mode == "undelivered" else "selected_evidence")
    if mode == "wrong_company":
        assert "wrong_source" in result.answer
    if mode == "wrong_ref":
        assert "unacknowledged_finding_reference" in result.answer


@pytest.mark.parametrize("years, sections", [((2023, 2024), ("item_1a",)), ((2024,), ("item_1a", "item_7"))])
def test_source_period_and_section_bindings_with_qualifications(corpus, years, sections):
    result, _, _ = run(corpus, companies=("AAPL",), years=years, sections=sections, qualified=True)
    conclusion, = result.coverage.invocations[0].conclusions
    assert {b.filing_year for b in conclusion.bindings} == set(years)
    assert {b.section_id for b in conclusion.bindings} == set(sections)
    assert "[partial, unverified]" in result.answer and result.answer_disposition == "partial"
    assert "The estimates differ." in result.answer and "The effect is conditional." in result.answer


def test_comparison_split_between_roots(corpus):
    result, _, _ = run(corpus, batch_size=1)
    assert all(not n.conclusions for n in result.coverage.invocations)
    assert "separate roots remain unresolved" in result.answer and result.answer_disposition == "insufficient_evidence"
    assert len(result.answer_citations) == 2


def test_abstention_can_cite_reviewed_evidence_without_a_cross_source_conclusion(corpus):
    result, _, _ = run(corpus)
    root = result.coverage.invocations[0]
    left = root.conclusions[0].bindings[0]
    evidence = next(f for f in result.acknowledged_child_findings if f.finding_id == left.finding_id)
    known = {left.finding_id: evidence}
    record = root.conclusions[0].model_copy(update={
        "statement": "The reviewed passages do not disclose the requested operating metric.",
        "qualification": None,
        "disposition": "insufficient_evidence",
        "bindings": (left,),
    })
    validate_conclusion(record, known, root.invocation.required_operands, root.invocation.partitions)
    for statement in ("The company does not disclose this metric in its filing.",
                      "The company stopped reporting the metric in FY2019."):
        with pytest.raises(ValueError, match="unsupported_assertion"):
            validate_conclusion(record.model_copy(update={"statement": statement}), known, (), root.invocation.partitions)
    with pytest.raises(ValueError, match="missing_operand"):
        validate_conclusion(record.model_copy(update={"disposition": "answered"}), known,
                            root.invocation.required_operands, root.invocation.partitions)
    raw = {"statement": "The company stopped reporting the metric in FY2019.",
           "disposition": "insufficient_evidence", "bindings": []}
    records, issues = resolve_conclusions([raw], {}, root.invocation.required_operands,
                                          root.invocation.partitions, (), CoverageReport(ResearchScope(), (), (), ()))
    assert not issues and records[0].statement == ABSTENTION

    abstention, _, _ = run(corpus, mode="abstain")
    assert abstention.answer_disposition == "insufficient_evidence"
    assert ABSTENTION in abstention.response.answer and STATEMENT not in abstention.response.answer
    assert [chunk.ticker for chunk in abstention.response.citations] == ["AAPL"]


@pytest.mark.parametrize("compact, comparative", [(False, True), (True, True), (False, False)])
def test_only_generic_root_schema_offers_conclusions(compact, comparative):
    options = dict(recursive=True, comparative=comparative, reference_based=True, compact=compact)
    assert "conclusions" not in json.dumps(finish_input_schema(**options))
    schema = finish_input_schema(**options, generic_root=True)
    assert ("conclusions" in json.dumps(schema)) == (not comparative and not compact)


def test_period_anchor_and_authorized_partition(corpus):
    result, _, _ = run(corpus)
    root = result.coverage.invocations[0]
    record, = root.conclusions
    earlier, later = record.bindings
    # Synthetic quote about a historical reporting period in a FY2024 source.
    known = {f.finding_id: f for f in result.acknowledged_child_findings}
    f = known[earlier.finding_id]
    known[f.finding_id] = replace(f, evidence=(replace(f.evidence[0], quote="FY2022 costs rose by $3 million."),))
    period = earlier.model_copy(update={"period_text": "FY2022"})
    record = record.model_copy(update={"bindings": (period, later)})
    validate_conclusion(record, known, root.invocation.required_operands, root.invocation.partitions)
    with pytest.raises(ValueError, match="missing_period_text"):
        validate_conclusion(record.model_copy(update={"bindings": (period.model_copy(update={"period_text": "FY2021"}), later)}),
                            known, (), root.invocation.partitions)
    with pytest.raises(ValueError, match="wrong_source"):
        validate_conclusion(record, known, (), root.invocation.partitions[:1])
    with pytest.raises(ValueError, match="missing_operand"):
        validate_conclusion(record, known, [SourceOperand(ticker="AMD", filing_year=2024, section_id="item_1a")],
                            root.invocation.partitions)


def test_continuation_can_bind_delivered_history_and_verified_projection_stays_separate(corpus, path):
    result, runner, _ = run(corpus)
    original = create_checkpoint(corpus, result)

    def factory(inv):
        if inv.depth:
            return ScriptedModelClient([dict(type="finish", answer="No new findings.")])
        delegates = [tool("delegate_research", question="Inspect filing.", goal="Record evidence.",
                          partitions=[asdict(p) for p in group.targets]) for group in result.plan.groups]
        bindings = [dict(**operand.model_dump(), ref=f"d{i}") for i, operand in enumerate(inv.required_operands, 1)]
        return ScriptedModelClient([*delegates, tool("read_prior_findings"), dict(type="synthesis", synthesis={
            "conclusions": [dict(statement="Scoped reassessment.", bindings=bindings, disposition="partial")],
        })])

    runner.model_factory = factory
    updated = continue_research(original, runner, companies=("AAPL", "AMZN"), limits=result.limits)
    assert updated.runs[1].coverage.invocations[0].acknowledged_prior_finding_ids == tuple(f.finding_id for f in original.findings)
    first, second = updated.publication.published_conclusions
    assert first == result.published_conclusions[0] and first.invocation_id != second.invocation_id
    assert [b.finding_id for b in second.conclusion.bindings] == list(reversed([b.finding_id for b in first.conclusion.bindings]))

    class ScriptedTransport:
        def send(self, payload, *, timeout_seconds):
            cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
            reviews = [dict(ref=c["ref"], components=[dict(statement=c["claim"], status="supported", evidence=["e1"])])
                       for c in cases]
            return {"stopReason": "tool_use", "usage": {"inputTokens": 100, "outputTokens": 40},
                    "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                        "toolUseId": "test1", "name": "submit_verification", "input": {"reviews": reviews}}}]}}}

    verifier = BedrockSemanticVerifier(ScriptedTransport(), config=BedrockConfig(model="fixture"))
    review = verify_checkpoint(updated, corpus, verifier, limits=VerificationLimits())
    assert len(review.items) == 2 and all(item.status == "supported" for item in review.items)
    recorded = record_verification(updated, corpus, review)
    save_checkpoint(path, recorded)
    restored = load_checkpoint(path, corpus)
    assert restored.version == 4 and restored.publication.published_conclusions == (first, second)
    assert all(p.conclusion.semantic_status == "unverified" for p in restored.publication.published_conclusions)
    assert STATEMENT not in verification_report(restored)["answer"]


def test_local_receipts_required_for_durable_conclusions(corpus, path):
    template, runner, _ = run(corpus)
    plan = template.plan

    def factory(inv):
        if inv.depth:
            return ScriptedModelClient([dict(type="finish", answer="No findings.")])
        actions = [tool("delegate_research", question="Inspect disclosures.", goal="Find evidence.",
                        partitions=[asdict(p) for p in group.targets]) for group in plan.groups]
        bindings = []
        for i, operand in enumerate(inv.required_operands, 1):
            chunk, span, claim = EXCERPTS[operand.ticker, operand.filing_year, operand.section_id]
            actions.extend([tool("get_spans", chunk_id=chunk, span_ids=[span]), tool("submit_findings", findings=[
                dict(claim=claim, analysis_status="analyzed", evidence=[dict(chunk_id=chunk, span_id=span)])
            ])])
            bindings.append(dict(**operand.model_dump(), ref=f"s{i}"))
        actions.append(dict(type="synthesis", synthesis={"conclusions": [
            dict(statement=STATEMENT, disposition="answered", bindings=bindings)
        ]}))
        return ScriptedModelClient(actions)

    runner.model_factory = factory
    result = execute_corpus_research(plan, runner, limits=template.limits, max_groups=2)
    checkpoint = create_checkpoint(corpus, result)
    published, = checkpoint.publication.published_conclusions
    assert [b.finding_id for b in published.conclusion.bindings] == [f.finding_id for f in checkpoint.findings]
    save_checkpoint(path, checkpoint)
    data = json.loads(path.read_bytes())
    root = data["runs"][0]["coverage"]["invocations"][0]
    root["acknowledged_local_finding_ids"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="conclusion"):
        load_checkpoint(path, corpus)


def test_merge_scoped_conclusions_cannot_import_another_batch(corpus, path):
    result, _, _ = run(corpus, years=(2023, 2024), batch_size=1, require_all=False)
    left, right = result.published_conclusions
    assert left.invocation_id != right.invocation_id
    assert {r.ticker for r in left.citations} == {"AAPL"} and {r.ticker for r in right.citations} == {"AMZN"}
    assert result.answer_citations == (*left.citations, *right.citations)
    assert result.answer_disposition == "partial" and "separate roots remain unresolved" in result.answer
    checkpoint = create_checkpoint(corpus, result)
    assert checkpoint.publication.published_conclusions == (left, right)
    data = checkpoint.model_dump(mode="json")
    roots = [n for n in data["runs"][0]["coverage"]["invocations"] if not n["invocation"]["parent_id"]]
    roots[0]["conclusions"][0]["bindings"][0] = roots[1]["conclusions"][0]["bindings"][0]
    path.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="conclusion"):
        load_checkpoint(path, corpus)
