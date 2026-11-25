"""Resolve SEC research requirements without retrieval, execution, or provider calls."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import Page
from evidenceroute.research.rlm.planning import IntentInterpretationError, interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch


class MetadataCorpus:
    def __init__(self, reverse=False):
        self.rows = [
            ("BETA", "Beta Inc.", 2024, "b24", ("item_1a",)),
            ("ACME", "Acme Corp", 2024, "a24", ("item_1c", "item_1a")),
            ("ACME", "Acme Old Name", 2023, "a23", ("item_1a",)),
        ]
        if reverse:
            self.rows.reverse()

    @staticmethod
    def _page(items, offset, limit):
        assert limit > 0
        # Use short pages to ensure the resolver follows the metadata cursor.
        end = offset + 1
        return Page(tuple(items[offset:end]), len(items), end if end < len(items) else None)

    def list_filings(self, *, offset, limit):
        rows = [SimpleNamespace(ticker=t, company_name=n, fiscal_year=y, document_id=d)
                for t, n, y, d, _ in self.rows]
        return self._page(rows, offset, limit)

    def list_sections(self, document_id, *, offset, limit):
        sections = next(sections for _, _, _, doc, sections in self.rows if doc == document_id)
        return self._page([SimpleNamespace(section_id=s) for s in sections], offset, limit)


def requirements(**overrides):
    return {
        "company_selection": "all", "fiscal_years": [2024, 2023], "section_ids": ["item_1a"],
        "objective": "year_over_year", "topic": "Compare disclosures about flood risk.",
        "comprehensive": True, "inferred_fields": [], **overrides,
    }


def plan_for(corpus=None, **overrides):
    intent = interpret_intent("Compare the requested risk factors.", lambda *_: requirements(**overrides))
    return resolve_research_plan(intent, corpus or MetadataCorpus())


def test_all_companies_group_years_and_preserve_missing_requirements():
    plan = plan_for(section_ids=["ITEM_1C", "item_1a"])
    assert plan.requested_companies == ("ACME", "BETA")
    assert plan.required_target_count == 8
    assert [(group.ticker, group.targets) for group in plan.groups] == [
        ("ACME", (SectionTarget("a23", "item_1a"), SectionTarget("a24", "item_1a"), SectionTarget("a24", "item_1c"))),
        ("BETA", (SectionTarget("b24", "item_1a"),)),
    ]
    assert [(m.company, m.fiscal_year, m.section_id, m.reason, m.document_ids) for m in plan.missing_requirements] == [
        ("ACME", 2023, "item_1c", "missing_section", ("a23",)),
        ("BETA", 2023, "item_1a", "missing_filing", ()),
        ("BETA", 2023, "item_1c", "missing_filing", ()),
        ("BETA", 2024, "item_1c", "missing_section", ("b24",)),
    ]
    assert all(group.question == plan.intent.question for group in plan.groups)


def test_explicit_selection_resolves_names_and_deduplicates_aliases_and_requirements():
    plan = plan_for(company_selection="explicit", companies=[" acme ", "Acme Corp", "ACME  old NAME"],
                    fiscal_years=[2024, 2023, 2024], section_ids=["item_1a", "ITEM_1A"])
    assert plan.requested_companies == ("ACME",)
    assert plan.resolved_targets == (SectionTarget("a23", "item_1a"), SectionTarget("a24", "item_1a"))
    assert plan.required_target_count == 2 and not plan.missing_requirements


def test_unresolved_company_keeps_every_required_slot():
    plan = plan_for(company_selection="explicit", companies=[" ACME ", "Unknown", " unknown "],
                    section_ids=["item_1a", "item_999"])
    assert plan.requested_companies == ("ACME", "unknown")
    assert plan.required_target_count == 8
    assert [group.ticker for group in plan.groups] == ["ACME"]
    assert [m.reason for m in plan.missing_requirements].count("unknown_company") == 4
    assert [m.reason for m in plan.missing_requirements].count("missing_section") == 2


def test_ambiguous_company_or_filing_is_reported_without_guessing():
    corpus = MetadataCorpus()
    corpus.rows.append(("BETA", "Acme Old Name", 2023, "b23", ("item_1a",)))
    corpus.rows.append(("ACME", "Acme Corp", 2023, "a23-alt", ("item_1a",)))
    plan = plan_for(corpus, company_selection="explicit", companies=["Acme Old Name", "ACME"])
    assert [m.reason for m in plan.missing_requirements] == ["ambiguous_filing", "ambiguous_company", "ambiguous_company"]
    filing, company, _ = plan.missing_requirements
    assert filing.document_ids == ("a23", "a23-alt")
    assert company.candidate_tickers == ("ACME", "BETA")
    assert plan.resolved_targets == (SectionTarget("a24", "item_1a"),)


def test_order_is_stable_across_metadata_pages_and_input_order():
    first = plan_for(section_ids=["item_1c", "item_1a"])
    second = plan_for(MetadataCorpus(reverse=True), fiscal_years=[2023, 2024], section_ids=["item_1a", "item_1c"])
    assert first.groups == second.groups and first.missing_requirements == second.missing_requirements


def test_execution_budget_does_not_change_required_scope():
    plan = plan_for()
    assert plan.required_target_count == 4 and len(plan.resolved_targets) == 3
    assert plan.delegation_shortfalls(SharedLimits(max_depth=0, max_delegations=1, max_partitions=2)) == (
        "max_depth", "max_delegations", "max_partitions",
    )
    assert plan.delegation_shortfalls(SharedLimits(max_depth=1, max_delegations=2, max_partitions=3)) == ()
    assert plan.required_target_count == 4 and len(plan.missing_requirements) == 1
    missing = plan_for(fiscal_years=[2030], objective="review")
    assert missing.resolved_targets == () and missing.required_target_count == 2
    assert missing.delegation_shortfalls(SharedLimits()) == ("no_resolved_targets",)


@pytest.mark.parametrize("output", [
    None, "unstructured response", {}, requirements(company_selection="explicit"),
    requirements(companies=["ACME"]), requirements(fiscal_years=[2024, 2024]),
    requirements(fiscal_years=["2023", 2024]), requirements(section_ids=["Risk Factors"]),
    requirements(comprehensive="true"), requirements(inferred_fields=["section_ids"]),
    requirements(company_selection="industry"), requirements(question="model replaces question"),
])
def test_rejects_invalid_interpretations(output):
    with pytest.raises(IntentInterpretationError, match="Invalid research intent"):
        interpret_intent("Compare filings.", lambda *_: output)


def test_interpreter_receives_schema_and_host_question_is_preserved():
    question = "  Compare filings.  "

    def interpret(prompt, schema):
        assert prompt == question
        assert "question" not in schema["properties"]
        assert "section_ids" in schema["required"]
        return requirements(inferred_fields=["section_ids"], scope_notes="Use Item 1A for risk disclosures.")

    result = interpret_intent(question, interpret)
    assert result.question == question and result.inferred_fields == ("section_ids",)


def test_interpreter_failure_is_reported_without_emitting_raw_error():
    def fail(*_):
        raise RuntimeError("raw provider detail")

    with pytest.raises(IntentInterpretationError, match=r"^Research intent interpretation failed\.$"):
        interpret_intent("Compare filings.", fail)


@pytest.fixture(scope="module")
def canonical_corpus():
    try:
        return CorpusEnvironment()
    except FileNotFoundError:
        pytest.skip("Canonical local corpus unavailable")


def test_canonical_plan_demo_and_runner_handoff(canonical_corpus, monkeypatch, capsys):
    from evidenceroute.research.rlm.claude import ClaudeRootClient

    def unexpected_request(*_args, **_kwargs):
        pytest.fail("Offline planning requires no provider requests")

    monkeypatch.setattr(ClaudeRootClient, "_request", unexpected_request)
    script = Path(__file__).resolve().parents[1] / "examples" / "demo_rlm_plan.py"
    demo = runpy.run_path(str(script))
    demo["main"].__globals__["CorpusEnvironment"] = lambda: canonical_corpus
    assert demo["main"](["--max-delegations", "25", "--max-partitions", "49"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "offline planning demonstration"
    assert report["provider_attempts"] == 0 and not report["evidence_assessed"]
    assert (report["corpus_companies"], report["corpus_filings"]) == (25, 75)
    assert report["requested_companies"] == report["company_partition_groups"] == 25
    assert report["required_targets"] == report["resolved_targets"] == 50
    assert not report["missing_requirements"] and not report["example_delegation_admission"]["fits"]
    assert report["example_delegation_admission"]["blocking_limits"] == ["max_partitions"]
    assert all([t["fiscal_year"] for t in g["targets"]] == [2023, 2024] for g in report["representative_groups"])

    intent = interpret_intent(demo["QUESTION"], demo["scripted_interpretation"])
    plan = resolve_research_plan(intent, canonical_corpus)
    assert len(set(plan.resolved_targets)) == 50
    for group in plan.groups:
        filings = [canonical_corpus.get_filing(target.document_id) for target in group.targets]
        assert [(filing.ticker, filing.fiscal_year) for filing in filings] == [(group.ticker, 2023), (group.ticker, 2024)]
        assert all(target.section_id == "item_1a" for target in group.targets)

    group = plan.groups[0]
    runner = RecursiveResearch(
        canonical_corpus, lambda _: ScriptedModelClient([{"type": "finish", "answer": "No evidence reviewed."}]),
        shared_limits=SharedLimits(max_delegations=1, max_partitions=2),
        finalization=FinalizationPolicy(),
    )
    trace = runner.run(group.question, partitions=group.targets)
    assert trace.stop_reason == "finished" and trace.shared_budget.provider_attempts == 0
    assert trace.coverage.scope.required_sections == group.targets and trace.coverage_status == "missing"
