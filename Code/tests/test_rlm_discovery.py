"""Bounded, scoped lexical discovery and exact SEC provenance."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.discovery import DiscoverySpec, discover_research_plan
from evidenceroute.research.rlm.environment import Page
from evidenceroute.research.rlm.planning import CompanyPartition, ResearchPlan, interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.scoped import ScopedCorpus
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import CorpusToolDispatcher


class SectionFixture:
    def __init__(self, rows):
        self.rows = rows
        self.pages = []
        self.metadata = []
        self.now = 0.0
        self.seconds_per_read = 0.0

    def get_section(self, doc, section, *, limit):
        pair = SectionTarget(doc, section)
        self.metadata.append(pair)
        return SimpleNamespace(metadata=SimpleNamespace(ticker=doc.split("-")[0], fiscal_year=int(doc.split("-")[1]),
                                                       chunk_count=len(self.rows[pair])))

    def read_chunks(self, doc, section, *, offset, limit):
        pair = SectionTarget(doc, section)
        self.pages.append((pair, offset, limit))
        self.now += self.seconds_per_read
        rows = self.rows[pair]
        return Page(tuple(SimpleNamespace(chunk_id=key, text=text) for key, text in rows[offset:offset + limit]),
                    len(rows), offset + limit if offset + limit < len(rows) else None)


def _plan(pairs):
    tickers = tuple(dict.fromkeys(pair.document_id.split("-")[0] for pair in pairs))
    intent = interpret_intent("How does the company discuss exchange rates?", lambda *_: {
        "company_selection": "explicit", "companies": tickers, "fiscal_years": (2023, 2024),
        "section_ids": ("item_1a",), "objective": "year_over_year", "topic": "Exchange rates. Compare the disclosures.",
        "comprehensive": True, "inferred_fields": (),
    })
    return ResearchPlan(intent, tickers, tuple(CompanyPartition(ticker, intent.question, tuple(
        pair for pair in pairs if pair.document_id.split("-")[0] == ticker)) for ticker in tickers), ())


def test_discovery_validates_phrases_and_falls_back_to_intent_topic():
    plan = _plan((SectionTarget("ABC-2023", "item_1a"), SectionTarget("ABC-2024", "item_1a")))
    spec = DiscoverySpec.from_intent(plan.intent)
    assert spec.topic == plan.intent.topic and spec.phrases == ("Exchange rates",)
    with pytest.raises(ValidationError, match="lexical tokens"):
        DiscoverySpec.from_intent(plan.intent, ("---",))
    with pytest.raises(ValidationError):
        DiscoverySpec.from_intent(plan.intent, ("rates",), max_candidates_per_section=0)


def test_normalized_variants_rank_deterministically_inside_exact_plan_scope():
    first, second, outside = (SectionTarget("ABC-2023", "item_1a"), SectionTarget("ABC-2024", "item_1a"),
                              SectionTarget("ABC-2023", "item_7"))
    corpus = SectionFixture({first: [("z", "EXCHANGE-RATES fluctuate."), ("b", "Foreign currency exchange rates."),
                                     ("a", "ＦＯＲＥＩＧＮ currency exchange-rates."), ("c", "Superexchange rates.")],
                             second: [("d", "Currency conversions affect results.")], outside: [("x", "Foreign currency exchange rates.")]})
    plan = _plan((first, second))
    phrases = ("exchange rates", "foreign currency exchange rates", "exchange-rates")
    spec = DiscoverySpec.from_intent(plan.intent, phrases, max_candidates_per_section=2)
    scan = discover_research_plan(corpus, plan, spec)
    repeat = discover_research_plan(corpus, plan, DiscoverySpec.from_intent(plan.intent, phrases[::-1], max_candidates_per_section=2))
    assert scan.sections == repeat.sections and scan.terms == repeat.terms
    assert [candidate.chunk_id for candidate in scan.sections[0].candidates] == ["a", "b"]
    assert scan.sections[0].matched_chunks == 3 and scan.sections[0].candidates[0].matching_span_ids
    assert scan.sections[1].scanned_chunks == 1 and scan.sections[1].candidates == ()
    assert scan.shortlist((outside,)) == []
    assert outside not in corpus.metadata and all(pair != outside for pair, _, _ in corpus.pages)


@pytest.mark.parametrize(("limits", "seconds", "expected", "scanned"), [
    ({"max_chunks": 5}, 0, "chunk_limit", [2, 1, 1, 1]),
    ({"max_pages": 4}, 0, "page_limit", [10, 10, 10, 10]),
    ({"deadline_seconds": 2.0}, 1, "deadline", [10, 10, 0, 0]),
])
def test_budgets_interleave_sections_and_record_unscanned_work(limits, seconds, expected, scanned):
    pairs = tuple(SectionTarget(f"{ticker}-{year}", "item_1a") for ticker in ("ABC", "DEF") for year in (2023, 2024))
    corpus = SectionFixture({pair: [(f"{pair.document_id}-{i}", "Exchange rates.") for i in range(21)] for pair in pairs})
    corpus.seconds_per_read = seconds
    plan = _plan(pairs)
    scan = discover_research_plan(corpus, plan, DiscoverySpec.from_intent(plan.intent), clock=lambda: corpus.now, **limits)
    assert scan.stop_reason == expected and [row.scanned_chunks for row in scan.sections] == scanned
    assert scan.pages_read == len(corpus.pages) and scan.metadata_reads == len(corpus.metadata) == 4
    assert [pair for pair, _, _ in corpus.pages] == list(pairs[:scan.pages_read])


def test_canonical_targets_and_exact_span_provenance(monkeypatch, capsys):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    corpus = CorpusEnvironment()
    monkeypatch.setattr("evidenceroute.research.rlm.CorpusEnvironment", lambda: corpus)
    demo = runpy.run_path(str(examples / "demo_rlm_discovery.py"))
    demo["main"]()
    report = json.loads(capsys.readouterr().out)
    assert report["provider_attempts"] == 0 and not report["model_assessment_performed"]
    assert len(report["literal_ai_baseline"]["known_targets_found"]) == 2
    assert set(report["normalized_ai_phrases"]["known_targets_found"]) == demo["KNOWN_TARGETS"]
    assert report["foreign_currency_review"]["candidate_chunks"] > 0

    key = "4d1474550060251037c5f981"
    chunk = corpus.get_chunk(key)
    pair = SectionTarget(chunk.document_id, chunk.section_id)
    dispatcher = CorpusToolDispatcher(ScopedCorpus(corpus, (pair,)), include_spans=True, max_response_bytes=6000)
    spec_plan = resolve_research_plan(interpret_intent("Review semiconductor supply.", lambda *_: {
        "company_selection": "explicit", "companies": ["AMZN"], "fiscal_years": [2024], "section_ids": ["item_1a"],
        "objective": "review", "topic": "Artificial intelligence infrastructure", "comprehensive": False, "inferred_fields": [],
    }), corpus)
    scan = discover_research_plan(corpus, spec_plan, DiscoverySpec.from_intent(spec_plan.intent))
    hit = next(hit for hit in scan.sections[0].candidates if hit.chunk_id == key)
    response = dispatcher.dispatch("get_spans", {"chunk_id": key, "span_ids": [hit.matching_span_ids[0]]})
    span = hit.matching_span_ids[0]
    start, end = passage_spans(chunk.text)[span]
    assert json.loads(response.content)["result"]["spans"][span] == chunk.text[start:end]
    assert response.evidence[0].source_url == chunk.source_url
