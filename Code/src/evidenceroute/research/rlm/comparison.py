"""Deterministic candidate scanning and bounded, evidence-linked comparison decisions."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Literal

from pydantic import Field

from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import MAX_CHUNK_ROWS, CorpusEnvironment
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.planning import ResearchPlan
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import StrictInput


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    terms: tuple[str, ...]
    matching_span_ids: tuple[str, ...]


@dataclass(frozen=True)
class ScannedSection:
    pair: SectionTarget
    ticker: str
    fiscal_year: int
    inventory_chunks: int
    scanned_chunks: int
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class CorpusScan:
    terms: tuple[str, ...]
    years: tuple[int, int]
    sections: tuple[ScannedSection, ...]
    pages_read: int
    characters_scanned: int
    elapsed_seconds: float
    stop_reason: str | None

    def shortlist(self, pairs: tuple[SectionTarget, ...], *, per_section: int = 3) -> list[dict]:
        authorized = set(pairs)
        return [{"ticker": row.ticker, "fiscal_year": row.fiscal_year, "section_id": row.pair.section_id,
                 "document_id": row.pair.document_id, "candidate_count": len(row.candidates),
                 "chunks": [item.chunk_id for item in row.candidates[:per_section]]}
                for row in self.sections if row.pair in authorized]


def scan_research_plan(corpus: CorpusEnvironment, plan: ResearchPlan, terms: tuple[str, ...], *,
                       max_chunks: int = 30_000, deadline_seconds: float = 30.0) -> CorpusScan:
    """Inspect canonical chunk pages locally. Matches are candidates, never absence evidence."""
    years = sorted(set(plan.intent.fiscal_years))
    if len(years) != 2 or plan.intent.objective != "year_over_year":
        raise ValueError("Comparative scanning requires a two-year comparison.")
    terms = tuple(sorted({term.strip().casefold() for term in terms if term.strip()}))
    if not terms or len(terms) > 32 or any(len(term) > 200 for term in terms):
        raise ValueError("Supply 1 to 32 nonempty terms of at most 200 characters.")
    if max_chunks < 1 or not 0 < deadline_seconds < float("inf"):
        raise ValueError("Use a positive scan chunk limit and finite deadline.")
    started = monotonic()
    rows = []
    pages = chars = scanned = 0
    stop = None
    for pair in plan.resolved_targets:
        meta = corpus.get_section(pair.document_id, pair.section_id, limit=1).metadata
        matches = []
        offset = 0
        while offset is not None and not stop:
            if scanned >= max_chunks or monotonic() - started >= deadline_seconds:
                stop = "chunk_limit" if scanned >= max_chunks else "deadline"
                break
            page = corpus.read_chunks(pair.document_id, pair.section_id, offset=offset,
                                      limit=min(MAX_CHUNK_ROWS, max_chunks - scanned))
            pages += 1
            for chunk in page.items:
                text = chunk.text.casefold()
                matched = tuple(term for term in terms if term in text)
                if matched:
                    spans = tuple(span for span, (start, end) in passage_spans(chunk.text).items()
                                  if any(term in chunk.text[start:end].casefold() for term in matched))
                    matches.append(Candidate(chunk.chunk_id, matched, spans))
                chars += len(chunk.text)
            scanned += len(page.items)
            offset = page.next_offset
        rows.append(ScannedSection(pair, meta.ticker, meta.fiscal_year, meta.chunk_count,
                                   meta.chunk_count if offset is None else offset,
                                   tuple(sorted(matches, key=lambda hit: -len(hit.terms)))))
    return CorpusScan(terms, (years[0], years[1]), tuple(rows), pages, chars, monotonic() - started, stop)


class EvidenceAssessment(StrictInput):
    finding_index: int = Field(ge=1, le=8, description="One-based index in this finish's findings list.")
    relevance: Literal["infrastructure", "other", "uncertain"]
    entailment: Literal["supports", "contradicts", "uncertain"]
    rationale: str = Field(min_length=1, max_length=250)


class ComparisonDraft(StrictInput):
    risk: str = Field(min_length=1, max_length=250)
    same_risk: Literal["yes", "no", "uncertain"]
    assessments: list[EvidenceAssessment] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class Comparison:
    risk: str
    ticker: str
    years: tuple[int, int]
    outcome: Literal["comparable_both_years", "possible_new_needs_review", "conflicting", "insufficient"]
    same_risk: str
    assessments: tuple[EvidenceAssessment, ...]
    finding_ids: tuple[str, ...]
    assessed_chunk_ids: tuple[str, ...]


def resolve_comparison(draft: ComparisonDraft, findings: list[Finding], years: tuple[int, int]) -> Comparison:
    indices = [item.finding_index for item in draft.assessments]
    if len(set(indices)) != len(indices) or any(index > len(findings) for index in indices):
        raise ValueError("Comparison assessments must refer to distinct findings in this finish.")
    selected = [findings[index - 1] for index in indices]
    sources = [item.source for finding in selected for item in finding.evidence]
    if len({source.ticker for source in sources}) != 1 or any(source.fiscal_year not in years for source in sources):
        raise ValueError("A comparison must use evidence from one company and the requested years.")
    if any(not item.span_id for finding in selected for item in finding.evidence):
        raise ValueError("Comparison assessments require source-backed spans.")
    supported = {finding.evidence[0].source.fiscal_year for assessment, finding in zip(draft.assessments, selected)
                 if assessment.relevance == "infrastructure" and assessment.entailment == "supports"}
    contradicts = any(item.relevance == "infrastructure" and item.entailment == "contradicts"
                      for item in draft.assessments)
    if (supported and contradicts) or any(f.analysis_status == "conflicting" for f in selected):
        outcome = "conflicting"
    elif supported == set(years) and draft.same_risk == "yes":
        outcome = "comparable_both_years"
    elif supported == {years[1]} and any(source.fiscal_year == years[0] for source in sources):
        outcome = "possible_new_needs_review"
    else:
        outcome = "insufficient"
    return Comparison(draft.risk, sources[0].ticker, years, outcome, draft.same_risk, tuple(draft.assessments),
                      tuple(f.finding_id for f in selected), tuple(sorted({s.chunk_id for s in sources})))
