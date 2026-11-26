"""Deterministic candidate scanning and bounded, evidence-linked comparison decisions."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from pydantic import Field

from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import MAX_CHUNK_ROWS, CorpusEnvironment
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.planning import ResearchPlan
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import Identifier, StrictInput


class CandidatePageRequest(StrictInput):
    document_id: Identifier
    section_id: Identifier
    offset: int = Field(default=0, ge=0, description="Use next_offset from the shortlist or a candidate page.")
    limit: int = Field(default=5, ge=1, le=10)


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
                 "chunks": [item.chunk_id for item in row.candidates[:per_section]],
                 "matching_span_ids": self._span_hints(row.candidates[:per_section]),
                 "next_offset": per_section if len(row.candidates) > per_section else None}
                for row in self.sections if row.pair in authorized]

    def candidate_page(self, pair: SectionTarget, offset: int, limit: int) -> dict:
        row = next((row for row in self.sections if row.pair == pair), None)
        if row is None:
            raise KeyError(pair)
        end = offset + limit
        hits = row.candidates[offset:end]
        return {
            "ticker": row.ticker, "fiscal_year": row.fiscal_year, "document_id": pair.document_id,
            "section_id": pair.section_id, "candidate_count": len(row.candidates), "offset": offset,
            "chunks": [item.chunk_id for item in hits], "matching_span_ids": self._span_hints(hits),
            "next_offset": end if end < len(row.candidates) else None,
        }

    @staticmethod
    def _span_hints(candidates: tuple[Candidate, ...]) -> dict:
        return {hit.chunk_id: hit.matching_span_ids[:4] for hit in candidates if hit.matching_span_ids}


def scan_research_plan(corpus: CorpusEnvironment, plan: ResearchPlan, terms: tuple[str, ...], *,
                       max_chunks: int = 30_000, deadline_seconds: float = 30.0,
                       clock: Callable[[], float] = monotonic) -> CorpusScan:
    """Inspect canonical chunk pages locally. Matches are candidates, never absence evidence."""
    years = sorted(set(plan.intent.fiscal_years))
    if len(years) != 2 or plan.intent.objective != "year_over_year":
        raise ValueError("Comparative scanning requires a two-year comparison.")
    terms = tuple(sorted({term.strip().casefold() for term in terms if term.strip()}))
    if not terms or len(terms) > 32 or any(len(term) > 200 for term in terms):
        raise ValueError("Supply 1 to 32 nonempty terms of at most 200 characters.")
    if max_chunks < 1 or not 0 < deadline_seconds < float("inf"):
        raise ValueError("Use a positive scan chunk limit and finite deadline.")
    started = clock()
    rows = []
    pages = chars = scanned = 0
    stop = None
    for pair in plan.resolved_targets:
        meta = corpus.get_section(pair.document_id, pair.section_id, limit=1).metadata
        matches = []
        offset = 0
        while offset is not None and not stop:
            if scanned >= max_chunks or clock() - started >= deadline_seconds:
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
    return CorpusScan(terms, (years[0], years[1]), tuple(rows), pages, chars, clock() - started, stop)


class EvidenceAssessment(StrictInput):
    finding_index: int = Field(ge=1, le=8, description="One-based index in this finish's findings list.")
    relevance: Literal["infrastructure", "other", "uncertain"]
    entailment: Literal["supports", "contradicts", "uncertain"]
    rationale: str = Field(min_length=1, max_length=250)
    infrastructure_connection: str | None = Field(default=None, min_length=1, max_length=250, description=(
        "For infrastructure relevance, explain how the cited span links to the specific risk mechanism. "
        "Accelerator demand alone is insufficient. Relevance without this connection is treated as uncertain."
    ))
    ai_specific_connection: str | None = Field(default=None, min_length=1, max_length=250, description=(
        "If supported by this exact source span, describe how the risk mechanism specifically affects AI infrastructure. "
        "Leave empty for general infrastructure exposure or unrelated AI demand."
    ))
    ai_specificity: Literal["explicit", "contextual"] | None = Field(
        default=None, description="Use explicit when this finding directly links the risk to AI, or contextual for a supported inference."
    )
    ai_context_finding_index: int | None = Field(
        default=None, ge=1, le=8, description="For contextual support, use a same-year finding with an exact AI span."
    )


class ComparisonDraft(StrictInput):
    risk: str = Field(min_length=1, max_length=250, description="Neutral mechanism label. Leave novelty to the structured assessment.")
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
    evidence_years: tuple[int, ...]
    both_years_contributed: bool
    ai_specific_years: tuple[int, ...]
    ai_contextual_years: tuple[int, ...] = ()


_AI_TEXT = re.compile(r"\b(?:AI|artificial intelligence|machine learning)\b", re.IGNORECASE)


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
    assessments = tuple(item.model_copy(update={"relevance": "uncertain"})
                        if item.relevance == "infrastructure" and not item.infrastructure_connection else item
                        for item in draft.assessments)
    supported = {finding.evidence[0].source.fiscal_year for assessment, finding in zip(assessments, selected)
                 if assessment.relevance == "infrastructure" and assessment.entailment == "supports"}
    contradicts = any(item.relevance == "infrastructure" and item.entailment == "contradicts"
                      for item in assessments)
    if (supported and contradicts) or any(f.analysis_status == "conflicting" for f in selected):
        outcome = "conflicting"
    elif supported == set(years) and draft.same_risk == "yes":
        outcome = "comparable_both_years"
    elif supported == {years[1]} and any(source.fiscal_year == years[0] for source in sources):
        outcome = "possible_new_needs_review"
    else:
        outcome = "insufficient"
    evidence_years = tuple(sorted({source.fiscal_year for source in sources}))
    explicit, contextual = set(), set()
    context_indices = set()
    for assessment, finding in zip(assessments, selected):
        if (assessment.relevance != "infrastructure" or assessment.entailment != "supports"
                or not assessment.ai_specific_connection):
            continue
        year = finding.evidence[0].source.fiscal_year
        direct = any(_AI_TEXT.search(item.quote) for item in finding.evidence)
        if assessment.ai_specificity == "explicit" and direct:
            explicit.add(year)
        if assessment.ai_specificity == "contextual":
            index = assessment.ai_context_finding_index
            if index is not None and index > len(findings):
                raise ValueError("AI context must refer to a finding in this finish.")
            context = findings[index - 1].evidence if index is not None else ()
            anchored = any(item.span_id and item.source.fiscal_year == year and item.source.ticker == sources[0].ticker
                           and _AI_TEXT.search(item.quote) for item in context)
            if direct or anchored:
                contextual.add(year)
                if anchored and not direct:
                    context_indices.add(index)
    referenced = [*selected, *(findings[index - 1] for index in sorted(context_indices) if index not in indices)]
    return Comparison(draft.risk, sources[0].ticker, years, outcome, draft.same_risk, assessments,
                      tuple(f.finding_id for f in referenced),
                      tuple(sorted({item.source.chunk_id for finding in referenced for item in finding.evidence})),
                      evidence_years, evidence_years == years, tuple(sorted(explicit)), tuple(sorted(contextual - explicit)))


def render_comparisons(comparisons: tuple[Comparison, ...] | list[Comparison]) -> str:
    """Render conclusions from validated outcomes without adopting model-drafted prose."""
    lines = []
    for item in comparisons:
        earlier, later = (f"FY{year}" for year in item.years)
        conclusion = {
            "comparable_both_years": f"The assessed spans support the same underlying infrastructure risk in {earlier} and {later}.",
            "possible_new_needs_review": (f"The assessed spans support a risk in {later}. Whether it is new remains unresolved. "
                                          "New wording alone cannot establish a new underlying risk."),
            "conflicting": "The assessed evidence is conflicting. Novelty remains unresolved.",
            "insufficient": "The assessed evidence is insufficient to establish novelty.",
        }[item.outcome]
        ai_years = ", ".join(f"FY{year}" for year in item.ai_specific_years)
        context_years = ", ".join(f"FY{year}" for year in item.ai_contextual_years)
        ai = []
        if ai_years:
            ai.append(f"The model identified an explicit AI-specific connection in {ai_years}.")
        if context_years:
            ai.append(f"The model identified a contextual AI connection in {context_years}.")
        if not ai:
            ai.append("The assessments leave an AI-specific connection unestablished.")
        risk = " ".join(item.risk.split())
        lines.append(f'{item.ticker} ({earlier} and {later}), risk label “{risk}” [{item.outcome}]. '
                     f'{conclusion} {" ".join(ai)} Evidence chunks {", ".join(item.assessed_chunk_ids)}.')
    return "\n".join(lines) if lines else "No validated structured comparison was produced."


def render_comparative_scope(
    comparisons: tuple[Comparison, ...], answer: str, *, requested: tuple[str, ...], admitted: set[str],
    incomplete: bool,
) -> str:
    comparable = {item.ticker for item in comparisons if item.outcome == "comparable_both_years"}
    inconclusive = {item.ticker for item in comparisons if item.outcome != "comparable_both_years"}
    unresolved = [ticker for ticker in requested if ticker in admitted and (ticker not in comparable or ticker in inconclusive)]
    not_admitted = [ticker for ticker in requested if ticker not in admitted]
    return "\n\n".join([
        "Research is incomplete. Conclusions apply to the assessed passages only. "
        "This result cannot establish absence of a risk or answer for the full requested scope."
        if incomplete else "Research conclusions await independent semantic verification.",
        "Supported novelty. The validated comparisons establish no novelty conclusion.",
        "Comparable existing risks. " + (", ".join(sorted(comparable)) or "None established in the assessed comparisons."),
        "Structured assessments.\n" + answer,
        "Admitted companies with unresolved assessments. " + (", ".join(unresolved) or "None identified."),
        f"Companies outside the admitted scope ({len(not_admitted)}). " + (", ".join(not_admitted) or "None."),
    ])
