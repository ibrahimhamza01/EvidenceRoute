"""Deterministic candidate scanning and bounded, evidence-linked comparison decisions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from evidenceroute.research.rlm.discovery import (
    Candidate as Candidate, CandidatePageRequest as CandidatePageRequest, CorpusScan as CorpusScan,
    ScannedSection as ScannedSection, scan_research_plan as scan_research_plan,
)
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.tools import Identifier, StrictInput


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


FindingReference = Annotated[str, Field(pattern=r"^(d[1-9][0-9]{0,8}|[ns][1-8])$")]


class ReferenceAssessment(StrictInput):
    ref: FindingReference = Field(description="Accepted submission sN, delivered child finding dN, or new finding nN.")
    relevance: Literal["infrastructure", "other", "uncertain"]
    entailment: Literal["supports", "contradicts", "uncertain"]
    rationale: str = Field(min_length=1, max_length=250, description="Briefly explain relevance, entailment, and any AI connection.")
    ai_specificity: Literal["explicit", "contextual"] | None = None
    context_ref: FindingReference | None = Field(default=None, description="Optional same-company, same-year AI evidence for a contextual assessment.")

    @property
    def infrastructure_connection(self) -> str:
        return self.rationale

    @property
    def ai_specific_connection(self) -> str | None:
        return self.rationale if self.ai_specificity else None


class ReferenceComparisonDraft(StrictInput):
    risk: str = Field(min_length=1, max_length=250, description="Neutral mechanism label. Leave novelty to the assessment.")
    same_risk: Literal["yes", "no", "uncertain"]
    assessments: list[ReferenceAssessment] = Field(min_length=1, max_length=8)


class ResolvedReferenceAssessment(StrictInput):
    finding_id: Identifier
    context_finding_id: Identifier | None = None
    relevance: Literal["infrastructure", "other", "uncertain"]
    entailment: Literal["supports", "contradicts", "uncertain"]
    rationale: str
    ai_specificity: Literal["explicit", "contextual"] | None = None


@dataclass(frozen=True)
class Comparison:
    risk: str
    ticker: str
    years: tuple[int, int]
    outcome: Literal["comparable_both_years", "possible_new_needs_review", "conflicting", "insufficient"]
    same_risk: str
    assessments: tuple[EvidenceAssessment | ResolvedReferenceAssessment, ...]
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
    def context(assessment: EvidenceAssessment) -> Finding | None:
        index = assessment.ai_context_finding_index
        if index is None:
            return None
        if index > len(findings):
            raise ValueError("AI context must refer to a finding in this finish.")
        return findings[index - 1]

    return _resolve_comparison(draft, [findings[index - 1] for index in indices], context, years)


def resolve_reference_comparison(
    draft: ReferenceComparisonDraft, findings: Mapping[str, Finding], years: tuple[int, int],
) -> Comparison:
    refs = [assessment.ref for assessment in draft.assessments]
    if len(set(refs)) != len(refs):
        raise ValueError("Comparison assessments must refer to distinct findings.")
    selected = []
    contexts = {}
    for assessment in draft.assessments:
        finding = findings.get(assessment.ref)
        if finding is None:
            raise ValueError(f"Unknown or unacknowledged finding reference: {assessment.ref}.")
        selected.append(finding)
        ref = assessment.context_ref
        if ref is None:
            continue
        context = findings.get(ref)
        if context is None:
            raise ValueError(f"Unknown or unacknowledged AI context reference: {ref}.")
        source = finding.evidence[0].source
        if assessment.ai_specificity != "contextual" or not _has_ai_context(context, source.ticker, source.fiscal_year):
            raise ValueError("AI context requires a same-company, same-year finding with an exact AI span.")
        contexts[assessment.ref] = context
    return _resolve_comparison(draft, selected, lambda assessment: contexts.get(assessment.ref), years)


def _has_ai_context(finding: Finding, ticker: str, year: int) -> bool:
    return any(item.span_id and item.source.fiscal_year == year and item.source.ticker == ticker
               and _AI_TEXT.search(item.quote) for item in finding.evidence)


def _resolve_comparison(draft, selected, lookup_context, years: tuple[int, int]) -> Comparison:
    if len({finding.finding_id for finding in selected}) != len(selected):
        raise ValueError("Comparison assessments must refer to distinct findings.")
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
    if (supported and contradicts) or (isinstance(draft, ComparisonDraft)
                                        and any(f.analysis_status == "conflicting" for f in selected)):
        outcome = "conflicting"
    elif supported == set(years) and draft.same_risk == "yes":
        outcome = "comparable_both_years"
    elif supported == {years[1]} and any(source.fiscal_year == years[0] for source in sources):
        outcome = "possible_new_needs_review"
    else:
        outcome = "insufficient"
    evidence_years = tuple(sorted({source.fiscal_year for source in sources}))
    explicit, contextual = set(), set()
    contexts: dict[str, Finding] = {}
    resolved = []
    for assessment, finding in zip(assessments, selected):
        context_id = None
        if (assessment.relevance == "infrastructure" and assessment.entailment == "supports"
                and assessment.ai_specific_connection):
            year = finding.evidence[0].source.fiscal_year
            direct = _has_ai_context(finding, sources[0].ticker, year)
            if assessment.ai_specificity == "explicit" and direct:
                explicit.add(year)
            if assessment.ai_specificity == "contextual":
                context = lookup_context(assessment)
                anchored = context is not None and _has_ai_context(context, sources[0].ticker, year)
                if direct or anchored:
                    contextual.add(year)
                if anchored and (not direct or isinstance(assessment, ReferenceAssessment)):
                    context_id = context.finding_id
                    contexts[context_id] = context
        if isinstance(assessment, ReferenceAssessment):
            resolved.append(ResolvedReferenceAssessment(finding_id=finding.finding_id, context_finding_id=context_id,
                            **assessment.model_dump(exclude={"ref", "context_ref"})))
        else:
            resolved.append(assessment)
    referenced = {finding.finding_id: finding for finding in selected}
    referenced.update(contexts)
    return Comparison(draft.risk, sources[0].ticker, years, outcome, draft.same_risk, tuple(resolved),
                      tuple(referenced),
                      tuple(sorted({item.source.chunk_id for finding in referenced.values() for item in finding.evidence})),
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
