"""Finding drafts from the model and findings resolved against SEC evidence."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, ValidationError

from evidenceroute.research.rlm.coverage import CoverageReport, SectionTarget
from evidenceroute.research.rlm.tools import EvidenceRef, Identifier, StrictInput

AnalysisStatus = Literal["analyzed", "incomplete", "conflicting"]
CoverageStatus = Literal["complete", "partial", "missing"]
MAX_FINDINGS = 8
MAX_FINDINGS_PER_SUBMISSION = 4

_NONDISCLOSURE = re.compile(
    r"\b(?:does|did|do|is|are|was|were|has|have|had)(?:\s+not|n['’]t)\s+"
    r"(?:\w+\s+){0,4}(?:disclos\w*|report\w*|provid\w*|contain\w*|mention\w*|includ\w*)\b"
    r"|\b(?:never\s+(?:disclos\w*|report\w*|provid\w*)|"
    r"no (?:mention|disclosure|data|information|evidence)\b|undisclosed|unreported|nowhere)\b"
    r"|\bno (?:\w+ ){1,6}(?:was|were|is|are) (?:reported|disclosed|provided)\b"
    r"|\b(?:filings?|reports?) (?:lacks?|omits?|fails? to (?:disclose|report|provide))\b"
    r"|\babsent from\b"
    r"|\b(?:only|solely)\s+(?:reports?|discloses?|provides?|includes?)\b",
    re.I,
)
_PASSAGE_LIMIT = re.compile(
    r"\b(?:reviewed|retrieved|examined|cited|available|supplied|delivered)\s+"
    r"(?:\w+\s+){0,2}(?:passages?|excerpts?|chunks?|evidence)\b"
    r"|\b(?:this|these|the cited) (?:passages?|excerpts?|tables?)\b", re.I,
)
_REPORTING_HISTORY = re.compile(
    r"\b(?:ceased|stopped|discontinued|began|started|resumed|no longer)\s+"
    r"(?:\w+\s+){0,3}(?:report|disclos|provid)\w*\b", re.I,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_RELATIVE_PERIOD = re.compile(r"\b(?:past|last|previous|prior) (?:\w+ ){0,2}(?:years?|quarters?|months?)\b", re.I)
_CAUSAL = re.compile(r"\b(?:because|due to|attribut\w+ to|as a result of|driven by|reflect(?:s|ed|ing)|resulted from)\b", re.I)


def unsupported_assertion(text: str, quotes: Iterable[str], filing_years: Iterable[int] = ()) -> bool:
    """Reject common scope and historical overreach before semantic review."""
    passages = tuple(" ".join(quote.casefold().split()) for quote in quotes)
    years = set(_YEAR.findall(" ".join(passages)))
    source_years = {str(year) for year in filing_years}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        normalized = " ".join(sentence.casefold().split()).rstrip(".")
        if not normalized or any(normalized in quote for quote in passages):
            continue
        stated_years = set(_YEAR.findall(sentence))
        history = _REPORTING_HISTORY.search(normalized)
        if history and not any(history[0] in quote and stated_years <= set(_YEAR.findall(quote))
                               for quote in passages):
            return True
        relative_periods = _RELATIVE_PERIOD.findall(normalized)
        if any(not any(period in passage for passage in passages) for period in relative_periods):
            return True
        if _CAUSAL.search(sentence) and (stated_years or relative_periods) and (
            stated_years - years - source_years or not any(_CAUSAL.search(quote) for quote in passages)
        ):
            return True
        for match in _NONDISCLOSURE.finditer(sentence):
            clause = re.split(r"[;,]|\b(?:but|however|while)\b", sentence[max(0, match.start() - 140):match.start()])[-1]
            local = clause + sentence[match.start():].split(",", 1)[0]
            if not _PASSAGE_LIMIT.search(local):
                return True
    return False


def grounded_finding(finding: "Finding") -> bool:
    return not any(unsupported_assertion(text, (item.quote for item in finding.evidence),
                                         (item.source.fiscal_year for item in finding.evidence))
                   for text in (finding.claim, finding.uncertainty, finding.conflict) if text)


class EvidenceDraft(StrictInput):
    chunk_id: Identifier
    quote: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=1, max_length=500, pattern=r"\S",
        description="Copy one short contiguous passage exactly from delivered evidence. "
                    "Keep punctuation and whitespace. Do not insert ellipses, paraphrase, or join passages.",
    )


class SpanEvidenceDraft(StrictInput):
    chunk_id: Identifier
    span_id: str = Field(pattern=r"^[0-9]{1,7}:[0-9]{1,7}$",
                         description="Select a span ID supplied with this chunk or a delivered excerpt.")


class FindingDraft(StrictInput):
    claim: str = Field(min_length=1, max_length=700)
    evidence: list[EvidenceDraft | SpanEvidenceDraft] = Field(min_length=1, max_length=3)
    analysis_status: AnalysisStatus
    uncertainty: str | None = Field(default=None, max_length=400)
    conflict: str | None = Field(default=None, max_length=400)


class SpanFindingDraft(FindingDraft):
    evidence: list[SpanEvidenceDraft] = Field(min_length=1, max_length=3,
        description="Select directly acknowledged spans in one document and section.")


class SubmitFindings(StrictInput):
    findings: list[SpanFindingDraft] = Field(min_length=1, max_length=MAX_FINDINGS_PER_SUBMISSION)


class CaptureFindings(SubmitFindings):
    findings: list[SpanFindingDraft] = Field(max_length=MAX_FINDINGS_PER_SUBMISSION, description=(
        "Record an atomic observation supported by acknowledged text from one fiscal year and filing section. "
        "A supported infrastructure dependency, disruption or constraint is eligible with an unresolved AI link "
        "or missing comparison year. Record uncertainty separately from the claim. Leave AI relevance, "
        "cross-year relationships and novelty to comparison. Use an empty list when no meaningful scoped "
        "observation is supported by the acknowledged text."
    ))


@dataclass(frozen=True)
class FindingValidationIssue:
    path: str
    category: str
    code: str


class FindingValidationError(ValueError):
    def __init__(self, path: str, category: str, code: str, message: str):
        super().__init__(message)
        self.issue = FindingValidationIssue(path, category, code)


_FINDING_FIELDS = frozenset({
    "synthesis", "new_findings", "findings", "claim", "evidence", "chunk_id", "span_id", "quote",
    "analysis_status", "uncertainty", "conflict", "used_finding_ids", "cite_refs", "cited_chunk_ids", "answer",
    "comparisons", "risk", "assessments", "ref", "context_ref", "finding_index", "ai_context_finding_index",
    "same_risk", "relevance", "entailment", "rationale", "infrastructure_connection", "ai_specificity",
    "ai_specific_connection",
    "conclusions", "statement", "qualification", "disposition", "bindings",
    "ticker", "filing_year", "section_id", "period_text",
})


def finding_schema_issues(error: ValidationError, prefix: str = "") -> tuple[FindingValidationIssue, ...]:
    """Use only schema fields and bounded indices, dropping unknown keys and rejected values."""
    issues = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        path = prefix
        location = item["loc"]
        if location and location[0] in ("finish", "synthesis") and not prefix:
            location = location[1:]
        if item["type"] == "extra_forbidden":
            location = location[:-1]
        for part in location:
            if isinstance(part, int) and 0 <= part < 50:
                path += f"[{part}]"
            elif part in _FINDING_FIELDS:
                path = f"{path}.{part}" if path else part
            elif part not in ("EvidenceDraft", "SpanEvidenceDraft"):
                break
        issue = FindingValidationIssue(path or "finish", "schema", item["type"])
        if issue not in issues:
            issues.append(issue)
        if len(issues) == 4:
            break
    return tuple(issues)


@dataclass(frozen=True)
class FindingEvidence:
    source: EvidenceRef
    quote: str
    span_id: str | None = None


@dataclass(frozen=True)
class Finding:
    finding_id: str
    company_name: str
    claim: str
    evidence: tuple[FindingEvidence, ...]
    analysis_status: AnalysisStatus
    coverage_status: CoverageStatus
    uncertainty: str | None
    conflict: str | None


@dataclass(frozen=True)
class PartitionCoverage:
    partition: SectionTarget
    status: CoverageStatus


def render_findings(findings: Iterable[Finding]) -> str:
    lines = []
    for finding in findings:
        source = finding.evidence[0].source
        citations = ", ".join(dict.fromkeys(item.source.chunk_id for item in finding.evidence))
        line = f"{source.ticker} FY{source.fiscal_year} [{finding.analysis_status}] {finding.claim} Evidence chunks {citations}"
        if finding.uncertainty:
            line += f" Uncertainty. {finding.uncertainty}"
        if finding.conflict:
            line += f" Conflict. {finding.conflict}"
        lines.append(line)
    return "\n".join(lines)


def partition_status(report: CoverageReport, partition: SectionTarget) -> CoverageStatus:
    related = [
        page for page in report.pagination
        if page.operation in {"read_chunks", "get_section"}
        and page.document_id == partition.document_id
        and page.section_id in (partition.section_id, None)
    ]
    if any(page.status == "complete" for page in related):
        return "complete"
    if any(page.ranges for page in related) or any(
        (item.document_id, item.section_id) == (partition.document_id, partition.section_id)
        for item in report.visible_chunks
    ):
        return "partial"
    return "missing"
