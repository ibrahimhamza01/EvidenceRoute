"""Finding drafts from the model and findings resolved against SEC evidence."""

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
        "One year and one filing section per finding. Submit an empty list when acknowledged evidence is insufficient."
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
