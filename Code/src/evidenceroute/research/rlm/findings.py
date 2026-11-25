"""Finding drafts from the model and findings resolved against SEC evidence."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from evidenceroute.research.rlm.coverage import CoverageReport, SectionTarget
from evidenceroute.research.rlm.tools import EvidenceRef, Identifier, StrictInput

AnalysisStatus = Literal["analyzed", "incomplete", "conflicting"]
CoverageStatus = Literal["complete", "partial", "missing"]


class EvidenceDraft(StrictInput):
    chunk_id: Identifier
    quote: str = Field(min_length=1, max_length=500)


class FindingDraft(StrictInput):
    claim: str = Field(min_length=1, max_length=700)
    evidence: list[EvidenceDraft] = Field(min_length=1, max_length=3)
    analysis_status: AnalysisStatus
    uncertainty: str | None = Field(default=None, max_length=400)
    conflict: str | None = Field(default=None, max_length=400)


@dataclass(frozen=True)
class FindingEvidence:
    source: EvidenceRef
    quote: str


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
