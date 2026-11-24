"""Track scope and model-visible access without inferring research completeness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.tools import EvidenceRef


@dataclass(frozen=True)
class SectionTarget:
    document_id: str
    section_id: str


@dataclass(frozen=True)
class ResearchScope:
    """Host-established requirements. None leaves that part of the scope unknown."""

    required_filings: tuple[str, ...] | None = None
    required_sections: tuple[SectionTarget, ...] | None = None

    def validate(self, corpus: CorpusEnvironment) -> None:
        for document_id in self.required_filings or ():
            corpus.get_filing(document_id)
        for section in self.required_sections or ():
            corpus.get_section(section.document_id, section.section_id, limit=1)


@dataclass(frozen=True)
class PaginationCoverage:
    operation: str
    document_id: str
    section_id: str | None
    total: int
    ranges: tuple[tuple[int, int], ...]
    status: Literal["partial", "complete"]


@dataclass(frozen=True)
class CoverageReport:
    scope: ResearchScope
    search_discovered: tuple[EvidenceRef, ...]
    visible_chunks: tuple[EvidenceRef, ...]
    pagination: tuple[PaginationCoverage, ...]


class CoverageTracker:
    """Record only the access that the corresponding tool actually represents."""

    def __init__(self, scope: ResearchScope):
        self._scope = scope
        self._discovered: dict[str, EvidenceRef] = {}
        self._visible: dict[str, EvidenceRef] = {}
        self._pagination: dict[tuple[str, str, str | None], PaginationCoverage] = {}

    def record_search(self, evidence: tuple[EvidenceRef, ...]) -> None:
        self._discovered.update((ref.chunk_id, ref) for ref in evidence)

    def record_delivery(
        self, operation: str, arguments: dict, result: dict, evidence: tuple[EvidenceRef, ...],
    ) -> None:
        self._visible.update((ref.chunk_id, ref) for ref in evidence)
        if operation == "get_section":
            total = result["metadata"]["char_count"]
            size = len(result["text"])
        elif operation in {"read_chunks", "list_sections"}:
            total = result["total"]
            size = len(result["items"])
        else:
            return

        document_id = arguments["document_id"]
        section_id = arguments.get("section_id")
        key = (operation, document_id, section_id)
        previous = self._pagination.get(key)
        ranges = list(previous.ranges) if previous else []
        start = arguments["offset"]
        if size:
            ranges.append((start, start + size))
        merged = _merge_ranges(ranges)
        complete = total == 0 or merged == ((0, total),)
        self._pagination[key] = PaginationCoverage(
            operation, document_id, section_id, total, merged, "complete" if complete else "partial",
        )

    def report(self) -> CoverageReport:
        return CoverageReport(
            self._scope, tuple(self._discovered.values()), tuple(self._visible.values()),
            tuple(self._pagination.values()),
        )


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(end, previous_end))
        else:
            merged.append((start, end))
    return tuple(merged)
