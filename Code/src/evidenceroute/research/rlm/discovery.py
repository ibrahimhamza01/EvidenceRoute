"""Bounded lexical candidate discovery over canonical filing sections."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Annotated

from pydantic import ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import MAX_CHUNK_ROWS, CorpusEnvironment, SectionMetadata
from evidenceroute.research.rlm.planning import ResearchIntent, ResearchPlan
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.research.rlm.tools import Identifier, StrictInput
from evidenceroute.retrieval.tokenization import tokenize


_Phrase = Annotated[str, Field(min_length=1, max_length=200)]
DiscoveryPhrases = Annotated[tuple[_Phrase, ...], Field(min_length=1, max_length=32)]


class DiscoverySpec(StrictInput):
    """A topic and caller-supplied lexical phrases, independent of scope and model provider."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(min_length=1, max_length=2_000)
    phrases: DiscoveryPhrases
    max_candidates_per_section: int = Field(default=16, ge=1, le=50)

    @field_validator("phrases")
    @classmethod
    def validate_phrases(cls, phrases: DiscoveryPhrases) -> DiscoveryPhrases:
        if any(not _normalize(phrase) for phrase in phrases):
            raise PydanticCustomError("non_lexical_phrase", "Every discovery phrase must contain lexical tokens.")
        return phrases

    @classmethod
    def from_intent(cls, intent: ResearchIntent, phrases: tuple[str, ...] = (), **limits) -> DiscoverySpec:
        """Use refined phrases when provided, otherwise the leading clause of the intent topic."""
        topic_phrase = re.split(r"[.!?;](?:\s|$)", intent.topic, maxsplit=1)[0].strip()
        return cls(topic=intent.topic, phrases=phrases or (topic_phrase,), **limits)


def _normalize(text: str) -> str:
    return " ".join(tokenize(text, remove_stopwords=False))


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
    matched_chunks: int = 0


@dataclass(frozen=True)
class CorpusScan:
    terms: tuple[str, ...]
    years: tuple[int, ...]
    sections: tuple[ScannedSection, ...]
    pages_read: int
    characters_scanned: int
    elapsed_seconds: float
    stop_reason: str | None
    metadata_reads: int = 0
    unvisited_targets: tuple[SectionTarget, ...] = ()
    discovery: DiscoverySpec | None = None

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
                       max_pages: int | None = None, clock: Callable[[], float] = monotonic) -> CorpusScan:
    """Reproduce the literal-term comparative baseline using the same bounded scanner."""
    years = sorted(set(plan.intent.fiscal_years))
    if len(years) != 2 or plan.intent.objective != "year_over_year":
        raise ValueError("Comparative scanning requires a two-year comparison.")
    terms = tuple(sorted({term.strip().casefold() for term in terms if term.strip()}))
    if not terms or len(terms) > 32 or any(len(term) > 200 for term in terms):
        raise ValueError("Supply 1 to 32 nonempty terms of at most 200 characters.")
    return _scan(corpus, plan, terms, max_chunks, deadline_seconds, max_pages, clock, spec=None)


def discover_research_plan(corpus: CorpusEnvironment, plan: ResearchPlan, spec: DiscoverySpec, *,
                           max_chunks: int = 30_000, deadline_seconds: float = 30.0,
                           max_pages: int | None = None, clock: Callable[[], float] = monotonic) -> CorpusScan:
    """Find normalized phrase matches within the resolved plan, without widening evidence access."""
    terms = tuple(sorted({_normalize(phrase) for phrase in spec.phrases}))
    return _scan(corpus, plan, terms, max_chunks, deadline_seconds, max_pages, clock, spec=spec)


@dataclass
class _SectionScan:
    pair: SectionTarget
    metadata: SectionMetadata
    scanned: int = 0
    candidates: list[Candidate] = field(default_factory=list)


def _scan(
    corpus: CorpusEnvironment, plan: ResearchPlan, terms: tuple[str, ...], max_chunks: int,
    deadline_seconds: float, max_pages: int | None, clock: Callable[[], float], *, spec: DiscoverySpec | None,
) -> CorpusScan:
    if (type(max_chunks) is not int or max_chunks < 1 or not 0 < deadline_seconds < float("inf")
            or (max_pages is not None and (type(max_pages) is not int or max_pages < 0))):
        raise ValueError("Use a positive scan chunk limit, nonnegative page limit and finite deadline.")
    max_pages = max_chunks if max_pages is None else max_pages
    started = clock()
    deadline = started + deadline_seconds
    targets = plan.resolved_targets
    sections = []
    pages = chars = scanned = 0
    stop = None
    for pair in targets:
        if clock() >= deadline:
            stop = "deadline"
            break
        meta = corpus.get_section(pair.document_id, pair.section_id, limit=1).metadata
        sections.append(_SectionScan(pair, meta))

    def matching_terms(text: str, queries: tuple[str, ...]) -> tuple[str, ...]:
        if spec is None:
            folded = text.casefold()
            return tuple(term for term in queries if term in folded)
        normalized = f" {_normalize(text)} "
        return tuple(term for term in queries if f" {term} " in normalized)

    active = [section for section in sections if section.metadata.chunk_count]
    while active and stop is None:
        for index, section in enumerate(active):
            if clock() >= deadline:
                stop = "deadline"
            elif scanned >= max_chunks:
                stop = "chunk_limit"
            elif pages >= max_pages:
                stop = "page_limit"
            if stop is not None:
                break
            sections_left = len(active) - index
            share = (max_chunks - scanned + sections_left - 1) // sections_left
            pair = section.pair
            page = corpus.read_chunks(pair.document_id, pair.section_id, offset=section.scanned,
                                      limit=min(MAX_CHUNK_ROWS, share))
            pages += 1
            for chunk in page.items:
                matched = matching_terms(chunk.text, terms)
                if matched:
                    spans = tuple(span for span, (start, end) in passage_spans(chunk.text).items()
                                  if matching_terms(chunk.text[start:end], matched))
                    section.candidates.append(Candidate(chunk.chunk_id, matched, spans))
                chars += len(chunk.text)
            scanned += len(page.items)
            section.scanned += len(page.items)
        active = [section for section in active if section.scanned < section.metadata.chunk_count]

    rows = []
    for section in sections:
        meta = section.metadata
        if spec is None:
            candidates = sorted(section.candidates, key=lambda hit: -len(hit.terms))
        else:
            candidates = sorted(section.candidates, key=lambda hit: (
                -max(len(term.split()) for term in hit.terms), -len(hit.terms), hit.chunk_id,
            ))[:spec.max_candidates_per_section]
        rows.append(ScannedSection(section.pair, meta.ticker, meta.fiscal_year, meta.chunk_count, section.scanned,
                                   tuple(candidates), len(section.candidates)))
    return CorpusScan(terms, tuple(sorted(set(plan.intent.fiscal_years))), tuple(rows), pages, chars, clock() - started,
                      stop, len(sections), targets[len(sections):], spec)
