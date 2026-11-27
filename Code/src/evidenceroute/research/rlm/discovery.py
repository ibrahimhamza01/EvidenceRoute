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
from evidenceroute.retrieval.bm25 import BM25Config, score_texts
from evidenceroute.retrieval.tokenization import tokenize


_Phrase = Annotated[str, Field(min_length=1, max_length=200)]
DiscoveryPhrases = Annotated[tuple[_Phrase, ...], Field(min_length=1, max_length=32)]


class DiscoverySpec(StrictInput):
    """A topic and caller-supplied lexical phrases, independent of scope and model provider."""

    model_config = ConfigDict(frozen=True)

    topic: str = Field(min_length=1, max_length=2_000)
    phrases: DiscoveryPhrases
    expand_context: bool = True
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


# Domain vocabulary supplies retrieval anchors only. The investigator determines
# whether the selected evidence actually describes a risk and its AI connection.
_INFRA = re.compile(
    r"\b(?:data cent(?:er|re)s?|comput(?:e|ing) capacity|cloud (?:services?|infrastructure)|"
    r"accelerators?|gpus?|semiconductors?|chips?|servers?|third party (?:equipment|infrastructure)|"
    r"power (?:supply|availability|capacity)|electricity|water supply|cooling|infrastructure)\b"
)
_AI = re.compile(r"\b(?:ai|artificial intelligence|machine learning|large language models?)\b")
_INFRA_STRONG = re.compile(
    r"\b(?:data cent(?:er|re)s?|comput(?:e|ing) capacity|third party (?:equipment|infrastructure)|"
    r"power (?:supply|availability|capacity)|water supply|cooling)\b"
)
_RISK = re.compile(
    r"\b(?:risks?|depend\w*|rely|reliance|constrain\w*|shortages?|interrupt\w*|"
    r"disrupt\w*|availab\w*|limit\w*|fail\w*|outages?|supply|sufficient|unable)\b"
)
_DISRUPTION = re.compile(r"\b(?:shortages?|outages?|constrain\w*|disrupt\w*|interrupt\w*|impair\w*)\b")
# Standard filing-section names. Used only to align a question clause with the
# section being ranked. They do not add evidence or change scope.
_SECTION_ALIASES = {
    "item_1": ("business",),
    "item_1a": ("risk factors", "risk factor"),
    "item_1c": ("cybersecurity",),
    "item_7": ("md&a", "mda", "management's discussion", "management discussion"),
    "item_7a": ("market risk",),
    "item_8": ("financial statements",),
    "item_15": ("exhibits",),
}
_QUERY_BOILERPLATE = frozenset({
    "fiscal", "form", "filing", "annual", "report", "section", "item", "say", "10k",
})
# A component or balance the question did not name is a different measure.
_MEASURE_QUALIFIERS = frozenset({
    "product", "products", "service", "services", "segment", "segments", "deferred",
    "subscription", "software", "geographic", "component", "components", "balance", "balances",
})
# Question-term coverage leads. With length normalization off, repeated query
# terms still matter and equal scores fall through to the existing phrase rank.
_QUESTION_BM25 = BM25Config(b=0)


def _section_query_terms(question: str, section_id: str, section_name: str, companies: tuple[str, ...]) -> tuple[str, ...]:
    """Terms that ask for information from this section, without scope boilerplate."""
    aliases = _SECTION_ALIASES.get(section_id.lower(), ())
    title_terms = set(tokenize(section_name))
    title_terms.update(tokenize(" ".join(aliases)))
    clauses = re.split(r"\s*,?\s+and\s+(?=what\b|how\b|whether\b)|;\s*", question, flags=re.IGNORECASE)
    selected = []
    for clause in clauses:
        folded = clause.casefold()
        if any(alias in folded for alias in aliases) or (set(tokenize(clause)) & (title_terms - _QUERY_BOILERPLATE)):
            selected.append(_clause_focus(clause, aliases, title_terms))
    text = " ".join(selected) if selected else question
    blocked = set(_QUERY_BOILERPLATE)
    blocked.update(title_terms)
    blocked.update(tokenize(" ".join(companies)))
    return tuple(dict.fromkeys(
        token for token in tokenize(text)
        if token not in blocked and not (len(token) == 4 and token.isdigit())
    ))


def _clause_focus(clause: str, aliases: tuple[str, ...], title_terms: set[str]) -> str:
    folded = clause.casefold()
    start = len(clause)
    for alias in aliases:
        index = folded.find(alias)
        if index >= 0:
            start = min(start, index)
    for term in title_terms - _QUERY_BOILERPLATE:
        match = re.search(rf"\b{re.escape(term)}\b", folded)
        if match:
            start = min(start, match.start())
    return clause[start:] if start < len(clause) else clause


_INFLECTIONS = {
    "accelerators": "accelerator", "centers": "center", "centres": "center", "centre": "center",
    "chips": "chip", "constraints": "constraint", "dependencies": "dependency",
    "gpus": "gpu", "models": "model", "processors": "processor", "risks": "risk",
    "servers": "server", "shortages": "shortage", "units": "unit",
}


def _singularize(normalized: str) -> str:
    return " ".join(_INFLECTIONS.get(token, token) for token in normalized.split())


def _context(text: str) -> tuple[int, tuple[str, ...]]:
    # Score nearby signals above incidental matches elsewhere in the chunk.
    anchors = tuple(_INFRA.finditer(text))
    best = (0, ())
    for anchor in anchors:
        window = text[max(0, anchor.start() - 320):anchor.end() + 320]
        ai, risk = _AI.search(window), _RISK.search(window)
        if ai and risk:
            best = max(best, (3, (anchor.group(), ai.group(), risk.group())))
        elif risk and _INFRA_STRONG.fullmatch(anchor.group()):
            best = max(best, (2, (anchor.group(), risk.group())))
        elif ai:
            best = max(best, (1, (anchor.group(), ai.group())))
    return best


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

    def shortlist(
        self, pairs: tuple[SectionTarget, ...], *, per_section: int = 3, mechanism_hint: bool = False,
        include_identity: bool = True,
    ) -> list[dict]:
        authorized = set(pairs)
        shortlist = []
        for row in self.sections:
            if row.pair not in authorized:
                continue
            candidates = row.candidates[:per_section]
            entry = {"section_id": row.pair.section_id, "document_id": row.pair.document_id,
                     "candidate_count": len(row.candidates), "chunks": [item.chunk_id for item in candidates],
                     "matching_span_ids": self._span_hints(candidates),
                     "next_offset": per_section if len(row.candidates) > per_section else None}
            if include_identity:
                entry = {"ticker": row.ticker, "fiscal_year": row.fiscal_year, **entry}
            if mechanism_hint:
                for hit in row.candidates:
                    if hit.matching_span_ids and any(_INFRA_STRONG.fullmatch(term) for term in hit.terms) and any(
                        _DISRUPTION.search(term) for term in hit.terms
                    ):
                        entry["infrastructure_mechanism_hint"] = self._span_hints((hit,))
                        break
            shortlist.append(entry)
        return shortlist

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
    ranks: dict[str, tuple] = field(default_factory=dict)
    passages: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    question_rank: dict[str, tuple[int, float]] = field(default_factory=dict)


def _rank_question_passages(sections: list[_SectionScan], question: str, companies: tuple[str, ...]) -> None:
    """Rank lexical hits by how many requested terms a matching passage covers."""
    for section in sections:
        name = getattr(section.metadata, "section_name", "") or ""
        terms = _section_query_terms(question, section.pair.section_id, name, companies)
        if not terms or not section.passages:
            continue
        documents = [" ".join(tokens) for _, tokens in section.passages if tokens]
        owners = [chunk_id for chunk_id, tokens in section.passages if tokens]
        token_rows = [tokens for _, tokens in section.passages if tokens]
        if not documents:
            continue
        scores = score_texts(" ".join(terms), documents, _QUESTION_BM25)
        wanted = set(terms)
        asked = set(tokenize(question))
        best: dict[str, tuple[int, int, float]] = {}
        for chunk_id, tokens, score in zip(owners, token_rows, scores):
            covered = len(wanted.intersection(tokens))
            narrower = sum(1 for token in set(tokens) if token in _MEASURE_QUALIFIERS and token not in asked
                           and token.rstrip("s") not in asked)
            # Coverage first, then the measure the question named. Equal term
            # frequencies stay tied for the phrase-rank fallback.
            candidate = (covered, -narrower, round(score, 6))
            if candidate > best.get(chunk_id, (-1, -10**9, -1.0)):
                best[chunk_id] = candidate
        section.question_rank = best


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

    expanded = bool(spec and spec.expand_context)
    # The interpreted topic gates domain expansion. Unrelated questions keep
    # their own phrases and inflections, without AI infrastructure candidates.
    topic = _normalize(spec.topic) if spec else ""
    use_context = expanded and bool(_AI.search(topic) and _INFRA.search(topic))
    normalized_queries = {term: f" {_singularize(term) if expanded else term} " for term in terms}

    def matching_terms(text: str) -> tuple[tuple[str, ...], tuple]:
        if spec is None:
            folded = text.casefold()
            return tuple(term for term in terms if term in folded), ()
        normalized = _normalize(text)
        searchable = f" {_singularize(normalized) if expanded else normalized} "
        exact = tuple(term for term, query in normalized_queries.items() if query in searchable)
        length = max((len(term.split()) for term in exact), default=0)
        score, anchors = _context(normalized) if use_context else (0, ())
        # Specific phrases and local AI + risk co-occurrence lead. Single
        # broad terms (including GPU) cannot crowd out risk-bearing passages.
        rank = (max(4 if length >= 3 else 3 if length == 2 else length, score),
                score, length, len(exact))
        return tuple(dict.fromkeys((*exact, *anchors))), rank

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
                matched, rank = matching_terms(chunk.text)
                if matched:
                    spans = []
                    for span, (start, end) in passage_spans(chunk.text).items():
                        if not matching_terms(chunk.text[start:end])[0]:
                            continue
                        spans.append(span)
                        section.passages.append((chunk.chunk_id, tuple(tokenize(chunk.text[start:end]))))
                    section.ranks[chunk.chunk_id] = rank
                    section.candidates.append(Candidate(chunk.chunk_id, matched, tuple(spans)))
                chars += len(chunk.text)
            scanned += len(page.items)
            section.scanned += len(page.items)
        active = [section for section in active if section.scanned < section.metadata.chunk_count]

    if spec is not None and not use_context:
        _rank_question_passages(sections, plan.intent.question, plan.intent.companies)

    rows = []
    for section in sections:
        meta = section.metadata
        if spec is None:
            candidates = sorted(section.candidates, key=lambda hit: -len(hit.terms))
        else:
            candidates = sorted(section.candidates, key=lambda hit: (
                tuple(-value for value in section.question_rank.get(hit.chunk_id, (0, 0.0))),
                tuple(-value for value in section.ranks[hit.chunk_id]), hit.chunk_id,
            ))[:spec.max_candidates_per_section]
        rows.append(ScannedSection(section.pair, meta.ticker, meta.fiscal_year, meta.chunk_count, section.scanned,
                                   tuple(candidates), len(section.candidates)))
    return CorpusScan(terms, tuple(sorted(set(plan.intent.fiscal_years))), tuple(rows), pages, chars, clock() - started,
                      stop, len(sections), targets[len(sections):], spec)
