"""Bounded, read-only access to a local SEC corpus snapshot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from typing import Generic, TypeVar

from evidenceroute.ingestion.downloader import DATA_DIR, load_canonical_manifest, load_manifest
from evidenceroute.retrieval import BM25Retriever, ChunkFilter, CorpusError, load_chunks
from evidenceroute.schemas import Chunk, Filing, RetrievalResult, Section

MAX_METADATA_ROWS = 25
MAX_CHUNK_ROWS = 10
MAX_SECTION_CHARS = 4_000
MAX_QUERY_CHARS = 1_000

T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    total: int
    next_offset: int | None


@dataclass(frozen=True)
class CompanyCoverage:
    ticker: str
    company_name: str
    fiscal_years: tuple[int, ...]


@dataclass(frozen=True)
class SectionMetadata:
    document_id: str
    section_id: str
    section_name: str
    ticker: str
    fiscal_year: int
    source_url: str
    char_count: int
    chunk_count: int


@dataclass(frozen=True)
class SectionWindow:
    metadata: SectionMetadata
    text: str
    offset: int
    next_offset: int | None


def _validate_page(offset: int, limit: int, maximum: int) -> None:
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be an integer from 1 to {maximum}")


def _page(items: Sequence[T], offset: int, limit: int) -> Page[T]:
    end = offset + limit
    return Page(tuple(items[offset:end]), len(items), end if end < len(items) else None)


class CorpusEnvironment:
    """Expose a validated local snapshot through bounded inspection methods.

    Pass explicit paths for offline fixtures or other trusted local snapshots.
    Calls use document and section IDs as lookup keys.
    """

    def __init__(
        self,
        *,
        manifest_path: Path | str | None = None,
        processed_dir: Path | str = DATA_DIR / "processed" / "full-v1",
    ) -> None:
        import pyarrow.parquet as pq

        filings = (
            load_canonical_manifest() if manifest_path is None
            else load_manifest(Path(manifest_path))
        )
        self._filings = {
            filing.document_id: filing
            for filing in sorted(filings, key=lambda f: (f.ticker, f.fiscal_year, f.document_id))
        }
        self._sections: dict[str, dict[str, Section]] = {
            document_id: {} for document_id in self._filings
        }
        directory = Path(processed_dir)
        for row in pq.read_table(directory / "sections.parquet").to_pylist():
            section = Section.model_validate(row)
            filing = self._filings.get(section.document_id)
            if filing is None or section.source_url != filing.source_url:
                raise CorpusError(f"Section provenance conflicts with manifest: {section.document_id}")
            sections = self._sections[section.document_id]
            if section.section_id in sections:
                raise CorpusError(f"Duplicate section: {section.document_id}/{section.section_id}")
            sections[section.section_id] = section

        self._chunks = load_chunks(directory / "chunks.parquet")
        self._section_chunks: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for chunk in self._chunks:
            self._validate_chunk(chunk)
            self._section_chunks[chunk.document_id, chunk.section_id].append(chunk)

        for document_id, sections in self._sections.items():
            if not sections:
                raise CorpusError(f"Missing sections for {document_id}")
            for section_id in sections:
                chunks = self._section_chunks.get((document_id, section_id))
                if not chunks:
                    raise CorpusError(f"Missing chunks for {document_id}/{section_id}")
                chunks.sort(key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))

    def _validate_chunk(self, chunk: Chunk) -> None:
        filing = self._filings.get(chunk.document_id)
        section = self._sections.get(chunk.document_id, {}).get(chunk.section_id)
        if filing is None or section is None:
            raise CorpusError(f"Chunk has no corresponding filing and section: {chunk.chunk_id}")
        if (chunk.ticker, chunk.fiscal_year, chunk.company_name, chunk.source_url) != (
            filing.ticker, filing.fiscal_year, filing.company_name, filing.source_url
        ) or chunk.section_name != section.section_name:
            raise CorpusError(f"Chunk metadata conflicts with filing or section: {chunk.chunk_id}")
        if (
            chunk.char_end > len(section.text)
            or section.text[chunk.char_start:chunk.char_end] != chunk.text
        ):
            raise CorpusError(f"Chunk text conflicts with section offsets: {chunk.chunk_id}")

    @cached_property
    def snapshot_identity(self) -> str:
        """Fingerprint loaded canonical records independently of file paths and Parquet encoding."""
        digest = sha256(b"EvidenceRoute canonical corpus v1\0")
        records = (sorted(self._filings.values(), key=lambda item: item.document_id),
                   [section for key in sorted(self._sections) for _, section in sorted(self._sections[key].items())],
                   sorted(self._chunks, key=lambda item: item.chunk_id))
        for collection in records:
            digest.update(len(collection).to_bytes(8, "big"))
            for record in collection:
                data = record.model_dump_json().encode("utf-8")
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
        return f"sha256:{digest.hexdigest()}"

    @cached_property
    def _searcher(self) -> BM25Retriever:
        return BM25Retriever(self._chunks)

    def list_companies(self, *, offset: int = 0, limit: int = 25) -> Page[CompanyCoverage]:
        """List tickers, latest recorded company names, and available fiscal years."""
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        by_ticker: dict[str, list[Filing]] = defaultdict(list)
        for filing in self._filings.values():
            by_ticker[filing.ticker].append(filing)
        companies = [
            CompanyCoverage(ticker, filings[-1].company_name,
                            tuple(sorted({f.fiscal_year for f in filings})))
            for ticker, filings in by_ticker.items()
        ]
        return _page(companies, offset, limit)

    def list_filings(
        self, *, ticker: str | None = None, fiscal_year: int | None = None,
        offset: int = 0, limit: int = 10,
    ) -> Page[Filing]:
        """List filing metadata sorted by ticker, year and document ID."""
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        ticker = ticker.strip().upper() if ticker is not None else None
        filings = [
            filing for filing in self._filings.values()
            if (ticker is None or filing.ticker.upper() == ticker)
            and (fiscal_year is None or filing.fiscal_year == fiscal_year)
        ]
        page = _page(filings, offset, limit)
        return Page(tuple(f.model_copy() for f in page.items), page.total, page.next_offset)

    def get_filing(self, document_id: str) -> Filing:
        """Look up a filing by its original ID. Unknown IDs raise KeyError."""
        return self._filings[document_id].model_copy()

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Look up an original chunk by ID. Unknown IDs raise KeyError."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise KeyError(f"Unknown chunk ID: {chunk_id}")
        return chunk.model_copy()

    def _section_metadata(self, section: Section) -> SectionMetadata:
        filing = self._filings[section.document_id]
        return SectionMetadata(
            section.document_id, section.section_id, section.section_name, filing.ticker,
            filing.fiscal_year, section.source_url, len(section.text),
            len(self._section_chunks[section.document_id, section.section_id]),
        )

    def list_sections(
        self, document_id: str, *, offset: int = 0, limit: int = 10,
    ) -> Page[SectionMetadata]:
        """List a filing's sections in processed-file order, without their text."""
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        sections = self._sections[document_id].values()
        return _page([self._section_metadata(section) for section in sections], offset, limit)

    def get_section(
        self, document_id: str, section_id: str, *, offset: int = 0, limit: int = MAX_SECTION_CHARS,
    ) -> SectionWindow:
        """Read exact processed text at a character offset within one section."""
        _validate_page(offset, limit, MAX_SECTION_CHARS)
        section = self._sections[document_id][section_id]
        end = offset + limit
        return SectionWindow(
            self._section_metadata(section), section.text[offset:end], offset,
            end if end < len(section.text) else None,
        )

    def read_chunks(
        self, document_id: str, section_id: str | None = None, *,
        offset: int = 0, limit: int = MAX_CHUNK_ROWS,
        _allowed_sections: frozenset[tuple[str, str]] | None = None,
    ) -> Page[Chunk]:
        """Read whole chunks in processed section order, then by chunk index."""
        _validate_page(offset, limit, MAX_CHUNK_ROWS)
        sections = self._sections[document_id]
        section_ids = sections if section_id is None else (sections[section_id].section_id,)
        chunks = [
            chunk for sid in section_ids
            if _allowed_sections is None or (document_id, sid) in _allowed_sections
            for chunk in self._section_chunks[document_id, sid]
        ]
        page = _page(chunks, offset, limit)
        return Page(tuple(c.model_copy() for c in page.items), page.total, page.next_offset)

    def search_chunks(
        self, query: str, *, filters: ChunkFilter | None = None,
        offset: int = 0, limit: int = MAX_CHUNK_ROWS,
        _allowed_sections: frozenset[tuple[str, str]] | None = None,
    ) -> Page[RetrievalResult]:
        """Rank keyword matches by BM25 score, breaking ties by chunk ID."""
        _validate_page(offset, limit, MAX_CHUNK_ROWS)
        if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query must be a string of at most {MAX_QUERY_CHARS} characters")
        if not query.strip():
            return Page((), 0, None)
        scores = self._searcher.scores(query, filters)
        ranked = sorted(
            (index for index in scores if _allowed_sections is None or
             (self._chunks[index].document_id, self._chunks[index].section_id) in _allowed_sections),
            key=lambda index: (-scores[index], self._chunks[index].chunk_id),
        )
        page = _page(ranked, offset, limit)
        hits = tuple(
            RetrievalResult(chunk=self._chunks[index].model_copy(), score=scores[index])
            for index in page.items
        )
        return Page(hits, page.total, page.next_offset)
