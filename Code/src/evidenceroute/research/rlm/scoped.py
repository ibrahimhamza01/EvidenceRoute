"""Read-only view restricted to exact document and section pairs."""

from __future__ import annotations

from collections import defaultdict

from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import (
    MAX_METADATA_ROWS,
    CompanyCoverage,
    CorpusEnvironment,
    _page,
    _validate_page,
)


class ScopedCorpus:
    def __init__(self, corpus: CorpusEnvironment, partitions: tuple[SectionTarget, ...]):
        self._corpus = corpus
        self._pairs = frozenset((p.document_id, p.section_id) for p in partitions)
        self._documents = frozenset(p.document_id for p in partitions)

    def get_filing(self, document_id: str):
        self._require(document_id)
        return self._corpus.get_filing(document_id)

    def list_filings(self, *, ticker=None, fiscal_year=None, offset=0, limit=10):
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        ticker = ticker.strip().upper() if ticker is not None else None
        filings = sorted((self.get_filing(doc) for doc in self._documents),
                         key=lambda filing: (filing.ticker, filing.fiscal_year, filing.document_id))
        return _page([f for f in filings if (ticker is None or f.ticker.upper() == ticker)
                      and (fiscal_year is None or f.fiscal_year == fiscal_year)], offset, limit)

    def list_companies(self, *, offset=0, limit=MAX_METADATA_ROWS):
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        grouped = defaultdict(list)
        for document_id in sorted(self._documents):
            filing = self.get_filing(document_id)
            grouped[filing.ticker].append(filing)
        companies = [
            CompanyCoverage(ticker, max(filings, key=lambda filing: filing.fiscal_year).company_name,
                            tuple(sorted({f.fiscal_year for f in filings})))
            for ticker, filings in sorted(grouped.items())
        ]
        return _page(companies, offset, limit)

    def list_sections(self, document_id, *, offset=0, limit=10):
        self._require(document_id)
        _validate_page(offset, limit, MAX_METADATA_ROWS)
        sections = [self.get_section(doc, sid, limit=1).metadata
                    for doc, sid in sorted(self._pairs) if doc == document_id]
        return _page(sections, offset, limit)

    def get_section(self, document_id, section_id, **page):
        self._require(document_id, section_id)
        return self._corpus.get_section(document_id, section_id, **page)

    def read_chunks(self, document_id, section_id=None, **page):
        self._require(document_id, section_id)
        return self._corpus.read_chunks(document_id, section_id, _allowed_sections=self._pairs, **page)

    def get_chunk(self, chunk_id):
        chunk = self._corpus.get_chunk(chunk_id)
        self._require(chunk.document_id, chunk.section_id)
        return chunk

    def search_chunks(self, query, **page):
        return self._corpus.search_chunks(query, _allowed_sections=self._pairs, **page)

    def _require(self, document_id, section_id=None):
        if (document_id not in self._documents or
                (section_id is not None and (document_id, section_id) not in self._pairs)):
            raise KeyError("No corpus record matches the supplied IDs.")
