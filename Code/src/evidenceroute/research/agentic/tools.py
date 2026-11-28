"""Read-only SEC corpus tools for agentic research."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.schemas import Chunk, RetrievalResult
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.retrieval.base import Retriever


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    document_ids: list[str] | None = None
    section_ids: list[str] | None = None

    def build_filters(self) -> ChunkFilter:
        return ChunkFilter(
            tickers=self.tickers,
            fiscal_years=self.fiscal_years,
            document_ids=self.document_ids,
            section_ids=self.section_ids,
        )


class SECTools:
    """Expose controlled, read-only corpus operations."""

    def __init__(
        self,
        corpus: Sequence[Chunk],
        retriever: Retriever,
    ):
        self.corpus = list(corpus)
        self.retriever = retriever

        self.documents = {}

        for chunk in self.corpus:
            self.documents.setdefault(
                chunk.document_id, []
            ).append(chunk)

    def search_chunks(
        self,
        request: SearchRequest,
    ) -> list[RetrievalResult]:

        results = self.retriever.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.build_filters(),
        )

        return list(results or [])

    def get_filings(
        self,
        tickers: list[str] | None = None,
        fiscal_years: list[int] | None = None,
    ) -> list[dict]:

        filters = ChunkFilter(
            tickers=tickers,
            fiscal_years=fiscal_years,
        )

        filings = []

        for document_id, chunks in self.documents.items():
            chunk = chunks[0]

            if filters.matches(chunk):
                filings.append(
                    self.get_document_metadata(document_id)
                )

        return sorted(
            filings,
            key=lambda x: (
                x["ticker"],
                x["fiscal_year"],
                x["document_id"],
            ),
        )

    def get_document_metadata(
        self,
        document_id: str,
    ) -> dict:

        chunks = self.documents.get(document_id)

        if not chunks:
            raise ValueError("Unknown document ID")

        chunk = chunks[0]

        return {
            "document_id": document_id,
            "ticker": chunk.ticker,
            "company_name": chunk.company_name,
            "fiscal_year": chunk.fiscal_year,
            "source_url": chunk.source_url,
            "chunk_count": len(chunks),
            "section_ids": sorted({
                c.section_id for c in chunks
            }),
        }

    def search_sections(
        self,
        request: SearchRequest,
    ) -> list[dict]:

        results = self.search_chunks(request)
        sections = {}

        for result in results:
            chunk = result.chunk
            key = (chunk.document_id, chunk.section_id)

            if key not in sections:
                sections[key] = {
                    "document_id": chunk.document_id,
                    "section_id": chunk.section_id,
                    "section_name": chunk.section_name,
                    "ticker": chunk.ticker,
                    "fiscal_year": chunk.fiscal_year,
                    "score": result.score,
                    "source_url": chunk.source_url,
                }

        return list(sections.values())

    def retrieve_section(
        self,
        document_id: str,
        section_id: str,
        limit: int = 20,
    ) -> list[Chunk]:

        if not 1 <= limit <= 100:
            raise ValueError("Invalid section retrieval limit")

        chunks = self.documents.get(document_id, [])

        matches = [
            chunk for chunk in chunks
            if chunk.section_id.lower() == section_id.lower()
        ]

        return sorted(
            matches,
            key=lambda chunk: chunk.chunk_index,
        )[:limit]
