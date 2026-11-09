from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Filing(BaseModel):
    document_id: str
    company_name: str
    ticker: str
    cik: str
    accession_number: str
    form_type: Literal["10-K"]
    fiscal_year: int
    report_period_end: date
    filing_date: date
    primary_document: str
    source_url: str
    raw_sha256: str | None = None


class Section(BaseModel):
    document_id: str
    section_id: str
    section_name: str
    text: str = Field(min_length=1)
    source_url: str
    

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    section_id: str
    section_name: str
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    source_url: str


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float


class RouterResult(BaseModel):
    scope: Literal["simple", "multi_source", "corpus_wide"]
    confidence: float | None = Field(default=None, ge=0, le=1)


class ResearchResponse(BaseModel):
    question: str
    answer: str
    strategy: Literal["rag", "agentic_rag", "rlm"]
    citations: list[Chunk]
    verification_status: Literal[
        "not_checked", "supported", "partial", "unsupported"
    ] = "not_checked"