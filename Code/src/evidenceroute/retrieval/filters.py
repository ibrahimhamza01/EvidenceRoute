"""Metadata filters shared by BM25, dense and hybrid retrieval.

Semantics: a field left as ``None`` places no constraint. A field with values keeps
chunks whose metadata equals any of them (OR within a field). Different fields are
combined with AND.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from evidenceroute.schemas import Chunk


def _as_tuple(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return (value,)
    values = tuple(value)
    if not values:
        raise ValueError("Filter value lists must not be empty; use None for no constraint")
    return values


class ChunkFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tickers: tuple[str, ...] | None = None
    fiscal_years: tuple[int, ...] | None = None
    document_ids: tuple[str, ...] | None = None
    section_ids: tuple[str, ...] | None = None

    @field_validator("tickers", "fiscal_years", "document_ids", "section_ids", mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        return _as_tuple(value)

    @field_validator("tickers")
    @classmethod
    def _upper(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else tuple(sorted({v.strip().upper() for v in value}))

    @field_validator("section_ids")
    @classmethod
    def _lower(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else tuple(sorted({v.strip().lower() for v in value}))

    @field_validator("document_ids")
    @classmethod
    def _dedupe(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return None if value is None else tuple(sorted({v.strip() for v in value}))

    @field_validator("fiscal_years")
    @classmethod
    def _years(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        return None if value is None else tuple(sorted(set(value)))

    def is_empty(self) -> bool:
        return all(v is None for v in self.conditions().values())

    def conditions(self) -> dict[str, tuple]:
        """Map ``Chunk`` field name to allowed values, for active constraints only."""
        fields = {
            "ticker": self.tickers,
            "fiscal_year": self.fiscal_years,
            "document_id": self.document_ids,
            "section_id": self.section_ids,
        }
        return {k: v for k, v in fields.items() if v is not None}

    def matches(self, chunk: Chunk) -> bool:
        values = filter_values(chunk)
        return all(values[field] in allowed for field, allowed in self.conditions().items())


def filter_values(chunk: Chunk) -> dict[str, Any]:
    """A chunk's filterable metadata, normalized exactly as ``ChunkFilter`` values are.

    Shared by ``ChunkFilter.matches`` and retrievers that precompute filter indexes,
    so both always apply the same matching rules.
    """
    return {
        "ticker": chunk.ticker.upper(),
        "fiscal_year": chunk.fiscal_year,
        "document_id": chunk.document_id,
        "section_id": chunk.section_id.lower(),
    }


def matches(filters: ChunkFilter | None, chunk: Chunk) -> bool:
    return filters is None or filters.matches(chunk)
