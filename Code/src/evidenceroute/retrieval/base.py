"""Shared retrieval contracts and deterministic ordering helpers."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Every retriever returns validated ``RetrievalResult`` objects, best first."""

    def search(self, query: str, top_k: int = 10,
               filters: ChunkFilter | None = None) -> list[RetrievalResult]:
        ...


def validate_top_k(top_k: int) -> int:
    # bool is an int subclass; reject it explicitly.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k


def is_blank(query: str) -> bool:
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}")
    return not query.strip()


def rank_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    """Order by score descending, then ``chunk_id`` ascending (deterministic tie-break)."""
    return sorted(results, key=lambda r: (-r.score, r.chunk.chunk_id))
