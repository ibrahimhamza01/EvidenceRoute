"""Retrieval ranking and citation-ID quality metrics."""

from collections.abc import Sequence

from evidenceroute.research.citations import extract_citation_ids
from evidenceroute.schemas import RetrievalResult


def recall_at_k(
    results: Sequence[RetrievalResult],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of known relevant chunks retrieved in the top k."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0

    retrieved = {
        result.chunk.chunk_id
        for result in results[:k]
    }
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(
    results: Sequence[RetrievalResult],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant chunk in the top k."""
    if k < 1:
        raise ValueError("k must be positive")

    for rank, result in enumerate(results[:k], start=1):
        if result.chunk.chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def citation_id_precision(
    answer: str,
    results: Sequence[RetrievalResult],
) -> float:
    """Fraction of unique citation IDs present in retrieved evidence."""
    cited_ids = extract_citation_ids(answer)

    if not cited_ids:
        return 0.0

    available_ids = {
        result.chunk.chunk_id
        for result in results
    }

    valid_count = sum(
        chunk_id in available_ids
        for chunk_id in cited_ids
    )

    return valid_count / len(cited_ids)
