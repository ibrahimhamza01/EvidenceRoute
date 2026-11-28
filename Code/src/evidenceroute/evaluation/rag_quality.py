"""RAG-specific citation quality metrics."""

from collections.abc import Sequence

from evidenceroute.research.citations import extract_citation_ids
from evidenceroute.schemas import RetrievalResult


def citation_id_precision(
    answer: str,
    results: Sequence[RetrievalResult],
) -> float:
    """Fraction of cited IDs present in the retrieved evidence."""

    cited_ids = extract_citation_ids(answer)

    if not cited_ids:
        return 0.0

    available_ids = {
        result.chunk.chunk_id for result in results
    }

    return sum(
        chunk_id in available_ids
        for chunk_id in cited_ids
    ) / len(cited_ids)
