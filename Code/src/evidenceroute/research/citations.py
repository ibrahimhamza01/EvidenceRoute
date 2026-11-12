
"""Citation extraction and validation for Standard RAG."""

import re
from collections.abc import Sequence

from evidenceroute.schemas import Chunk, RetrievalResult


CITATION_PATTERN = re.compile(r"\[chunk:([^\[\]\s]+)\]")


def extract_citation_ids(answer: str) -> list[str]:
    """Extract unique cited chunk IDs in order of appearance."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(answer)))


def validate_citations(
    answer: str,
    evidence: Sequence[RetrievalResult],
) -> tuple[list[Chunk], list[str]]:
    """Match citations to supplied evidence.

    Returns:
        (valid_chunks, invalid_chunk_ids)
    """
    available_chunks = {
        result.chunk.chunk_id: result.chunk
        for result in evidence
    }

    valid_chunks = []
    invalid_ids = []

    for chunk_id in extract_citation_ids(answer):
        if chunk_id in available_chunks:
            valid_chunks.append(available_chunks[chunk_id])
        else:
            invalid_ids.append(chunk_id)

    return valid_chunks, invalid_ids
