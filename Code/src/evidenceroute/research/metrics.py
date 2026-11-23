"""Performance metrics for SEC research pipelines."""

from dataclasses import dataclass
from typing import Literal


RAGStatus = Literal[
    "success",
    "no_evidence",
    "retrieval_error",
    "generation_error",
    "unsupported",
]


@dataclass(frozen=True)
class RAGMetrics:
    """Timing and outcome information for one Standard RAG request."""

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
    status: RAGStatus = "success"
    evidence_count: int = 0
    citation_count: int = 0
