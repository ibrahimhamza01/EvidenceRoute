"""Performance metrics for SEC research pipelines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGMetrics:
    """Timing information for one Standard RAG request."""

    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float
