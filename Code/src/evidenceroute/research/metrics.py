"""Performance metrics for SEC research pipelines."""

from dataclasses import dataclass
from typing import Literal


RAGStatus = Literal[
    "success",
    "no_evidence",
    "retrieval_error",
    "generation_error",
    "budget_exhausted",
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
    model_name: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    llm_calls: int = 0
    llm_latency_seconds: float | None = None
