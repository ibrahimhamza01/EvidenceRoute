"""Connect SEC retrieval with Standard RAG answer generation."""

import time


from evidenceroute.research.llm_client import LLMClient
from evidenceroute.research.metrics import RAGMetrics
from evidenceroute.research.standard_rag import generate_rag_answer
from evidenceroute.retrieval.base import Retriever, validate_top_k
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import ResearchResponse

RETRIEVAL_ERROR_MESSAGE = "Unable to retrieve SEC evidence at this time."



def run_standard_rag(
    question: str,
    retriever: Retriever,
    llm_client: LLMClient,
    *,
    top_k: int = 10,
    filters: ChunkFilter | None = None,
    metrics_callback=None,
) -> ResearchResponse:
    """Retrieve relevant SEC evidence and generate a cited answer."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    validate_top_k(top_k)

    start = time.perf_counter()
    retrieval_start = start

    try:
        evidence = retriever.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )
    except Exception:
        return ResearchResponse(
            question=question,
            answer=RETRIEVAL_ERROR_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    retrieval_seconds = time.perf_counter() - retrieval_start

    if evidence is None:
        evidence = []

    generation_start = time.perf_counter()
    response = generate_rag_answer(
        question=question,
        evidence=evidence,
        llm_client=llm_client,
    )
    end = time.perf_counter()

    if metrics_callback is not None:
        metrics_callback(RAGMetrics(
            retrieval_seconds=retrieval_seconds,
            generation_seconds=end - generation_start,
            total_seconds=end - start,
        ))

    return response
