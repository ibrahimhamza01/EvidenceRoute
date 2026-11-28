"""Connect SEC retrieval with Standard RAG answer generation."""

import logging
import time

from evidenceroute.research.llm_client import LLMClient
from evidenceroute.research.metrics import RAGMetrics
from evidenceroute.research.standard_rag import (
    ABSTENTION_MESSAGE,
    BUDGET_ERROR_MESSAGE,
    SERVICE_ERROR_MESSAGE,
    generate_rag_answer,
)
from evidenceroute.retrieval.base import Retriever, validate_top_k
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import ResearchResponse


logger = logging.getLogger(__name__)

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

    try:
        evidence = retriever.search(
            query=question,
            top_k=top_k,
            filters=filters,
        )
    except Exception:
        logger.exception("SEC evidence retrieval failed")

        end = time.perf_counter()

        if metrics_callback is not None:
            metrics_callback(RAGMetrics(
                retrieval_seconds=end - start,
                generation_seconds=0.0,
                total_seconds=end - start,
                status="retrieval_error",
            ))

        return ResearchResponse(
            question=question,
            answer=RETRIEVAL_ERROR_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    retrieval_seconds = time.perf_counter() - start

    if evidence is None:
        evidence = []

    llm_results = []

    generation_start = time.perf_counter()

    response = generate_rag_answer(
        question=question,
        evidence=evidence,
        llm_client=llm_client,
        llm_result_callback=llm_results.append,
    )

    end = time.perf_counter()

    if not evidence:
        status = "no_evidence"
    elif response.answer == BUDGET_ERROR_MESSAGE:
        status = "budget_exhausted"
    elif response.answer == SERVICE_ERROR_MESSAGE:
        status = "generation_error"
    elif response.verification_status == "unsupported":
        status = "unsupported"
    elif response.answer == ABSTENTION_MESSAGE:
        status = "no_evidence"
    else:
        status = "success"

    llm_result = llm_results[0] if llm_results else None

    input_tokens = (
        llm_result.input_tokens
        if llm_result and type(llm_result.input_tokens) is int
        else None
    )
    output_tokens = (
        llm_result.output_tokens
        if llm_result and type(llm_result.output_tokens) is int
        else None
    )

    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )

    if metrics_callback is not None:
        metrics_callback(RAGMetrics(
            retrieval_seconds=retrieval_seconds,
            generation_seconds=end - generation_start,
            total_seconds=end - start,
            status=status,
            evidence_count=len(evidence),
            citation_count=len(response.citations),
            model_name=(
                llm_result.model_name
                if llm_result and isinstance(llm_result.model_name, str)
                else None
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            llm_calls=(
                1 if evidence and status != "budget_exhausted"
                else 0
            ),
            llm_latency_seconds=(
                llm_result.latency_seconds
                if llm_result and type(llm_result.latency_seconds) in (int, float)
                else None
            ),
        ))

    return response
