
"""Standard RAG answer generation for SEC filings."""

from collections.abc import Sequence

from evidenceroute.schemas import (
    ResearchResponse,
    RetrievalResult,
)
from evidenceroute.research.prompts import build_rag_prompt
from evidenceroute.research.citations import validate_citations
from evidenceroute.research.llm_client import (
    LLMClient,
    LLMClientError,
)


ABSTENTION_MESSAGE = (
    "Insufficient evidence in the supplied SEC filings "
    "to answer this question."
)

SERVICE_ERROR_MESSAGE = (
    "Unable to generate an answer because the language "
    "model service is temporarily unavailable."
)


def _fallback_response(
    question: str,
    message: str,
    status: str = "not_checked",
) -> ResearchResponse:
    """Return a safe response without citations."""

    return ResearchResponse(
        question=question,
        answer=message,
        strategy="rag",
        citations=[],
        verification_status=status,
    )


def generate_rag_answer(
    question: str,
    evidence: Sequence[RetrievalResult],
    llm_client: LLMClient,
) -> ResearchResponse:
    """Generate an evidence-grounded research response."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not evidence:
        return _fallback_response(
            question,
            ABSTENTION_MESSAGE,
        )

    prompt = build_rag_prompt(question, evidence)

    try:
        result = llm_client.generate(prompt)
    except (LLMClientError, TimeoutError):
        return _fallback_response(
            question,
            SERVICE_ERROR_MESSAGE,
        )

    answer = result.text.strip()

    if not answer:
        return _fallback_response(
            question,
            ABSTENTION_MESSAGE,
        )

    citations, invalid_ids = validate_citations(
        answer,
        evidence,
    )

    if invalid_ids:
        return _fallback_response(
            question,
            ABSTENTION_MESSAGE,
            status="unsupported",
        )

    if not citations:
        return _fallback_response(
            question,
            ABSTENTION_MESSAGE,
        )

    return ResearchResponse(
        question=question,
        answer=answer,
        strategy="rag",
        citations=citations,
        verification_status="not_checked",
    )
