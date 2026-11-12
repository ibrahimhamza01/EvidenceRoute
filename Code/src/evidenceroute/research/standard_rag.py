
"""Standard RAG answer generation for SEC filings."""
from evidenceroute.research.citations import validate_citations
from collections.abc import Sequence

from evidenceroute.schemas import (
    ResearchResponse,
    RetrievalResult,
)
from evidenceroute.research.prompts import build_rag_prompt
from evidenceroute.research.llm_client import (
    LLMClient,
    LLMResult,
)


ABSTENTION_MESSAGE = (
    "Insufficient evidence in the supplied SEC filings "
    "to answer this question."
)


def generate_rag_answer(
    question: str,
    evidence: Sequence[RetrievalResult],
    llm_client: LLMClient,
) -> ResearchResponse:
    """Generate a research answer from supplied SEC evidence."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not evidence:
        return ResearchResponse(
            question=question,
            answer=ABSTENTION_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    prompt = build_rag_prompt(question, evidence)

    result: LLMResult = llm_client.generate(prompt)


    answer = result.text.strip()

    if not answer:
        answer = ABSTENTION_MESSAGE

    citations, invalid_ids = validate_citations(
        answer,
        evidence,
    )

    if invalid_ids:
        # Do not expose fabricated citation references.
        return ResearchResponse(
            question=question,
            answer=ABSTENTION_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="unsupported",
        )

    if not citations:
        # An uncited answer cannot be treated as verified.
        return ResearchResponse(
            question=question,
            answer=ABSTENTION_MESSAGE,
            strategy="rag",
            citations=[],
            verification_status="not_checked",
        )

    return ResearchResponse(
        question=question,
        answer=answer,
        strategy="rag",
        citations=citations,
        verification_status="not_checked",
    )

