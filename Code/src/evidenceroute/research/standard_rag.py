
"""Standard RAG answer generation for SEC filings."""

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

    return ResearchResponse(
        question=question,
        answer=answer,
        strategy="rag",
        citations=[],
        verification_status="not_checked",
    )
