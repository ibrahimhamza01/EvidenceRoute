"""Connect SEC retrieval with Standard RAG answer generation."""

from evidenceroute.research.llm_client import LLMClient
from evidenceroute.research.standard_rag import generate_rag_answer
from evidenceroute.retrieval.base import Retriever, validate_top_k
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import ResearchResponse


def run_standard_rag(
    question: str,
    retriever: Retriever,
    llm_client: LLMClient,
    *,
    top_k: int = 10,
    filters: ChunkFilter | None = None,
) -> ResearchResponse:
    """Retrieve relevant SEC evidence and generate a cited answer."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    validate_top_k(top_k)

    evidence = retriever.search(
        query=question,
        top_k=top_k,
        filters=filters,
    )

    return generate_rag_answer(
        question=question,
        evidence=evidence,
        llm_client=llm_client,
    )
