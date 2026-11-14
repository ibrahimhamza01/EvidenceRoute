
"""Demonstrate Standard RAG with SEC evidence and a mock LLM.

Run from the Code directory:
    python examples/demo_standard_rag.py

No API key, network connection, or vector database required.
"""

from evidenceroute.schemas import Chunk, RetrievalResult
from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.standard_rag import generate_rag_answer


class DemoLLMClient:
    """Return a deterministic answer without contacting Gemini."""

    def generate(self, prompt: str) -> LLMResult:
        return LLMResult(
            text=(
                "NVIDIA reported approximately 29,600 employees "
                "in 36 countries at the end of fiscal 2024. "
                "[chunk:8b7deb3e2b688baf905dd233]"
            ),
            model_name="demo-mock",
            latency_seconds=0.0,
            input_tokens=0,
            output_tokens=0,
        )


def main() -> None:
    question = (
        "How many employees did NVIDIA report at the end "
        "of fiscal year 2024, and in how many countries?"
    )

    # Verified excerpt from the SEC evidence benchmark (q0001).
    # This is an excerpt, not the complete processed chunk.
    chunk = Chunk(
        chunk_id="8b7deb3e2b688baf905dd233",
        document_id="0001045810_0001045810-24-000029",
        company_name="NVIDIA",
        ticker="NVDA",
        fiscal_year=2024,
        section_id="item_1",
        section_name="Business",
        chunk_index=0,
        text=(
            "As of the end of fiscal year 2024, we had "
            "approximately 29,600 employees in 36 countries"
        ),
        char_start=40259,
        char_end=40347,
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1045810/"
            "000104581024000029/nvda-20240128.htm"
        ),
    )

    evidence = [RetrievalResult(chunk=chunk, score=0.95)]

    response = generate_rag_answer(
        question=question,
        evidence=evidence,
        llm_client=DemoLLMClient(),
    )

    print("\n=== EvidenceRoute: Standard RAG Demo ===")
    print(f"\nQuestion: {response.question}")
    print(f"\nAnswer: {response.answer}")
    print(f"\nStrategy: {response.strategy}")
    print(f"Verification status: {response.verification_status}")

    print("\nCitations:")
    for cited_chunk in response.citations:
        print(f"  Chunk ID: {cited_chunk.chunk_id}")
        print(f"  Company: {cited_chunk.company_name}")
        print(f"  Fiscal year: {cited_chunk.fiscal_year}")
        print(f"  Source: {cited_chunk.source_url}")


if __name__ == "__main__":
    main()
