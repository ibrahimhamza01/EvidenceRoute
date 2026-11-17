
"""Demonstrate Standard RAG with SEC evidence and a mock LLM.

Run from the Code directory:
    python examples/demo_standard_rag.py

Requires the canonical full-v1 chunks; no API key, network or vector database required.
"""

from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.standard_rag import generate_rag_answer


class DemoLLMClient:
    """Echo supplied evidence with its real citation; no factual answer assumptions."""

    def __init__(self, chunk):
        self.chunk = chunk

    def generate(self, prompt: str) -> LLMResult:
        return LLMResult(
            text=f"{self.chunk.text.splitlines()[0]} [chunk:{self.chunk.chunk_id}]",
            model_name="demo-mock", latency_seconds=0.0, input_tokens=0, output_tokens=0,
        )


def main() -> None:
    from pathlib import Path
    from evidenceroute.retrieval import BM25Retriever, load_chunks

    path = Path(__file__).resolve().parents[2] / "data/processed/full-v1/chunks.parquet"
    corpus = load_chunks(path)
    evidence = BM25Retriever(corpus).search("financial results", top_k=1)
    chunk = evidence[0].chunk
    question = "Quote the opening line of the retrieved section."

    response = generate_rag_answer(
        question=question,
        evidence=evidence,
        llm_client=DemoLLMClient(chunk),
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
