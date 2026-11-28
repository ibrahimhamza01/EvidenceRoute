"""Run Standard RAG against SEC filing evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidenceroute.research.config import StandardRAGConfig
from evidenceroute.research.budget import (
    ExecutionLimits,
    ExecutionBudget,
    BudgetedLLMClient,
)
from evidenceroute.research.llm_client import LLMResult
from evidenceroute.research.claude_client import ClaudeClient
from evidenceroute.research.pipeline import run_standard_rag
from evidenceroute.retrieval import load_chunks
from evidenceroute.research.retriever import (
    build_research_retriever,
    RETRIEVAL_METHODS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data/processed/full-v1/chunks.parquet"


class MockCitingLLM:
    """Offline LLM substitute for testing pipeline integration."""

    def generate(self, prompt: str) -> LLMResult:
        import re

        matches = re.findall(
            r"^Chunk ID:\s*(\S+)",
            prompt,
            re.MULTILINE,
        )

        if not matches:
            return LLMResult(
                text="Insufficient evidence in the supplied SEC filings to answer this question.",
                model_name="mock",
                latency_seconds=0.0,
            )

        return LLMResult(
            text=f"Offline test response [chunk:{matches[0]}].",
            model_name="mock",
            latency_seconds=0.0,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval", choices=RETRIEVAL_METHODS,
        default="bm25",
    )
    parser.add_argument(
        "--embedder", default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--index-dir", type=Path,
        default=ROOT / "data/indexes/full-v1",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument(
        "--provider",
        choices=["mock", "claude"],
        default="mock",
    )
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=32000)
    parser.add_argument("--max-execution-seconds", type=float, default=180.0)
    parser.add_argument("--llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--llm-max-retries", type=int, default=1)

    args = parser.parse_args(argv)

    config = StandardRAGConfig(
        corpus_path=args.chunks,
        retrieval_method=args.retrieval,
        embedder_name=args.embedder,
        index_dir=args.index_dir,
        rebuild_index=args.rebuild_index,
        top_k=args.top_k,
        llm_provider=args.provider,
        max_llm_calls=args.max_llm_calls,
        max_total_tokens=args.max_total_tokens,
        max_execution_seconds=args.max_execution_seconds,
        llm_timeout_seconds=args.llm_timeout_seconds,
        max_output_tokens=args.max_output_tokens,
        llm_max_retries=args.llm_max_retries,
        tickers=args.ticker,
        fiscal_years=args.year,
    )

    corpus = load_chunks(config.corpus_path)
    retriever = build_research_retriever(
        corpus,
        config.retrieval_method,
        embedder_name=config.embedder_name,
        index_dir=config.index_dir,
        rebuild=config.rebuild_index,
    )

    llm = (
        MockCitingLLM()
        if args.provider == "mock"
        else ClaudeClient(
            timeout_seconds=config.llm_timeout_seconds,
            max_tokens=config.max_output_tokens,
            max_retries=config.llm_max_retries,
        )
    )

    budget = ExecutionBudget(
        ExecutionLimits(
            max_llm_calls=config.max_llm_calls,
            max_total_tokens=config.max_total_tokens,
            max_seconds=config.max_execution_seconds,
        )
    )

    llm = BudgetedLLMClient(llm, budget)

    collected_metrics = []

    response = run_standard_rag(
        question=args.question,
        retriever=retriever,
        llm_client=llm,
        top_k=config.top_k,
        filters=config.build_filters(),
        metrics_callback=collected_metrics.append,
    )

    print(json.dumps({
        "response": response.model_dump(mode="json"),
        "budget": budget.snapshot(),
        "metrics": (
            vars(collected_metrics[0])
            if collected_metrics else None
        ),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
