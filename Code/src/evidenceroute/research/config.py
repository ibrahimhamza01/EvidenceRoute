"""Validated configuration for Standard RAG."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.retrieval.filters import ChunkFilter


class StandardRAGConfig(BaseModel):
    """Configuration for SEC Standard RAG execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_path: Path = Path("data/processed/full-v1/chunks.parquet")

    retrieval_method: Literal[
        "bm25", "hybrid", "hybrid-rerank"
    ] = "bm25"

    embedder_name: str = "BAAI/bge-small-en-v1.5"

    index_dir: Path = Path("data/indexes/full-v1")

    rebuild_index: bool = False

    top_k: int = Field(default=5, ge=1, strict=True)

    llm_provider: Literal["mock", "claude"] = "claude"

    tickers: list[str] | None = None

    fiscal_years: list[int] | None = None

    def build_filters(self) -> ChunkFilter | None:
        """Convert configuration into retrieval filters."""

        if not self.tickers and not self.fiscal_years:
            return None

        return ChunkFilter(
            tickers=self.tickers,
            fiscal_years=self.fiscal_years,
        )
