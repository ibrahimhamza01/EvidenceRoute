"""Optional cross-encoder reranking on top of any first-stage retriever.

Pipeline: question -> first-stage retriever (usually hybrid BM25 + dense + RRF)
-> ``candidate_k`` candidates -> cross-encoder scores each (question, passage) pair
-> final ``top_k`` evidence.

A cross-encoder reads the question and the passage together, so it is more precise
than comparing separately computed embeddings, but it must run for every candidate
of every query. That is why it only reorders a small candidate set.

``RerankingRetriever`` exposes the same ``search(query, top_k, filters)`` interface as
the other retrievers and returns the original ``Chunk`` objects unchanged, so SEC
metadata and evidence text are preserved. Reranking is switched off with
``RerankConfig(enabled=False)``, which returns the first-stage ranking unchanged.
When reranking is enabled but the model cannot be loaded or scored, a
``RerankerUnavailableError`` is raised with the underlying cause; the pipeline never
silently falls back to an unreranked ranking.

``sentence_transformers`` is imported only when a ``SentenceTransformerCrossEncoder``
is created, so this module loads without model dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.retrieval.base import Retriever, is_blank, validate_top_k
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import Chunk, RetrievalResult

# Default chosen from the gold-evidence evaluation on full-v1 (59 answerable questions;
# see Code/docs/retrieval.md). Over hybrid retrieval, bge-reranker-base raised MRR@10
# from 0.372 to 0.528 without filters and from 0.470 to 0.588 with ticker/year filters
# (simple and multi-source questions), while cross-encoder/ms-marco-MiniLM-L-6-v2
# changed MRR@10 much less and lowered Recall@10 slightly. The cost is
# latency: about 1.8 s per query for 50 candidates on Apple Silicon (MPS), versus
# 0.33 s for MiniLM. Strategies that search many times can lower candidate_k or set
# RerankConfig(enabled=False).
DEFAULT_RERANKER = "BAAI/bge-reranker-base"
DEFAULT_MAX_LENGTH = 512


class RerankerError(RuntimeError):
    """The cross-encoder returned unusable scores."""


class RerankerUnavailableError(RerankerError):
    """Reranking is enabled but the cross-encoder cannot be loaded or run."""


@runtime_checkable
class CrossEncoderScorer(Protocol):
    @property
    def model_id(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


def _unavailable(model_name: str, action: str, exc: BaseException) -> RerankerUnavailableError:
    return RerankerUnavailableError(
        f"Cross-encoder reranker '{model_name}' could not {action}: "
        f"{type(exc).__name__}: {exc}. "
        "Check that sentence-transformers is installed and the model weights can be "
        "downloaded or are in the Hugging Face cache, or disable reranking with "
        "RerankConfig(enabled=False)."
    )


class SentenceTransformerCrossEncoder:
    """Pretrained cross-encoder from sentence-transformers (one relevance score per pair)."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, *, revision: str | None = None,
                 device: str | None = None, batch_size: int = 32,
                 max_length: int = DEFAULT_MAX_LENGTH):
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name, revision=revision, device=device,
                                       max_length=max_length)
        except Exception as exc:  # ImportError, OSError, network and HF errors
            raise _unavailable(model_name, "be loaded", exc) from exc

    @property
    def model_id(self) -> str:
        suffix = f"@{self.revision}" if self.revision else ""
        return f"cross-encoder:{self.model_name}{suffix}#max_length={self.max_length}"

    @property
    def device(self) -> str:
        return str(getattr(self._model, "device", "n/a"))

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        try:
            scores = self._model.predict([(query, p) for p in passages],
                                         batch_size=self.batch_size,
                                         show_progress_bar=False, convert_to_numpy=True)
        except Exception as exc:
            raise _unavailable(self.model_name, "score candidates", exc) from exc
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class RerankConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    candidate_k: int = Field(default=50, ge=1)
    model_name: str = DEFAULT_RERANKER


@dataclass(frozen=True)
class RerankedHit:
    """Reranking details for analysis. The public API returns ``RetrievalResult`` only."""

    chunk: Chunk
    score: float
    first_stage_rank: int
    first_stage_score: float

    def to_result(self) -> RetrievalResult:
        return RetrievalResult(chunk=self.chunk, score=self.score)


class RerankingRetriever:
    """Retrieve ``candidate_k`` candidates, rerank them with a cross-encoder, keep ``top_k``.

    Ordering is deterministic: rerank score descending, then first-stage rank, then
    ``chunk_id``. When disabled, the first-stage ranking and scores are returned as is.
    """

    def __init__(self, retriever: Retriever, scorer: CrossEncoderScorer | None = None,
                 config: RerankConfig | None = None):
        self.retriever = retriever
        self.config = config or RerankConfig()
        if self.config.enabled and scorer is None:
            raise ValueError("Reranking is enabled but no cross-encoder scorer was provided; "
                             "pass a scorer or use RerankConfig(enabled=False)")
        self.scorer = scorer

    def search_with_details(self, query: str, top_k: int = 10,
                            filters: ChunkFilter | None = None) -> list[RerankedHit]:
        validate_top_k(top_k)
        if is_blank(query):
            return []
        if not self.config.enabled:
            # Ask for top_k directly so the result is identical to the first stage alone
            # (a deeper hybrid request can change RRF order near the top).
            return [RerankedHit(chunk=c.chunk, score=c.score, first_stage_rank=i,
                                first_stage_score=c.score)
                    for i, c in enumerate(self.retriever.search(query, top_k, filters), start=1)]
        depth = max(self.config.candidate_k, top_k)
        candidates = self.retriever.search(query, depth, filters)
        if not candidates:
            return []
        scores = self.scorer.score(query, [c.chunk.text for c in candidates])
        if scores.shape != (len(candidates),):
            raise RerankerError(f"Cross-encoder returned {scores.shape} scores "
                                f"for {len(candidates)} candidates")
        if not all(math.isfinite(float(s)) for s in scores):
            raise RerankerError("Cross-encoder returned non-finite scores")
        hits = [RerankedHit(chunk=c.chunk, score=float(s), first_stage_rank=i,
                            first_stage_score=c.score)
                for i, (c, s) in enumerate(zip(candidates, scores), start=1)]
        hits.sort(key=lambda h: (-h.score, h.first_stage_rank, h.chunk.chunk_id))
        return hits[:top_k]

    def search(self, query: str, top_k: int = 10,
               filters: ChunkFilter | None = None) -> list[RetrievalResult]:
        return [hit.to_result() for hit in self.search_with_details(query, top_k, filters)]


def build_reranking_retriever(retriever: Retriever, config: RerankConfig | None = None, *,
                              device: str | None = None) -> RerankingRetriever:
    """Wrap ``retriever`` according to ``config``, loading the cross-encoder only if enabled."""
    config = config or RerankConfig()
    scorer = SentenceTransformerCrossEncoder(config.model_name, device=device) \
        if config.enabled else None
    return RerankingRetriever(retriever, scorer, config)
