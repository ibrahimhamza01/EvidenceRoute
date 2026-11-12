"""Hybrid retrieval: BM25 + dense rankings fused with Reciprocal Rank Fusion.

``rrf_score(chunk) = sum over rankings of 1 / (rrf_k + rank)``, with 1-based ranks
(Cormack, Clarke and Buettcher, 2009). Each ranking contributes at most once per
chunk; duplicates are removed by ``chunk_id``. Ties are broken by the best single
rank, then by ``chunk_id``, so output order is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.retrieval.base import Retriever, is_blank, validate_top_k
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.schemas import Chunk, RetrievalResult


class HybridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rrf_k: int = Field(default=60, ge=1)
    candidate_k: int = Field(default=50, ge=1)


@dataclass(frozen=True)
class FusedHit:
    """Fusion details for analysis. The public API returns ``RetrievalResult`` only."""

    chunk: Chunk
    score: float
    ranks: Mapping[str, int] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())

    def to_result(self) -> RetrievalResult:
        return RetrievalResult(chunk=self.chunk, score=self.score)


def reciprocal_rank_fusion(rankings: Mapping[str, Sequence[RetrievalResult]],
                           rrf_k: int = 60) -> list[FusedHit]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    chunks: dict[str, Chunk] = {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name in sorted(rankings):  # fixed summation order -> identical floats every run
        rank = 0
        seen: set[str] = set()
        for result in rankings[name]:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                continue  # duplicate within one ranking: keep its best position only
            seen.add(chunk_id)
            rank += 1
            chunks.setdefault(chunk_id, result.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            ranks.setdefault(chunk_id, {})[name] = rank
    hits = [FusedHit(chunk=chunks[cid], score=scores[cid], ranks=dict(ranks[cid]))
            for cid in chunks]
    hits.sort(key=lambda h: (-h.score, h.best_rank, h.chunk.chunk_id))
    return hits


class HybridRetriever:
    def __init__(self, retrievers: Mapping[str, Retriever], config: HybridConfig | None = None):
        if not retrievers:
            raise ValueError("At least one retriever is required")
        self.retrievers = dict(retrievers)
        self.config = config or HybridConfig()

    def search_with_details(self, query: str, top_k: int = 10,
                            filters: ChunkFilter | None = None) -> list[FusedHit]:
        validate_top_k(top_k)
        if is_blank(query):
            return []
        depth = max(self.config.candidate_k, top_k)
        rankings = {name: r.search(query, depth, filters) for name, r in self.retrievers.items()}
        return reciprocal_rank_fusion(rankings, self.config.rrf_k)[:top_k]

    def search(self, query: str, top_k: int = 10,
               filters: ChunkFilter | None = None) -> list[RetrievalResult]:
        return [hit.to_result() for hit in self.search_with_details(query, top_k, filters)]
