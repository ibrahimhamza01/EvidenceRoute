"""Hybrid evidence retrieval over ``Chunk`` records (BM25 + dense + RRF, optional reranking).

Heavy dependencies are imported lazily: ``qdrant_client`` loads with the dense module,
and ``sentence_transformers`` only when a ``SentenceTransformerEmbedder`` or
``SentenceTransformerCrossEncoder`` is created.
"""
from evidenceroute.retrieval.base import Retriever
from evidenceroute.retrieval.bm25 import BM25Config, BM25Retriever
from evidenceroute.retrieval.corpus import (
    ChunkCorpus,
    CorpusError,
    load_chunks,
    load_chunks_jsonl,
    load_chunks_parquet,
)
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.retrieval.hybrid import (
    FusedHit,
    HybridConfig,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from evidenceroute.retrieval.rerank import (
    RerankConfig,
    RerankerError,
    RerankerUnavailableError,
    RerankingRetriever,
)

__all__ = [
    "BM25Config",
    "BM25Retriever",
    "ChunkCorpus",
    "ChunkFilter",
    "CorpusError",
    "FusedHit",
    "HybridConfig",
    "HybridRetriever",
    "RerankConfig",
    "RerankerError",
    "RerankerUnavailableError",
    "RerankingRetriever",
    "Retriever",
    "load_chunks",
    "load_chunks_jsonl",
    "load_chunks_parquet",
    "reciprocal_rank_fusion",
]
