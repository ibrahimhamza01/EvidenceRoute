"""Convenience constructor wiring BM25, dense and RRF from the same chunks and config."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient

from evidenceroute.retrieval.bm25 import BM25Config, BM25Retriever
from evidenceroute.retrieval.corpus import ChunkCorpus, as_corpus
from evidenceroute.retrieval.dense import DEFAULT_COLLECTION, DenseRetriever, Embedder
from evidenceroute.retrieval.hybrid import HybridConfig, HybridRetriever
from evidenceroute.schemas import Chunk


def build_hybrid_retriever(chunks: ChunkCorpus | Iterable[Chunk], embedder: Embedder, *,
                           bm25_config: BM25Config | None = None,
                           hybrid_config: HybridConfig | None = None,
                           client: QdrantClient | None = None,
                           index_path: Path | str | None = None,
                           collection_name: str = DEFAULT_COLLECTION,
                           rebuild: bool = False) -> HybridRetriever:
    corpus = as_corpus(chunks)
    dense = DenseRetriever(corpus, embedder, client=client, index_path=index_path,
                           collection_name=collection_name)
    if rebuild:
        dense.build()
    else:
        dense.ensure_index()
    return HybridRetriever({"bm25": BM25Retriever(corpus, bm25_config), "dense": dense},
                           hybrid_config)
