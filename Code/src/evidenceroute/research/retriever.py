"""Shared retrieval construction for SEC research workflows."""

from pathlib import Path

from evidenceroute.retrieval import BM25Retriever
from evidenceroute.retrieval.dense import DEFAULT_MODEL


RETRIEVAL_METHODS = (
    "bm25",
    "hybrid",
    "hybrid-rerank",
)

DEFAULT_INDEX_DIR = Path("data/indexes/full-v1")


def build_research_retriever(
    corpus,
    method="bm25",
    *,
    embedder_name=DEFAULT_MODEL,
    index_dir=DEFAULT_INDEX_DIR,
    rebuild=False,
):
    """Construct a research retriever using existing retrieval components."""

    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unsupported retrieval method: {method}")

    if method == "bm25":
        return BM25Retriever(corpus)

    from evidenceroute.retrieval.dense import (
        HashingEmbedder,
        SentenceTransformerEmbedder,
    )
    from evidenceroute.retrieval.factory import (
        build_hybrid_retriever,
    )

    embedder = (
        HashingEmbedder()
        if embedder_name == "hashing"
        else SentenceTransformerEmbedder(embedder_name)
    )

    slug = (
        embedder.model_id
        .replace("/", "_")
        .replace(":", "_")[:80]
    )

    retriever = build_hybrid_retriever(
        corpus,
        embedder,
        index_path=Path(index_dir) / slug,
        rebuild=rebuild,
    )

    if method == "hybrid-rerank":
        from evidenceroute.retrieval.rerank import (
            RerankConfig,
            build_reranking_retriever,
        )

        retriever = build_reranking_retriever(
            retriever,
            RerankConfig(enabled=True),
        )

    return retriever
