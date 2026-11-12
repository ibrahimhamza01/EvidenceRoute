"""Dense semantic retrieval: SentenceTransformer embeddings stored in Qdrant.

Local development uses Qdrant's embedded mode, either in memory or persisted to a
directory, so no server is required. Passing a ``QdrantClient`` for a server or
Qdrant Cloud uses the same code path.

Reproducibility: the collection is rebuilt from the chunk corpus in ``chunk_id``
order with deterministic point IDs. An index manifest (embedding model, dimension,
corpus fingerprint, schema version) is stored in the collection metadata, and
``ensure_index`` rebuilds whenever the stored manifest differs from the current one.
"""
from __future__ import annotations

import hashlib
import uuid
import warnings
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient, models

from evidenceroute.retrieval.base import is_blank, rank_results, validate_top_k
from evidenceroute.retrieval.corpus import ChunkCorpus, as_corpus
from evidenceroute.retrieval.filters import ChunkFilter
from evidenceroute.retrieval.tokenization import tokenize
from evidenceroute.schemas import Chunk, RetrievalResult

INDEX_SCHEMA_VERSION = 1
DEFAULT_COLLECTION = "evidenceroute_chunks"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 models expect this instruction on queries, not on passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/ibrahimhamza01/EvidenceRoute/chunk")
FILTER_FIELDS = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
    "document_id": models.PayloadSchemaType.KEYWORD,
    "section_id": models.PayloadSchemaType.KEYWORD,
}


class DenseIndexError(RuntimeError):
    """The vector index is missing, stale, or inconsistent with the corpus."""


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """L2-normalized SentenceTransformer embeddings (cosine similarity in Qdrant)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, revision: str | None = None,
                 device: str | None = None, batch_size: int = 32,
                 query_prefix: str | None = None, document_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        # sentence-transformers >= 6 renamed the accessor; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        dim = get_dim()
        if not dim:
            raise DenseIndexError(f"Could not determine embedding dimension for {model_name}")
        self._dimension = int(dim)

    @property
    def model_id(self) -> str:
        parts = [f"sentence-transformers:{self.model_name}"]
        if self.revision:
            parts.append(f"@{self.revision}")
        if self.query_prefix or self.document_prefix:
            prefixes = f"{self.query_prefix}|{self.document_prefix}".encode("utf-8")
            parts.append(f"#prefix={hashlib.sha256(prefixes).hexdigest()[:12]}")
        return "".join(parts)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), batch_size=self.batch_size,
                                             normalize_embeddings=True,
                                             convert_to_numpy=True,
                                             show_progress_bar=False), dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing, for offline tests and smoke runs.

    This is *not* a semantic model. It lets the Qdrant code path run without
    downloading weights. Use ``SentenceTransformerEmbedder`` for real retrieval.
    """

    def __init__(self, dimension: int = 256):
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"hashing-bow:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self._dimension] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else \
            np.zeros((0, self._dimension), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class DenseIndexManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = INDEX_SCHEMA_VERSION
    embedding_model: str
    dimension: int
    distance: str = "cosine"
    corpus_fingerprint: str
    chunk_count: int


def point_id(chunk_id: str) -> str:
    """Qdrant needs integer or UUID IDs; derive a stable UUID from ``chunk_id``."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def to_qdrant_filter(filters: ChunkFilter | None) -> models.Filter | None:
    if filters is None or filters.is_empty():
        return None
    must = [models.FieldCondition(key=field, match=models.MatchAny(any=list(values)))
            for field, values in filters.conditions().items()]
    return models.Filter(must=must)


class DenseRetriever:
    def __init__(self, chunks: ChunkCorpus | Iterable[Chunk], embedder: Embedder, *,
                 client: QdrantClient | None = None, index_path: Path | str | None = None,
                 collection_name: str = DEFAULT_COLLECTION, batch_size: int = 64,
                 tie_buffer: int = 10):
        if client is not None and index_path is not None:
            raise ValueError("Pass either client or index_path, not both")
        self.corpus = as_corpus(chunks)
        self.embedder = embedder
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.tie_buffer = tie_buffer
        if client is None:
            if index_path is None:
                client = QdrantClient(":memory:")
            else:
                Path(index_path).mkdir(parents=True, exist_ok=True)
                client = QdrantClient(path=str(index_path))
        self.client = client

    def manifest(self) -> DenseIndexManifest:
        return DenseIndexManifest(embedding_model=self.embedder.model_id,
                                  dimension=self.embedder.dimension,
                                  corpus_fingerprint=self.corpus.fingerprint(),
                                  chunk_count=len(self.corpus))

    def stored_manifest(self) -> DenseIndexManifest | None:
        if not self.client.collection_exists(self.collection_name):
            return None
        metadata = self.client.get_collection(self.collection_name).config.metadata or {}
        try:
            return DenseIndexManifest.model_validate(metadata)
        except ValueError:
            return None

    def is_current(self) -> bool:
        stored = self.stored_manifest()
        if stored != self.manifest():
            return False
        count = self.client.count(self.collection_name, exact=True).count
        return count == len(self.corpus)

    def build(self) -> DenseIndexManifest:
        """Drop and recreate the collection from the corpus."""
        manifest = self.manifest()
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            self.collection_name,
            vectors_config=models.VectorParams(size=manifest.dimension,
                                               distance=models.Distance.COSINE),
            metadata=manifest.model_dump(),
        )
        with warnings.catch_warnings():
            # Embedded (local) Qdrant ignores payload indexes and warns; servers use them.
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field, schema in FILTER_FIELDS.items():
                self.client.create_payload_index(self.collection_name, field, schema)
        chunks = list(self.corpus)
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start:start + self.batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            if vectors.shape != (len(batch), manifest.dimension):
                raise DenseIndexError(f"Embedder returned shape {vectors.shape}")
            points = [models.PointStruct(id=point_id(c.chunk_id), vector=v.tolist(),
                                         payload=c.model_dump(mode="json"))
                      for c, v in zip(batch, vectors)]
            self.client.upsert(self.collection_name, points=points, wait=True)
        return manifest

    def ensure_index(self) -> bool:
        """Build only if missing or stale. Returns True when a rebuild happened."""
        if self.is_current():
            return False
        self.build()
        return True

    def search(self, query: str, top_k: int = 10,
               filters: ChunkFilter | None = None) -> list[RetrievalResult]:
        validate_top_k(top_k)
        if is_blank(query) or not len(self.corpus):
            return []
        if not self.client.collection_exists(self.collection_name):
            raise DenseIndexError("Vector index has not been built; call build() or ensure_index()")
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32).tolist()
        points = self._fetch_with_complete_ties(vector, top_k, to_qdrant_filter(filters))
        results = []
        for point in points:
            chunk_id = (point.payload or {}).get("chunk_id")
            chunk = self.corpus.get(chunk_id) if chunk_id else None
            if chunk is None:
                raise DenseIndexError(f"Index returned unknown chunk {chunk_id!r}; rebuild the index")
            results.append(RetrievalResult(chunk=chunk, score=float(point.score)))
        return rank_results(results)[:top_k]

    def _fetch_with_complete_ties(self, vector: list[float], top_k: int,
                                  query_filter: models.Filter | None) -> list:
        """Fetch enough points that every point tied with the k-th score is included.

        Qdrant orders equal scores arbitrarily. If the fetched window ends inside a
        group of points tied at the cutoff score, widen it until the whole group is
        present, so the ``chunk_id`` tie-break sees every candidate.
        """
        limit = top_k + self.tie_buffer
        while True:
            points = self.client.query_points(self.collection_name, query=vector,
                                              query_filter=query_filter, limit=limit,
                                              with_payload=True).points
            if len(points) < limit or len(points) < top_k:
                return points  # every matching point was returned
            cutoff = sorted((p.score for p in points), reverse=True)[top_k - 1]
            if min(p.score for p in points) < cutoff:
                return points  # the tie group at the cutoff ends inside the window
            limit *= 2

    def close(self) -> None:
        self.client.close()
