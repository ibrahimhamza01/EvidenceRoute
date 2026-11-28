"""BM25 lexical retrieval over ``Chunk`` records.

Okapi BM25 with the non-negative IDF variant used by Lucene,
``idf = ln(1 + (N - df + 0.5) / (df + 0.5))``. The classic Robertson IDF turns
negative for terms in more than half the documents, which is common in small or
filtered corpora and would reward documents for containing very common words.

IDF and average document length are computed over the whole corpus. Filters only
restrict which chunks are scored, so scores remain comparable across filters. The
chunks matching each metadata value are indexed once at construction, so a filtered
search intersects small sets instead of checking every chunk.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.retrieval.base import is_blank, rank_results, validate_top_k
from evidenceroute.retrieval.corpus import ChunkCorpus, as_corpus
from evidenceroute.retrieval.filters import ChunkFilter, filter_values
from evidenceroute.retrieval.tokenization import tokenize
from evidenceroute.schemas import Chunk, RetrievalResult


class BM25Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = Field(default=1.5, ge=0)
    b: float = Field(default=0.75, ge=0, le=1)
    remove_stopwords: bool = True

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        ).hexdigest()


class BM25Retriever:
    def __init__(self, chunks: ChunkCorpus | Iterable[Chunk], config: BM25Config | None = None):
        self.corpus = as_corpus(chunks)
        self.config = config or BM25Config()
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, chunk in enumerate(self.corpus):
            counts = Counter(tokenize(chunk.text, remove_stopwords=self.config.remove_stopwords))
            self._doc_lengths.append(sum(counts.values()))
            for term in sorted(counts):
                self._postings.setdefault(term, []).append((doc_index, counts[term]))
        self._filter_index: dict[str, dict[object, frozenset[int]]] = {}
        groups: dict[str, dict[object, set[int]]] = {}
        for doc_index, chunk in enumerate(self.corpus):
            for field, value in filter_values(chunk).items():
                groups.setdefault(field, {}).setdefault(value, set()).add(doc_index)
        for field, by_value in groups.items():
            self._filter_index[field] = {v: frozenset(ids) for v, ids in by_value.items()}
        n_docs = len(self.corpus)
        self._avgdl = (sum(self._doc_lengths) / n_docs) if n_docs else 0.0
        self._idf = {
            term: math.log(1.0 + (n_docs - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.corpus)

    def query_terms(self, query: str) -> list[str]:
        """Unique query terms in first-occurrence order."""
        return list(dict.fromkeys(tokenize(query, remove_stopwords=self.config.remove_stopwords)))

    def scores(self, query: str, filters: ChunkFilter | None = None) -> dict[int, float]:
        """Positive BM25 scores keyed by corpus position, for chunks passing ``filters``."""
        k1, b = self.config.k1, self.config.b
        allowed = self.allowed_positions(filters)
        if allowed is not None and not allowed:
            return {}
        totals: dict[int, float] = {}
        for term in self.query_terms(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_index, tf in self._postings[term]:
                if allowed is not None and doc_index not in allowed:
                    continue
                length_norm = 1.0 - b + b * (self._doc_lengths[doc_index] / self._avgdl)
                totals[doc_index] = totals.get(doc_index, 0.0) + idf * (
                    tf * (k1 + 1.0) / (tf + k1 * length_norm)
                )
        return {i: s for i, s in totals.items() if s > 0.0}

    def allowed_positions(self, filters: ChunkFilter | None) -> frozenset[int] | None:
        """Corpus positions passing ``filters``; ``None`` means no constraint."""
        if filters is None or filters.is_empty():
            return None
        allowed: frozenset[int] | None = None
        for field, values in filters.conditions().items():
            index = self._filter_index.get(field, {})
            ids = frozenset().union(*(index.get(v, frozenset()) for v in values))
            allowed = ids if allowed is None else allowed & ids
            if not allowed:
                return frozenset()
        return allowed

    def search(self, query: str, top_k: int = 10,
               filters: ChunkFilter | None = None) -> list[RetrievalResult]:
        validate_top_k(top_k)
        if is_blank(query) or not len(self.corpus):
            return []
        results = [RetrievalResult(chunk=self.corpus[i], score=score)
                   for i, score in self.scores(query, filters).items()]
        return rank_results(results)[:top_k]
