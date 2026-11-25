from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from evidenceroute.retrieval import BM25Retriever, ChunkFilter, HybridRetriever, load_chunks

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = ROOT / "data/processed/full-v1/chunks.parquet"
DEFAULT_INDEX_DIR = ROOT / "data/indexes/full-v1"
DEFAULT_QUERIES = Path(__file__).resolve().parent / "retrieval_queries.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmarks"
PACKAGES = ("sentence-transformers", "torch", "qdrant-client", "pyarrow", "numpy")


class TimedEmbedder:
    """Wraps an embedder and records time spent in document and query embedding.

    It exposes the same ``model_id`` and ``dimension`` as the wrapped embedder, so the
    dense index manifest is unchanged and an existing index is still reused.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.document_seconds = 0.0
        self.query_seconds: list[float] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        start = time.perf_counter()
        vectors = self.inner.embed_documents(texts)
        self.document_seconds += time.perf_counter() - start
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        start = time.perf_counter()
        vector = self.inner.embed_query(text)
        self.query_seconds.append(time.perf_counter() - start)
        return vector


def timed(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def summarize(values_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=float)
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_ms": round(float(arr.mean()), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (macOS reports bytes, Linux KiB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_queries(path: Path) -> tuple[dict, list[tuple[dict, ChunkFilter | None]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = [q["id"] for q in data["queries"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate query IDs in {path}")
    queries = []
    for item in data["queries"]:
        filters = ChunkFilter(**item.get("filters", {}))
        queries.append((item, None if filters.is_empty() else filters))
    return data, queries


def make_embedder(name: str, device: str | None):
    from evidenceroute.retrieval.dense import HashingEmbedder, SentenceTransformerEmbedder

    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name, device=device)


def embedder_device(embedder: Any) -> str:
    model = getattr(embedder, "_model", None)
    return str(getattr(model, "device", "n/a"))


def benchmark_mode(retriever, queries, *, top_k: int, repeats: int, warmup_rounds: int,
                   embedder: TimedEmbedder) -> dict:
    for _ in range(warmup_rounds):
        for item, filters in queries:
            retriever.search(item["query"], top_k, filters)
    embedder.query_seconds.clear()

    all_ms: list[float] = []
    per_query = {}
    consistent = True
    filters_respected = True
    for item, filters in queries:
        query_ms = []
        first_ids = None
        first_results = None
        for _ in range(repeats):
            start = time.perf_counter()
            results = retriever.search(item["query"], top_k, filters)
            query_ms.append((time.perf_counter() - start) * 1000)
            ids = [r.chunk.chunk_id for r in results]
            if first_ids is None:
                first_ids, first_results = ids, results
            elif ids != first_ids:
                consistent = False
        if filters is not None and not all(filters.matches(r.chunk) for r in first_results):
            filters_respected = False
        all_ms.extend(query_ms)
        per_query[item["id"]] = {
            "p50_ms": round(float(np.percentile(query_ms, 50)), 3),
            "result_count": len(first_results),
            "top_results": [
                {"chunk_id": r.chunk.chunk_id, "ticker": r.chunk.ticker,
                 "fiscal_year": r.chunk.fiscal_year, "section_id": r.chunk.section_id,
                 "score": round(r.score, 6)}
                for r in first_results[:3]
            ],
        }
    summary = summarize(all_ms)
    if embedder.query_seconds:
        summary["query_embedding"] = summarize([s * 1000 for s in embedder.query_seconds])
    return {"latency": summary, "ranking_consistent": consistent,
            "filters_respected": filters_respected, "queries": per_query}


def main(argv: list[str] | None = None) -> int:
    from evidenceroute.retrieval.dense import DEFAULT_MODEL, DenseRetriever

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--embedder", default=DEFAULT_MODEL,
                        help=f"SentenceTransformer model (default {DEFAULT_MODEL}) or 'hashing'")
    parser.add_argument("--device", default=None, help="e.g. mps, cuda, cpu (default: auto)")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the vector index so indexing stages are measured")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup_rounds < 0 or args.top_k < 1:
        parser.error("--repeats and --top-k must be >= 1 and --warmup-rounds >= 0")

    query_set, queries = load_queries(args.queries)
    indexing: dict[str, Any] = {}

    corpus, indexing["corpus_load_s"] = timed(load_chunks, args.chunks)
    fingerprint, indexing["corpus_fingerprint_s"] = timed(corpus.fingerprint)
    bm25, indexing["bm25_build_s"] = timed(BM25Retriever, corpus)

    inner, indexing["model_load_s"] = timed(make_embedder, args.embedder, args.device)
    embedder = TimedEmbedder(inner)
    slug = embedder.model_id.replace("/", "_").replace(":", "_")[:80]
    index_path = args.index_dir / slug
    dense = DenseRetriever(corpus, embedder, index_path=index_path)
    if args.rebuild:
        _, total = timed(dense.build)
        action = "rebuilt"
    else:
        rebuilt, total = timed(dense.ensure_index)
        action = "rebuilt" if rebuilt else "reused"
    indexing.update({
        "index_action": action,
        "dense_index_total_s": total,
        "document_embedding_s": embedder.document_seconds,
        "qdrant_write_and_overhead_s": total - embedder.document_seconds,
        "index_size_mb": round(dir_bytes(index_path) / (1024 * 1024), 1),
        "peak_rss_mb_after_indexing": peak_rss_mb(),
    })
    indexing = {k: round(v, 3) if isinstance(v, float) else v for k, v in indexing.items()}

    retrievers = {
        "bm25": bm25,
        "dense": dense,
        "hybrid": HybridRetriever({"bm25": bm25, "dense": dense}),
    }
    modes = {}
    for name, retriever in retrievers.items():
        print(f"[benchmark] {name}: {len(queries)} queries x {args.repeats} repeats", file=sys.stderr)
        modes[name] = benchmark_mode(retriever, queries, top_k=args.top_k, repeats=args.repeats,
                                     warmup_rounds=args.warmup_rounds, embedder=embedder)

    now = datetime.now(timezone.utc)
    report = {
        "created_at_utc": now.isoformat(),
        "note": "Engineering measurements only; the query set has no gold evidence.",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "embedding_device": embedder_device(inner),
            "packages": package_versions(),
        },
        "config": {
            "chunks": str(args.chunks),
            "query_set": query_set.get("name"),
            "query_count": len(queries),
            "embedding_model": embedder.model_id,
            "embedding_dimension": embedder.dimension,
            "index_path": str(index_path),
            "top_k": args.top_k,
            "repeats": args.repeats,
            "warmup_rounds": args.warmup_rounds,
            "hybrid_rrf_k": retrievers["hybrid"].config.rrf_k,
            "hybrid_candidate_k": retrievers["hybrid"].config.candidate_k,
        },
        "corpus": {"chunk_count": len(corpus), "fingerprint": fingerprint},
        "indexing": indexing,
        "modes": modes,
        "peak_rss_mb": peak_rss_mb(),
    }

    output = args.output or DEFAULT_OUTPUT_DIR / f"retrieval-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nIndexing")
    for key, value in indexing.items():
        print(f"  {key:32} {value}")
    print(f"\nQuery latency (ms, {len(queries)} queries x {args.repeats} repeats, top_k={args.top_k})")
    print(f"  {'mode':8} {'p50':>9} {'p95':>9} {'mean':>9}  consistent  filters_ok")
    for name, result in modes.items():
        lat = result["latency"]
        print(f"  {name:8} {lat['p50_ms']:9.2f} {lat['p95_ms']:9.2f} {lat['mean_ms']:9.2f}  "
              f"{str(result['ranking_consistent']):10}  {result['filters_respected']}")
    print(f"\nSaved {output}")
    dense.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
