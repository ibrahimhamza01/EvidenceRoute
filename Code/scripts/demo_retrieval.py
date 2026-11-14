"""Search verified SEC passages with BM25, dense (SentenceTransformer + Qdrant) or hybrid RRF.

Examples (from the repository root):

    PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "H100 export license" --mode bm25
    PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py \
        "reliance on outside chip manufacturers" --ticker AMD --year 2024
    PYTHONPATH=Code/src python Code/scripts/demo_retrieval.py "data center revenue growth" \
        --section item_7 --top-k 3 --show-ranks

The default chunk source is the committed development fixture. ``--chunks`` accepts a
JSONL or the pipeline's ``chunks.parquet``. The Qdrant index is persisted under
``--index-dir`` (default ``data/indexes/dev``, ignored by Git) and is rebuilt
automatically whenever the chunks or embedding model change.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from evidenceroute.retrieval import BM25Retriever, ChunkFilter, HybridRetriever, load_chunks

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = ROOT / "Code/tests/fixtures/retrieval_dev_chunks.jsonl"
DEFAULT_INDEX_DIR = ROOT / "data/indexes/dev"


def make_embedder(name: str):
    from evidenceroute.retrieval.dense import HashingEmbedder, SentenceTransformerEmbedder

    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)


def main(argv: list[str] | None = None) -> int:
    from evidenceroute.retrieval.dense import DEFAULT_MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ticker", action="append", help="Repeatable, e.g. --ticker NVDA --ticker AMD")
    parser.add_argument("--year", type=int, action="append", help="Fiscal year; repeatable")
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--section", action="append", help="Section ID, e.g. item_1a; repeatable")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--embedder", default=DEFAULT_MODEL,
                        help=f"SentenceTransformer model name (default {DEFAULT_MODEL}), "
                             "or 'hashing' for an offline, non-semantic smoke test")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Force a vector index rebuild")
    parser.add_argument("--show-ranks", action="store_true", help="Show per-retriever ranks (hybrid)")
    parser.add_argument("--json", action="store_true", help="Print RetrievalResult JSON")
    args = parser.parse_args(argv)

    corpus = load_chunks(args.chunks)
    filters = ChunkFilter(tickers=args.ticker, fiscal_years=args.year,
                          document_ids=args.document_id, section_ids=args.section)
    filters = None if filters.is_empty() else filters

    retrievers = {}
    if args.mode in {"bm25", "hybrid"}:
        retrievers["bm25"] = BM25Retriever(corpus)
    if args.mode in {"dense", "hybrid"}:
        from evidenceroute.retrieval.dense import DenseRetriever

        embedder = make_embedder(args.embedder)
        slug = embedder.model_id.replace("/", "_").replace(":", "_")[:80]
        dense = DenseRetriever(corpus, embedder, index_path=args.index_dir / slug)
        if args.rebuild:
            dense.build()
            status = "rebuilt"
        else:
            status = "built" if dense.ensure_index() else "reused (unchanged chunks and model)"
        print(f"[dense] {embedder.model_id} index {status} at {args.index_dir / slug}", file=sys.stderr)
        retrievers["dense"] = dense

    print(f"[corpus] {len(corpus)} chunks from {args.chunks} | mode={args.mode} | "
          f"filters={filters.model_dump(exclude_none=True) if filters else {}}\n", file=sys.stderr)

    if args.mode == "hybrid":
        hits = HybridRetriever(retrievers).search_with_details(args.query, args.top_k, filters)
        rows = [(h.to_result(), dict(h.ranks)) for h in hits]
    else:
        rows = [(r, {}) for r in retrievers[args.mode].search(args.query, args.top_k, filters)]

    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r, _ in rows], indent=2, ensure_ascii=False))
        return 0
    if not rows:
        print("No results.")
    for position, (result, ranks) in enumerate(rows, start=1):
        c = result.chunk
        extra = f" ranks={ranks}" if args.show_ranks and ranks else ""
        print(f"{position}. score={result.score:.5f}{extra}")
        print(f"   {c.ticker} FY{c.fiscal_year} | {c.section_id} {c.section_name} | {c.chunk_id}")
        print(f"   {c.source_url}")
        print(textwrap.indent(textwrap.shorten(c.text, 300, placeholder=" ..."), "   "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
