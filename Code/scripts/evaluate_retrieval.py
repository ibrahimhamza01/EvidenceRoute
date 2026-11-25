"""Score retrieval modes against gold evidence: Recall@K, MRR@10 and full coverage.

Each answerable question with supporting gold evidence is searched with every mode
(BM25, dense, hybrid and hybrid + cross-encoder reranking) under two conditions:

* ``none``: the question text only.
* ``metadata``: the question text plus a filter on the question's annotated tickers
  and fiscal years, as a planner that already knows the target filings would use.

Recall@K is the share of a question's evidence groups with at least one supporting
chunk in the top K; MRR@10 uses the first supporting chunk. Results are reported
overall and per scope, with a per-question comparison of each reranked mode against
hybrid. Corpus-wide questions can have more evidence groups than K, so their recall
is bounded below 1 by design.

The benchmark and gold-evidence CSVs are passed explicitly; this script does not
depend on where they are stored. Results are written as JSON under
``data/benchmarks/`` (ignored by Git).

Example (from the repository root):

    PYTHONPATH=Code/src python Code/scripts/evaluate_retrieval.py \\
        --benchmark data/benchmark/questions.csv --gold data/benchmark/gold_evidence.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evidenceroute.evaluation.benchmark import gold_groups, read_csv
from evidenceroute.evaluation.metrics import full_coverage_at_k, recall_at_k, reciprocal_rank
from evidenceroute.retrieval import (
    BM25Retriever,
    ChunkFilter,
    HybridRetriever,
    RerankConfig,
    RerankingRetriever,
    load_chunks,
)
from evidenceroute.retrieval.rerank import DEFAULT_RERANKER, SentenceTransformerCrossEncoder

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = ROOT / "data/processed/full-v1/chunks.parquet"
DEFAULT_INDEX_DIR = ROOT / "data/indexes/full-v1"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmarks"
SCOPES = ("simple", "multi_source", "corpus_wide")
MRR_K = 10


def split_field(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def metadata_filter(question: dict) -> ChunkFilter | None:
    tickers = split_field(question.get("tickers", ""))
    years = [int(y) for y in split_field(question.get("fiscal_years", ""))]
    filters = ChunkFilter(tickers=tickers or None, fiscal_years=years or None)
    return None if filters.is_empty() else filters


def load_questions(benchmark: Path, gold: Path, split: str, corpus) -> tuple[list[dict], dict]:
    questions = read_csv(benchmark)
    gold_rows = read_csv(gold)
    selected, skipped = [], {"unanswerable": 0, "no_supporting_gold": 0, "other_split": 0}
    for q in questions:
        if split != "all" and q.get("eval_split") != split:
            skipped["other_split"] += 1
            continue
        if q.get("answerable") != "true":
            skipped["unanswerable"] += 1
            continue
        groups = gold_groups(gold_rows, q["question_id"])
        if not groups:
            skipped["no_supporting_gold"] += 1
            continue
        missing = sorted(c for ids in groups.values() for c in ids if c not in corpus)
        if missing:
            raise ValueError(f"{q['question_id']}: gold chunk IDs not in the corpus: {missing[:3]}")
        selected.append({**q, "gold_groups": groups})
    return selected, skipped


def score(groups: dict[str, set[str]], ranked: list[str], ks: list[int]) -> dict:
    result = {f"recall@{k}": recall_at_k(groups, ranked, k) for k in ks}
    result[f"mrr@{MRR_K}"] = reciprocal_rank(groups, ranked, MRR_K)
    result[f"full_coverage@{MRR_K}"] = float(full_coverage_at_k(groups, ranked, MRR_K))
    return result


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if k != "question_id"]
    return {"n": len(rows), **{k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}}


def paired(base: list[dict], other: list[dict], metric: str) -> dict:
    """Per-question wins/ties/losses of ``other`` against ``base`` on ``metric``."""
    diffs = [o[metric] - b[metric] for b, o in zip(base, other)]
    return {"metric": metric, "wins": sum(d > 1e-12 for d in diffs),
            "ties": sum(abs(d) <= 1e-12 for d in diffs),
            "losses": sum(d < -1e-12 for d in diffs),
            "mean_delta": round(float(np.mean(diffs)), 4)}


def main(argv: list[str] | None = None) -> int:
    from evidenceroute.retrieval.dense import DEFAULT_MODEL, DenseRetriever, HashingEmbedder, \
        SentenceTransformerEmbedder

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, required=True, help="questions CSV")
    parser.add_argument("--gold", type=Path, required=True, help="gold evidence CSV")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--embedder", default=DEFAULT_MODEL,
                        help=f"SentenceTransformer model (default {DEFAULT_MODEL}) or 'hashing'")
    parser.add_argument("--device", default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--rerankers", nargs="*", default=[DEFAULT_RERANKER],
                        help="Cross-encoders to evaluate; pass with no values to skip")
    parser.add_argument("--rerank-candidates", type=int, default=RerankConfig().candidate_k)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    ks = sorted(set(args.ks) | {MRR_K})
    depth = max(ks)
    if args.rerank_candidates < depth:
        parser.error(f"--rerank-candidates must be >= the largest K ({depth})")

    corpus = load_chunks(args.chunks)
    questions, skipped = load_questions(args.benchmark, args.gold, args.split, corpus)
    if not questions:
        parser.error("No answerable questions with supporting gold evidence were selected")

    bm25 = BM25Retriever(corpus)
    embedder = HashingEmbedder() if args.embedder == "hashing" else \
        SentenceTransformerEmbedder(args.embedder, device=args.device)
    slug = embedder.model_id.replace("/", "_").replace(":", "_")[:80]
    dense = DenseRetriever(corpus, embedder, index_path=args.index_dir / slug)
    dense.ensure_index()
    hybrid = HybridRetriever({"bm25": bm25, "dense": dense})
    modes = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
    for model_name in args.rerankers:
        scorer = SentenceTransformerCrossEncoder(model_name, device=args.device)
        name = f"rerank[{model_name.split('/')[-1]}]@{args.rerank_candidates}"
        modes[name] = RerankingRetriever(hybrid, scorer, RerankConfig(
            candidate_k=args.rerank_candidates, model_name=model_name))

    results: dict[str, dict] = {}
    for condition in ("none", "metadata"):
        for name, retriever in modes.items():
            print(f"[evaluate] {condition:8} {name}", file=sys.stderr)
            rows = []
            for q in questions:
                filters = metadata_filter(q) if condition == "metadata" else None
                ranked = [r.chunk.chunk_id for r in retriever.search(q["question"], depth, filters)]
                rows.append({"question_id": q["question_id"], **score(q["gold_groups"], ranked, ks),
                             "_ranked": ranked})
            results.setdefault(condition, {})[name] = rows

    report_modes = {}
    for condition, by_mode in results.items():
        for name, rows in by_mode.items():
            clean = [{k: v for k, v in r.items() if k != "_ranked"} for r in rows]
            by_scope = {s: aggregate([c for c, q in zip(clean, questions) if q["scope"] == s])
                        for s in SCOPES}
            entry = {"overall": aggregate(clean), "by_scope": by_scope}
            if name.startswith("rerank["):
                base = [{k: v for k, v in r.items() if k != "_ranked"}
                        for r in by_mode["hybrid"]]
                entry["vs_hybrid"] = [paired(base, clean, f"recall@{MRR_K}"),
                                      paired(base, clean, f"mrr@{MRR_K}")]
            entry["questions"] = rows
            report_modes.setdefault(condition, {})[name] = entry

    now = datetime.now(timezone.utc)
    report = {
        "created_at_utc": now.isoformat(),
        "benchmark": str(args.benchmark), "gold": str(args.gold), "split": args.split,
        "questions_scored": len(questions), "questions_skipped": skipped,
        "scope_counts": {s: sum(q["scope"] == s for q in questions) for s in SCOPES},
        "embedding_model": embedder.model_id,
        "rerankers": args.rerankers, "rerank_candidates": args.rerank_candidates, "ks": ks,
        "conditions": report_modes,
    }
    output = args.output or DEFAULT_OUTPUT_DIR / f"retrieval-eval-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nScored {len(questions)} questions {report['scope_counts']}; skipped {skipped}")
    cols = [f"recall@{k}" for k in ks] + [f"mrr@{MRR_K}", f"full_coverage@{MRR_K}"]
    for condition, by_mode in report_modes.items():
        print(f"\nFilter condition: {condition}")
        print(f"  {'mode':36}" + "".join(f"{c:>17}" for c in cols))
        for name, entry in by_mode.items():
            print(f"  {name:36}" + "".join(f"{entry['overall'][c]:17.3f}" for c in cols))
        print(f"  by scope (recall@{MRR_K} / mrr@{MRR_K}):")
        for name, entry in by_mode.items():
            parts = [f"{s} {entry['by_scope'][s][f'recall@{MRR_K}']:.3f}/"
                     f"{entry['by_scope'][s][f'mrr@{MRR_K}']:.3f}"
                     for s in SCOPES if entry["by_scope"][s]["n"]]
            print(f"    {name:34} " + "  ".join(parts))
        for name, entry in by_mode.items():
            for cmp in entry.get("vs_hybrid", []):
                print(f"  {name} vs hybrid on {cmp['metric']}: {cmp['wins']} better, "
                      f"{cmp['ties']} same, {cmp['losses']} worse (mean delta {cmp['mean_delta']:+.4f})")
    print(f"\nSaved {output}")
    dense.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
