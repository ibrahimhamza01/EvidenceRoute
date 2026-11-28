"""Evaluate SEC retrieval using the shared reviewed benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evidenceroute.evaluation.benchmark import read_csv, gold_groups
from evidenceroute.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    full_coverage_at_k,
)
from evidenceroute.retrieval import load_chunks
from evidenceroute.research.retriever import (
    build_research_retriever,
    RETRIEVAL_METHODS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data/processed/full-v1/chunks.parquet"
DEFAULT_QUESTIONS = ROOT / "data/benchmark/questions.csv"
DEFAULT_GOLD = ROOT / "data/benchmark/gold_evidence.csv"


def load_questions(questions_path, gold_path, split="heldout"):
    questions = read_csv(Path(questions_path))
    gold = read_csv(Path(gold_path))

    reviewed_gold = [
        row for row in gold
        if row["mapping_status"] == "mapped"
        and row["quote_verified"] == "true"
    ]

    selected = []

    for row in questions:
        if split != "all" and row["eval_split"] != split:
            continue

        if row["answerable"] != "true":
            continue

        if row["review_status"] != "approved":
            continue

        if row["quote_verified"] != "true":
            continue

        if row["gold_evidence_status"] != "complete":
            continue

        groups = gold_groups(
            reviewed_gold, row["question_id"]
        )

        if not groups:
            raise ValueError(
                f"No reviewed gold evidence for {row['question_id']}"
            )

        selected.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "scope": row["scope"],
            "gold_groups": groups,
        })

    if not selected:
        raise ValueError("No eligible reviewed questions found")

    return selected


def evaluate(retriever, questions, k=5):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    if not questions:
        raise ValueError("No questions to evaluate")

    rows = []

    for item in questions:
        start = time.perf_counter()

        results = retriever.search(
            item["question"], top_k=k
        )

        latency = time.perf_counter() - start

        ranked_ids = list(dict.fromkeys(
            result.chunk.chunk_id for result in results
        ))

        groups = item["gold_groups"]

        rows.append({
            "question_id": item["question_id"],
            "scope": item["scope"],
            "recall_at_k": recall_at_k(groups, ranked_ids, k),
            "precision_at_k": precision_at_k(groups, ranked_ids, k),
            "reciprocal_rank_at_k": reciprocal_rank(
                groups, ranked_ids, k
            ),
            "full_coverage_at_k": full_coverage_at_k(
                groups, ranked_ids, k
            ),
            "latency_ms": latency * 1000,
            "retrieved_chunk_ids": ranked_ids,
        })

    n = len(rows)

    return {
        "questions_evaluated": n,
        "k": k,
        "mean_recall_at_k": sum(
            r["recall_at_k"] for r in rows
        ) / n,
        "mean_precision_at_k": sum(
            r["precision_at_k"] for r in rows
        ) / n,
        "mean_mrr_at_k": sum(
            r["reciprocal_rank_at_k"] for r in rows
        ) / n,
        "full_coverage_rate": sum(
            r["full_coverage_at_k"] for r in rows
        ) / n,
        "mean_latency_ms": sum(
            r["latency_ms"] for r in rows
        ) / n,
        "results": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--questions", type=Path, default=DEFAULT_QUESTIONS
    )
    parser.add_argument(
        "--gold", type=Path, default=DEFAULT_GOLD
    )
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CORPUS
    )
    parser.add_argument(
        "--split", choices=["dev", "heldout", "all"],
        default="heldout"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval", choices=RETRIEVAL_METHODS,
        default="bm25",
    )
    parser.add_argument(
        "--embedder", default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--index-dir", type=Path,
        default=ROOT / "data/indexes/full-v1",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/rag_benchmark_results.json")
    )

    args = parser.parse_args(argv)

    questions = load_questions(
        args.questions, args.gold, args.split
    )

    corpus = load_chunks(args.chunks)
    corpus_ids = {chunk.chunk_id for chunk in corpus}

    missing = {
        chunk_id
        for question in questions
        for group in question["gold_groups"].values()
        for chunk_id in group
        if chunk_id not in corpus_ids
    }

    if missing:
        raise ValueError(
            f"{len(missing)} gold chunk IDs missing from corpus: "
            f"{sorted(missing)[:5]}"
        )

    retriever = build_research_retriever(
        corpus,
        args.retrieval,
        embedder_name=args.embedder,
        index_dir=args.index_dir,
        rebuild=args.rebuild_index,
    )

    report = evaluate(
        retriever, questions, args.top_k
    )

    report["corpus_chunks"] = len(corpus)
    report["retriever"] = args.retrieval
    report["embedder"] = (
        args.embedder if args.retrieval != "bm25" else None
    )
    report["split"] = args.split

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(json.dumps({
        key: value
        for key, value in report.items()
        if key != "results"
    }, indent=2))

    print("Full results:", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
