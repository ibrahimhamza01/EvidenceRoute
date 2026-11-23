"""Evaluate BM25 retrieval against explicitly reviewed SEC chunk labels."""

import argparse
import csv
import json
import time
from pathlib import Path

from evidenceroute.evaluation.rag_quality import (
    recall_at_k,
    reciprocal_rank_at_k,
)
from evidenceroute.retrieval import BM25Retriever, load_chunks


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data/processed/full-v1/chunks.parquet"


def load_questions(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"question_id", "question", "relevant_chunk_ids"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required columns: {sorted(required)}")

        questions = []

        for row in reader:
            ids = {
                item.strip()
                for item in row["relevant_chunk_ids"].split(";")
                if item.strip()
            }

            if not row["question_id"].strip() or not row["question"].strip() or not ids:
                raise ValueError("Every benchmark row needs an ID, question, and gold chunks")

            questions.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "relevant_ids": ids,
            })

    if not questions:
        raise ValueError("Benchmark must contain at least one question")

    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")

    return questions


def evaluate(retriever, questions, k=5):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    rows = []

    for item in questions:
        start = time.perf_counter()
        results = retriever.search(item["question"], top_k=k)
        latency = time.perf_counter() - start

        rows.append({
            "question_id": item["question_id"],
            "recall_at_k": recall_at_k(results, item["relevant_ids"], k),
            "mrr_at_k": reciprocal_rank_at_k(results, item["relevant_ids"], k),
            "latency_ms": latency * 1000,
        })

    n = len(rows)

    return {
        "questions_evaluated": n,
        "k": k,
        "mean_recall_at_k": sum(r["recall_at_k"] for r in rows) / n,
        "mean_mrr_at_k": sum(r["mrr_at_k"] for r in rows) / n,
        "mean_latency_ms": sum(r["latency_ms"] for r in rows) / n,
        "results": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("rag_benchmark_results.json"))
    args = parser.parse_args(argv)

    questions = load_questions(args.questions)
    corpus = load_chunks(args.chunks)

    corpus_ids = {chunk.chunk_id for chunk in corpus}
    missing = {
        chunk_id
        for question in questions
        for chunk_id in question["relevant_ids"]
        if chunk_id not in corpus_ids
    }

    if missing:
        raise ValueError(
            f"{len(missing)} gold chunk IDs are missing from the corpus: "
            f"{sorted(missing)[:5]}"
        )

    report = evaluate(BM25Retriever(corpus), questions, args.top_k)
    report["corpus_chunks"] = len(corpus)
    report["retriever"] = "bm25"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
