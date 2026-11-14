#!/usr/bin/env python3
"""Validate the evaluation benchmark and print review coverage.

Run from the repository root:
    PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py
    PYTHONPATH=Code/src python Code/scripts/validate_benchmark.py --processed-dir data/processed/pilot-v1

Exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from evidenceroute.evaluation.benchmark import read_csv, review_coverage, validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark.csv and gold_evidence.csv.")
    parser.add_argument("--benchmark", type=Path, default=Path("data/eval/benchmark.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/filings_manifest.csv"))
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Optional: also check that gold chunk IDs exist in chunks.parquet")
    args = parser.parse_args()

    benchmark = read_csv(args.benchmark)
    gold = read_csv(args.gold)
    manifest = read_csv(args.manifest)

    chunk_ids = None
    if args.processed_dir is not None:
        import pyarrow.parquet as pq  # only needed for this optional check
        table = pq.read_table(args.processed_dir / "chunks.parquet", columns=["chunk_id"])
        chunk_ids = set(table.column("chunk_id").to_pylist())

    errors = validate(benchmark, gold, manifest, chunk_ids)

    # Summary
    print(f"Questions: {len(benchmark)}   Evidence rows: {len(gold)}")
    splits = Counter((row["eval_split"], row["scope"]) for row in benchmark)
    for split in ("dev", "heldout"):
        counts = ", ".join(f"{scope}={splits[(split, scope)]}" for scope in ("simple", "multi_source", "corpus_wide"))
        print(f"  {split:8} {counts}")
    print("Gold evidence status:", dict(Counter(row["gold_evidence_status"] for row in benchmark)))
    print("Answerable:", dict(Counter(row["answerable"] for row in benchmark)))
    print("Review coverage (pilot questions with quotes checked by a person):")
    for split, (verified, total) in review_coverage(benchmark).items():
        print(f"  {split:8} {verified}/{total}")
    print("Chunk IDs checked against processed data:", "yes" if chunk_ids is not None else "no")

    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)")
        for error in errors:
            print("  -", error)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
