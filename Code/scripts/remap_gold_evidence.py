#!/usr/bin/env python3
"""Recompute section, chunk IDs and offsets in gold_evidence.csv from the stored quotes.

The quotes are the source of truth. Run this after reprocessing the filings
(for example with the full corpus or new chunk settings), then rerun
validate_benchmark.py.

Run from the repository root:
    PYTHONPATH=Code/src python Code/scripts/remap_gold_evidence.py --processed-dir data/processed/pilot-v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq

from evidenceroute.evaluation.benchmark import read_csv
from evidenceroute.evaluation.gold import containing_chunks, locate_quote


def main() -> int:
    parser = argparse.ArgumentParser(description="Remap gold evidence quotes to processed chunks.")
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.csv"))
    parser.add_argument("--processed-dir", type=Path, required=True)
    args = parser.parse_args()

    sections = pq.read_table(args.processed_dir / "sections.parquet").to_pylist()
    chunks = pq.read_table(args.processed_dir / "chunks.parquet").to_pylist()
    section_text = {(s["document_id"], s["section_id"]): s["text"] for s in sections}

    rows = read_csv(args.gold)
    changed = 0
    missing = []
    for row in rows:
        # Only rows that have, or should have, a location in the processed data.
        if row["mapping_status"] not in {"mapped", "section_not_processed"}:
            continue
        document_id = row["document_id"]
        # Try the recorded section first, then the other sections of the same filing.
        candidates = sorted(
            (key for key in section_text if key[0] == document_id),
            key=lambda key: key[1] != row["section_id"],
        )
        location = None
        for key in candidates:
            span = locate_quote(section_text[key], row["quote"])
            if span is not None:
                location = (key[1], *span)
                break

        if location is None:
            new = {"mapping_status": "section_not_processed", "chunk_ids": "", "char_start": "", "char_end": ""}
            if row["mapping_status"] == "mapped":
                missing.append(row["question_id"])
        else:
            section_id, start, end = location
            ids = containing_chunks(chunks, document_id, section_id, start, end)
            new = {"mapping_status": "mapped", "section_id": section_id, "chunk_ids": ";".join(ids),
                   "char_start": str(start), "char_end": str(end)}
        if any(row[field] != value for field, value in new.items()):
            changed += 1
            row.update(new)

    with args.gold.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows updated: {changed}")
    if missing:
        print("Previously mapped quotes no longer found:", ", ".join(sorted(set(missing))))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
