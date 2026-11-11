#!/usr/bin/env python3
"""Assign each routing question to the train or eval split.

Run from the repository root:
    python Code/scripts/make_splits.py

Questions that share a template_id always go to the same split, so the
classifier is never evaluated on a rewording of a question it trained on.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SCOPES = ("simple", "multi_source", "corpus_wide")


def assign_splits(rows: list[dict], eval_fraction: float, seed: int) -> list[dict]:
    """Fill the "split" field of every row and return the rows."""
    rng = random.Random(seed)  # fixed seed, so the split is the same every run

    for scope in SCOPES:
        # Group this scope's questions by template.
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["scope"] == scope:
                groups[row["template_id"]].append(row)

        # Shuffle templates (sorted first so the order only depends on the seed).
        templates = sorted(groups)
        rng.shuffle(templates)

        # Move whole templates into eval until this scope reaches its eval target.
        total = sum(len(g) for g in groups.values())
        target = max(1, round(total * eval_fraction))
        in_eval = 0
        for template in templates:
            split = "eval" if in_eval < target else "train"
            for row in groups[template]:
                row["split"] = split
            if split == "eval":
                in_eval += len(groups[template])

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign train/eval splits by template_id.")
    parser.add_argument("--questions", type=Path, default=Path("data/questions/questions.csv"))
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    assign_splits(rows, args.eval_fraction, args.seed)

    with args.questions.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Print a summary so the split can be checked at a glance.
    for scope in SCOPES:
        counts = {s: sum(1 for r in rows if r["scope"] == scope and r["split"] == s) for s in ("train", "eval")}
        print(f"{scope:13} train={counts['train']:3}  eval={counts['eval']:3}")


if __name__ == "__main__":
    main()
