#!/usr/bin/env python3
"""Train and evaluate the baseline scope routers on the pilot question set.

Run from the repository root:
    PYTHONPATH=Code/src python Code/scripts/train_routers.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from evidenceroute.routing.baselines import SEED, BaselineRouter, load_split

LABELS = ["simple", "multi_source", "corpus_wide"]


def evaluate(router: BaselineRouter, questions: list[str], gold: list[str]) -> dict:
    """Score one trained router on the eval questions."""
    predicted = [r.scope for r in router.predict_many(questions)]

    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        # Rows are the true label, columns the predicted label, both in LABELS order.
        "confusion_matrix": confusion_matrix(gold, predicted, labels=LABELS).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline routers.")
    parser.add_argument("--questions", type=Path, default=Path("data/questions/questions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("Code/results/routing_pilot"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_q, train_y = load_split(args.questions, "train")
    eval_q, eval_y = load_split(args.questions, "eval")

    report = {
        "note": "Pilot results on 4 filings (NVIDIA and AMD, FY2023-FY2024). Not final conclusions.",
        "seed": args.seed,
        "n_train": len(train_q),
        "n_eval": len(eval_q),
        "labels": LABELS,
        "models": {},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("logreg", "naive_bayes"):
        router = BaselineRouter(kind, seed=args.seed).fit(train_q, train_y)
        result = evaluate(router, eval_q, eval_y)
        result["config"] = {name: str(value) for name, value in router.pipeline.get_params(deep=False)["steps"]}
        report["models"][kind] = result

        # Confusion matrix as a small CSV for the report.
        with (args.out_dir / f"confusion_{kind}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["true \\ predicted"] + LABELS)
            for label, counts in zip(LABELS, result["confusion_matrix"]):
                writer.writerow([label] + counts)

        print(f"{kind:12} accuracy={result['accuracy']:.3f}  macro_f1={result['macro_f1']:.3f}")

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Saved results to {args.out_dir}")


if __name__ == "__main__":
    main()
