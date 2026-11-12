"""Load and validate the evaluation benchmark (benchmark.csv and gold_evidence.csv)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

SCOPES = {"simple", "multi_source", "corpus_wide"}
ANSWERABLE = {"true", "false", "pending"}
CORPORA = {"pilot", "expanded"}
SPLITS = {"dev", "heldout"}
EVIDENCE_STATUSES = {
    "complete", "incomplete_section_not_processed", "none", "related_only", "pending_expanded_corpus",
}
RELEVANCE = {"supporting", "related"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gold_groups(gold_rows: list[dict], question_id: str) -> dict[str, set[str]]:
    """Supporting chunk IDs for one question, grouped by evidence group (input for the metrics)."""
    groups: dict[str, set[str]] = defaultdict(set)
    for row in gold_rows:
        if row["question_id"] == question_id and row["relevance"] == "supporting" and row["chunk_ids"]:
            groups[row["evidence_group"]].update(row["chunk_ids"].split(";"))
    return dict(groups)


def validate(benchmark: list[dict], gold: list[dict], manifest: list[dict],
             chunk_ids: set[str] | None = None) -> list[str]:
    """Return a list of problems. An empty list means the benchmark is valid.

    chunk_ids is optional: pass the IDs from chunks.parquet to also check that
    every gold chunk exists in the processed data.
    """
    errors: list[str] = []
    filings = {row["document_id"]: row for row in manifest}
    by_id = {}

    # Question-level checks
    for row in benchmark:
        qid = row["question_id"]
        if qid in by_id:
            errors.append(f"{qid}: duplicate question_id")
        by_id[qid] = row
        if row["scope"] not in SCOPES:
            errors.append(f"{qid}: invalid scope {row['scope']!r}")
        if row["answerable"] not in ANSWERABLE:
            errors.append(f"{qid}: invalid answerable {row['answerable']!r}")
        if row["corpus"] not in CORPORA:
            errors.append(f"{qid}: invalid corpus {row['corpus']!r}")
        if row["eval_split"] not in SPLITS:
            errors.append(f"{qid}: invalid eval_split {row['eval_split']!r}")
        if row["gold_evidence_status"] not in EVIDENCE_STATUSES:
            errors.append(f"{qid}: invalid gold_evidence_status {row['gold_evidence_status']!r}")
        if not row["expected_answer"].strip():
            errors.append(f"{qid}: missing expected_answer")
        if (row["corpus"] == "expanded") != (row["answerable"] == "pending"):
            errors.append(f"{qid}: 'pending' answerability must be used for expanded-corpus questions only")
        if row["router_split"] == "eval" and row["eval_split"] != "heldout":
            errors.append(f"{qid}: router eval questions must be heldout")
        for doc in filter(None, row["document_ids"].split(";")):
            if doc not in filings:
                errors.append(f"{qid}: document {doc} not in filings manifest")

    # Template leakage: a template must not appear in both splits
    splits_per_template: dict[str, set[str]] = defaultdict(set)
    for row in benchmark:
        splits_per_template[row["template_id"]].add(row["eval_split"])
    for template, splits in splits_per_template.items():
        if len(splits) > 1:
            errors.append(f"template {template} appears in both dev and heldout")

    # Evidence-level checks
    groups_per_question: dict[str, set[str]] = defaultdict(set)
    for row in gold:
        qid = row["question_id"]
        if qid not in by_id:
            errors.append(f"gold evidence for unknown question {qid}")
            continue
        if row["mapping_status"] != "mapped":
            continue
        groups_per_question[qid].add(row["evidence_group"])
        if row["relevance"] not in RELEVANCE:
            errors.append(f"{qid}: invalid relevance {row['relevance']!r}")
        if not row["chunk_ids"] or not row["section_id"]:
            errors.append(f"{qid}: mapped evidence without section or chunk IDs")
        filing = filings.get(row["document_id"])
        if filing is None:
            errors.append(f"{qid}: evidence document {row['document_id']} not in manifest")
        elif row["source_url"] != filing["source_url"]:
            errors.append(f"{qid}: evidence source_url does not match the manifest")
        if chunk_ids is not None:
            for chunk_id in row["chunk_ids"].split(";"):
                if chunk_id not in chunk_ids:
                    errors.append(f"{qid}: chunk {chunk_id} not found in processed data")

    # Status must agree with the evidence actually present
    for qid, row in by_id.items():
        status = row["gold_evidence_status"]
        supporting = gold_groups(gold, qid)
        if status == "complete":
            if not supporting:
                errors.append(f"{qid}: status complete but no supporting evidence")
            if row["answerable"] != "true":
                errors.append(f"{qid}: status complete but question is not answerable")
            if row["scope"] == "multi_source" and len(supporting) < 2:
                errors.append(f"{qid}: multi_source question needs at least 2 evidence groups")
        if status in {"none", "related_only"} and row["answerable"] != "false":
            errors.append(f"{qid}: status {status} requires answerable = false")
        if status == "related_only" and supporting:
            errors.append(f"{qid}: related_only question must not have supporting evidence")

    return errors


def review_coverage(benchmark: list[dict]) -> dict[str, tuple[int, int]]:
    """(verified, total) per split, counting only pilot questions with evidence to review."""
    coverage: dict[str, list[int]] = {split: [0, 0] for split in sorted(SPLITS)}
    for row in benchmark:
        if row["corpus"] != "pilot":
            continue
        counts = coverage[row["eval_split"]]
        counts[1] += 1
        counts[0] += row["quote_verified"] == "true"
    return {split: (v, t) for split, (v, t) in coverage.items()}
