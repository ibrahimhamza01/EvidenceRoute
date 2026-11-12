"""Retrieval metrics computed against gold evidence groups.

A question's gold evidence is a dict mapping each evidence group to the set of
chunk IDs that contain it, e.g. {"g1": {"a1"}, "g2": {"b1", "b2"}}.
"""

from __future__ import annotations


def recall_at_k(gold_groups: dict[str, set[str]], ranked_chunk_ids: list[str], k: int) -> float:
    """Share of evidence groups with at least one chunk in the top k results."""
    if not gold_groups:
        raise ValueError("Recall is undefined for a question without gold evidence")
    top_k = set(ranked_chunk_ids[:k])
    found = sum(1 for chunks in gold_groups.values() if chunks & top_k)
    return found / len(gold_groups)


def precision_at_k(gold_groups: dict[str, set[str]], ranked_chunk_ids: list[str], k: int) -> float:
    """Share of the top k results that are relevant (supporting) chunks."""
    relevant = set().union(*gold_groups.values()) if gold_groups else set()
    hits = sum(1 for chunk_id in ranked_chunk_ids[:k] if chunk_id in relevant)
    return hits / k


def reciprocal_rank(gold_groups: dict[str, set[str]], ranked_chunk_ids: list[str], k: int) -> float:
    """1 / rank of the first relevant chunk in the top k, or 0 if there is none."""
    relevant = set().union(*gold_groups.values()) if gold_groups else set()
    for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def full_coverage_at_k(gold_groups: dict[str, set[str]], ranked_chunk_ids: list[str], k: int) -> bool:
    """True if every evidence group is found in the top k results."""
    return recall_at_k(gold_groups, ranked_chunk_ids, k) == 1.0
