"""Read-only publication checks against saved live research and review artifacts."""

import argparse
import json
import re
from pathlib import Path

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.checkpoint import load_checkpoint
from evidenceroute.research.rlm.tools import EvidenceRef, observation_json
from evidenceroute.research.rlm.verification import verification_report


def printed_ids(answer, pattern):
    return tuple(dict.fromkeys(key for group in re.findall(pattern, answer) for key in group.split(", ")))


def assert_citations(answer, citations, pattern, corpus):
    assert printed_ids(answer, pattern) == tuple(ref.chunk_id for ref in citations)
    assert all(ref == EvidenceRef.from_chunk(corpus.get_chunk(ref.chunk_id)) for ref in citations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="Directory of external saved reports and checkpoints.")
    artifacts = parser.parse_args().artifacts
    corpus = CorpusEnvironment()
    names = (
        "rlm-final-corpus-continued-checkpoint.json", "rlm-final-review-pass3-checkpoint.json",
        "rlm-live-initial-checkpoint.json", "rlm-live-continued-checkpoint.json",
    )
    findings_pattern = r"Evidence chunks ([0-9a-f]{24}(?:, [0-9a-f]{24})*)"
    verified_pattern = r"\[([0-9a-f]{24}(?:, [0-9a-f]{24})*)\]"
    continued, reviewed, *descriptive = (load_checkpoint(artifacts / name, corpus) for name in names)
    for name, checkpoint in zip(names, (continued, reviewed, *descriptive)):
        publication = checkpoint.publication
        assert_citations(publication.answer, publication.answer_citations, findings_pattern, corpus)
        print(f"{name}: {publication.publication_state}, {len(re.findall(findings_pattern, publication.answer))} "
              f"findings rendered, {len(publication.answer_citations)} canonical answer chunks")
    assert continued.publication == reviewed.publication
    assert len(continued.findings) == len(reviewed.findings) == 44
    assert continued.runs == reviewed.runs and len(reviewed.comparisons) == 33
    assert len(continued.publication.answer_citations) == 30
    assert len(re.findall(findings_pattern, continued.publication.answer)) == 36

    before = json.loads((artifacts / "rlm-final-corpus-continued-live.json").read_bytes())["final_report"]
    assert not printed_ids(before["answer"], verified_pattern)
    print("Before:", next(line.strip() for line in before["answer"].splitlines() if "AI semiconductor supply" in line))
    print("After:", next(line for line in continued.publication.answer.splitlines() if line.startswith("AMZN FY2024")))

    report = verification_report(reviewed)
    assert_citations(report["answer"], report["answer_citations"], verified_pattern, corpus)
    assert len(report["answer_citations"]) == 31 and len(report["conclusion_citations"]) == 21
    assert sum(bool(row["verified_relationship"]) for key in ("conclusions", "recovered_comparisons")
               for row in report[key]) == 10
    assert report["outstanding_verification"] == {"findings": 0, "comparisons": 1}
    assert report["research_status"] == "incomplete"
    assert len(report["assignment_evidence"]) == 25
    assert all(row["novelty_unresolved"] for row in report["assignment_evidence"])

    saved = json.loads((artifacts / "rlm-final-review-pass3.json").read_bytes())["reports"][-1]
    current = json.loads(observation_json(report))
    stable_keys = ("items", "comparative_novelty", "original_findings", "original_comparisons", "research_usage",
                   "verification_usage", "current_verification_usage", "combined_usage")
    assert all(current[key] == saved[key] for key in stable_keys)
    print("Verified replay: 44 findings, 33 comparisons, 31 answer chunks, 21 conclusion chunks, 10 relationships, "
          "1 outstanding comparison, incomplete research, 25 unresolved novelty conclusions, unchanged judgments and usage.")


if __name__ == "__main__":
    main()
