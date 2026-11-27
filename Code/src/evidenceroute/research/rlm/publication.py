"""Publish retained research evidence from the invocation ledger."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from evidenceroute.research.rlm.findings import Finding, render_findings
from evidenceroute.research.rlm.research_coverage import InvocationOutcome
from evidenceroute.research.rlm.tools import EvidenceRef

PublicationState = Literal["selected_evidence", "partial_evidence", "insufficient_evidence", "comparison_evidence"]


@dataclass(frozen=True)
class ResearchPublication:
    answer: str
    answer_citations: tuple[EvidenceRef, ...]
    publication_state: PublicationState


def publish_research_findings(
    invocations: Iterable[InvocationOutcome], *, incomplete: bool, compact_comparative: bool = False,
    prior_findings: Iterable[Finding] = (),
) -> ResearchPublication:
    """Project recorded selections and eligible partial findings without reassessing their claims."""
    nodes = tuple(invocations)
    known = {finding.finding_id: finding for finding in prior_findings}
    known.update((finding.finding_id, finding) for node in nodes for finding in node.findings)
    children: dict[str, list[InvocationOutcome]] = defaultdict(list)
    for node in nodes:
        if node.invocation.parent_id is not None:
            children[node.invocation.parent_id].append(node)

    selected: dict[str, Finding] = {}
    partial: dict[str, Finding] = {}
    missing_selection = False
    for root in nodes:
        if root.invocation.parent_id is not None:
            continue
        delivered = {finding.finding_id: finding
                     for child in children[root.invocation.invocation_id] for finding in child.acknowledged_findings}
        if root.stop_reason != "finished":
            partial.update((finding.finding_id, finding) for finding in root.findings)
            partial.update(delivered)
            continue
        local = set(root.selected_local_finding_ids)
        selected.update((finding.finding_id, finding) for finding in root.findings if finding.finding_id in local)
        if root.findings and not local:
            missing_selection = True
        delivered.update((fid, known[fid]) for fid in root.acknowledged_prior_finding_ids if fid in known)
        # Legacy finishes record their inherited selections only in used_finding_ids.
        for fid in root.used_finding_ids:
            if fid in delivered:
                selected[fid] = delivered[fid]
            else:
                missing_selection = True

    partial_findings = tuple(finding for fid, finding in partial.items() if fid not in selected)
    sections = []
    if compact_comparative:
        sections.append("No filing-wide novelty conclusion is established. Investigator comparisons are available "
                        "in the structured report for separate semantic review.")
    if incomplete:
        sections.append("Research is incomplete for the requested scope.")
    if selected or partial_findings:
        sections.append("Semantically unverified investigator findings. Claims apply to the cited passages. "
                        "Question-level sufficiency remains unassessed.")
    if selected:
        sections.append("Findings selected by finished roots.\n" + render_findings(selected.values()))
    if partial_findings:
        sections.append("Partial findings from unfinished execution.\n" + render_findings(partial_findings))
    if missing_selection:
        sections.append("Some finished-root findings lack a reconstructible selection and remain in the structured records.")
    if not selected and not partial_findings:
        sections.append("Insufficient captured evidence eligible for publication.")
    sections.append("Captured findings and lexical scan results cannot establish absence across the requested scope.")
    citations: dict[str, EvidenceRef] = {}
    for finding in (*selected.values(), *partial_findings):
        for item in finding.evidence:
            citations.setdefault(item.source.chunk_id, item.source)
    state: PublicationState = ("selected_evidence" if selected else
                               "partial_evidence" if partial_findings else "insufficient_evidence")
    return ResearchPublication("\n\n".join(sections), tuple(citations.values()), state)
