"""Publish retained research evidence from the invocation ledger."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from evidenceroute.research.rlm.findings import Finding, render_findings
from evidenceroute.research.rlm.planning import ResearchPlan
from evidenceroute.research.rlm.research_coverage import InvocationOutcome
from evidenceroute.research.rlm.synthesis import AnswerDisposition, Conclusion, assessed_disposition, render_conclusion, validate_conclusion
from evidenceroute.research.rlm.tools import EvidenceRef

PublicationState = Literal["selected_evidence", "partial_evidence", "insufficient_evidence", "comparison_evidence"]


@dataclass(frozen=True)
class PublishedConclusion:
    invocation_id: str
    conclusion: Conclusion
    citations: tuple[EvidenceRef, ...]


@dataclass(frozen=True)
class ResearchPublication:
    answer: str
    answer_citations: tuple[EvidenceRef, ...]
    publication_state: PublicationState
    answer_disposition: AnswerDisposition | None = None
    published_conclusions: tuple[PublishedConclusion, ...] = ()


def publish_research_findings(
    invocations: Iterable[InvocationOutcome], *, incomplete: bool, compact_comparative: bool = False,
    prior_findings: Iterable[Finding] = (), generic_plan: ResearchPlan | None = None,
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
    published = []
    conclusion_lines = []
    dispositions = []
    citations: dict[str, EvidenceRef] = {}
    roots = [node for node in nodes if node.invocation.parent_id is None]
    unresolved = any(root.withheld_conclusions or (root.invocation.required_operands and not root.conclusions)
                     for root in roots)
    if generic_plan and generic_plan.intent.required_operands and not roots:
        unresolved = True
    for root in roots:
        delivered = {finding.finding_id: finding
                     for child in children[root.invocation.invocation_id] for finding in child.acknowledged_findings}
        if root.stop_reason != "finished":
            partial.update((finding.finding_id, finding) for finding in root.findings)
            partial.update(delivered)
            continue
        delivered.update((fid, known[fid]) for fid in root.acknowledged_prior_finding_ids if fid in known)
        local = set(root.selected_local_finding_ids)
        eligible = {f.finding_id: f for f in root.findings
                    if f.finding_id in local and f.finding_id in root.acknowledged_local_finding_ids}
        eligible.update((fid, delivered[fid]) for fid in root.selected_inherited_finding_ids if fid in delivered)
        for record in root.conclusions:
            try:
                validate_conclusion(record, eligible, root.invocation.required_operands, root.invocation.partitions)
            except ValueError:
                unresolved = True
                continue
            conclusion_lines.append(render_conclusion(record, known))
            dispositions.append(assessed_disposition(record, known))
            sources = tuple(dict.fromkeys(item.source for binding in record.bindings
                                          for item in known[binding.finding_id].evidence))
            published.append(PublishedConclusion(root.invocation.invocation_id, record, sources))
            for source in sources:
                citations.setdefault(source.chunk_id, source)
        selected.update((finding.finding_id, finding) for finding in root.findings if finding.finding_id in local)
        if root.findings and not local:
            missing_selection = True
        # Legacy finishes record their inherited selections only in used_finding_ids.
        for fid in root.used_finding_ids:
            if fid in delivered:
                selected[fid] = delivered[fid]
            else:
                missing_selection = True

    partial_findings = tuple(finding for fid, finding in partial.items() if fid not in selected)
    sections = []
    disposition = None
    if conclusion_lines:
        sections.append("Conclusions selected by finished roots.\n" + "\n".join(conclusion_lines))
        whole_question = len(roots) == 1 and not unresolved
        if generic_plan is not None and whole_question:
            whole_question = (not generic_plan.missing_requirements
                              and set(roots[0].invocation.partitions or ()) == set(generic_plan.resolved_targets)
                              and (not generic_plan.intent.comprehensive or not incomplete))
        disposition = ("insufficient_evidence" if all(d == "insufficient_evidence" for d in dispositions)
                       else "answered" if whole_question
                       and all(d == "answered" for d in dispositions) else "partial")
    if unresolved:
        gap = "Evidence bindings are missing or invalid for a requested conclusion."
        if generic_plan and generic_plan.intent.required_operands:
            sources = [f"{o.ticker} FY{o.filing_year} {o.section_id}" + (f" ({o.period_text})" if o.period_text else "")
                       for o in generic_plan.intent.required_operands]
            gap += " Required together: " + ", ".join(sources) + "."
        sections.append(gap)
        disposition = disposition or "insufficient_evidence"
    if generic_plan is not None and len(roots) > 1:
        sections.append("Relationships requiring evidence from separate roots remain unresolved.")
    if disposition:
        sections.append(f"Question-level assessment (unverified): {disposition}.")
    if compact_comparative:
        sections.append("No filing-wide novelty conclusion is established. Investigator comparisons are available "
                        "in the structured report for separate semantic review.")
    if incomplete:
        sections.append("Research is incomplete for the requested scope.")
    if selected or partial_findings:
        notice = "Semantically unverified investigator findings. Claims apply to the cited passages."
        if disposition is None:
            notice += " Question-level sufficiency remains unassessed."
        sections.append(notice)
    if selected:
        sections.append("Findings selected by finished roots.\n" + render_findings(selected.values()))
    if partial_findings:
        sections.append("Partial findings from unfinished execution.\n" + render_findings(partial_findings))
    if missing_selection:
        sections.append("Some finished-root findings lack a reconstructible selection and remain in the structured records.")
    if not selected and not partial_findings:
        sections.append("Insufficient captured evidence eligible for publication.")
    sections.append("Captured findings and lexical scan results cannot establish absence across the requested scope.")
    for finding in (*selected.values(), *partial_findings):
        for item in finding.evidence:
            citations.setdefault(item.source.chunk_id, item.source)
    state: PublicationState = ("selected_evidence" if selected else
                               "partial_evidence" if partial_findings else "insufficient_evidence")
    return ResearchPublication("\n\n".join(sections), tuple(citations.values()), state, disposition, tuple(published))
