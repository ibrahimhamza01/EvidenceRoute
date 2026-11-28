"""Bound generic root conclusions to selected, single-section findings."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, ValidationError

from evidenceroute.research.rlm.coverage import CoverageReport, SectionTarget
from evidenceroute.research.rlm.findings import Finding, FindingValidationIssue, finding_schema_issues, unsupported_assertion
from evidenceroute.research.rlm.tools import FiscalYear, Identifier, StrictInput

AnswerDisposition = Literal["answered", "partial", "insufficient_evidence"]
Note = Annotated[str, Field(min_length=1, max_length=400)]
ABSTENTION = "The reviewed evidence is insufficient to answer the question."


class SourceOperand(StrictInput):
    model_config = ConfigDict(frozen=True)
    ticker: Identifier
    filing_year: FiscalYear
    section_id: Identifier | None = Field(description="Null allows any authorized section in this filing.")
    period_text: Note | None = Field(
        default=None,
        description="Leave null unless the question names this period and the bound finding contains that exact text.",
    )


class DraftBinding(SourceOperand):
    section_id: Identifier
    ref: str = Field(
        pattern=r"^(s[1-8]|d[1-9][0-9]{0,8})$",
        description="Acknowledged alias dN or sN from the delivered finding. Do not send finding_id.",
    )


class BoundOperand(SourceOperand):
    section_id: Identifier
    finding_id: Identifier


class Assessment(StrictInput):
    model_config = ConfigDict(frozen=True)
    statement: str = Field(
        min_length=1, max_length=1200,
        description="Answer directly in a few short sentences, at most 1200 characters. Avoid repeating findings.",
    )
    disposition: AnswerDisposition = Field(description="Investigator assessment of the requested answer within this root's scope.")
    qualification: str | None = Field(
        default=None, min_length=1, max_length=600,
        description="Optional caveat in at most 600 characters. Omit it when the statement is enough.",
    )


class ConclusionDraft(Assessment):
    bindings: list[DraftBinding] = Field(
        max_length=8,
        description="Bind each required source to a different acknowledged finding. Use one binding for a single-source "
                    "answer. With insufficient_evidence, cite any useful reviewed passages or leave bindings empty.",
    )


class Conclusion(Assessment):
    semantic_status: Literal["unverified"] = "unverified"
    bindings: tuple[BoundOperand, ...] = Field(max_length=8)


def validate_conclusion(record: Conclusion, available: Mapping[str, Finding], required: Sequence[SourceOperand],
                        scope: Sequence[SectionTarget] | None, *, check_assertions: bool = True) -> None:
    """Check source identity and literal period anchors. Meaning remains unverified."""
    matches: dict[SourceOperand, set[str]] = {}
    evidence = []
    for binding in record.bindings:
        finding = available.get(binding.finding_id)
        if finding is None:
            raise ValueError("unknown_finding")
        if not finding.evidence:
            raise ValueError("unacknowledged_evidence")
        operand = SourceOperand(**binding.model_dump(exclude={"finding_id"}))
        for item in finding.evidence:
            source = item.source
            if ((source.ticker, source.fiscal_year, source.section_id) !=
                (operand.ticker, operand.filing_year, operand.section_id)
                or scope is not None and SectionTarget(source.document_id, source.section_id) not in scope):
                raise ValueError("wrong_source")
        if operand.period_text and not any(operand.period_text in item.quote for item in finding.evidence):
            raise ValueError("missing_period_text")
        evidence.extend(finding.evidence)
        for section in {operand.section_id, None}:
            for period in {operand.period_text, None}:
                key = operand.model_copy(update={"section_id": section, "period_text": period})
                matches.setdefault(key, set()).add(binding.finding_id)
    if record.disposition != "insufficient_evidence":
        if not record.bindings:
            raise ValueError("missing_operand")
        # Each required operand must be independently attributable, even when
        # reporting-period anchors or alternate sections overlap.
        assigned: dict[str, SourceOperand] = {}

        def assign(operand: SourceOperand, seen: set[str]) -> bool:
            for fid in sorted(matches.get(operand, set()) - seen):
                seen.add(fid)
                if fid not in assigned or assign(assigned[fid], seen):
                    assigned[fid] = operand
                    return True
            return False

        if not all(assign(operand, set()) for operand in required):
            raise ValueError("missing_operand")
    if check_assertions and record.bindings and any(
        unsupported_assertion(text, (item.quote for item in evidence), (item.source.fiscal_year for item in evidence))
        for text in (record.statement, record.qualification) if text
    ):
        raise ValueError("unsupported_assertion")


_CONCLUSION_CODES = frozenset({"unknown_finding", "wrong_source", "missing_period_text", "missing_operand",
                               "unacknowledged_evidence", "unsupported_assertion"})


def resolve_conclusions(proposals: object, references: Mapping[str, Finding], required: Sequence[SourceOperand],
                        scope: Sequence[SectionTarget] | None, selected: Sequence[Finding],
                        coverage: CoverageReport) -> tuple[tuple[Conclusion, ...], tuple[FindingValidationIssue, ...]]:
    """Accept supported conclusions and return sanitized reasons for the rest."""
    if proposals == []:
        return (), ()
    if not isinstance(proposals, list) or len(proposals) > 3:
        return (), (FindingValidationIssue("synthesis.conclusions", "schema", "invalid_conclusions"),)
    known = {f.finding_id: f for f in references.values()}
    visible = {source.chunk_id for source in coverage.visible_chunks}
    cited = {item.source.chunk_id for f in selected for item in f.evidence}
    inherited = {f.finding_id for ref, f in references.items() if ref.startswith("d")}
    used = {f.finding_id for f in selected}
    records = []
    rejections: list[FindingValidationIssue] = []

    def reject(issue: FindingValidationIssue) -> None:
        if len(rejections) < 4:
            rejections.append(issue)

    for index, raw in enumerate(proposals):
        path = f"synthesis.conclusions[{index}]"
        try:
            draft = ConclusionDraft.model_validate(raw)
        except ValidationError as exc:
            for issue in finding_schema_issues(exc, path) or (FindingValidationIssue(path, "schema", "invalid_conclusion"),):
                reject(issue)
            continue
        try:
            bindings = tuple(BoundOperand(**b.model_dump(exclude={"ref"}), finding_id=references[b.ref].finding_id)
                             for b in draft.bindings)
            record = Conclusion(**draft.model_dump(exclude={"bindings"}), bindings=bindings)
            validate_conclusion(record, known, required, scope)
        except KeyError:
            reject(FindingValidationIssue(f"{path}.bindings", "provenance", "unacknowledged_finding_reference"))
            continue
        except ValueError as exc:
            code = str(exc)
            if code not in _CONCLUSION_CODES:
                code = "invalid_conclusion"
            reject(FindingValidationIssue(path, "provenance" if code in _CONCLUSION_CODES else "validation", code))
            continue
        if record.disposition == "insufficient_evidence":
            record = record.model_copy(update={"statement": ABSTENTION, "qualification": None})
        ids = {b.finding_id for b in bindings}
        chunks = {item.source.chunk_id for fid in ids for item in known[fid].evidence}
        if not chunks <= visible:
            reject(FindingValidationIssue(path, "provenance", "unacknowledged_evidence"))
            continue
        if len(cited | chunks) > 50 or len((used | ids) & inherited) > 32:
            reject(FindingValidationIssue(path, "limit", "synthesis_limit"))
            continue
        records.append(record)
        used.update(ids)
        cited.update(chunks)
    return tuple(records), tuple(rejections)


def render_conclusion(record: Conclusion, known: Mapping[str, Finding]) -> str:
    if record.disposition == "insufficient_evidence":
        return f"[insufficient_evidence, unverified] {ABSTENTION}"
    findings = [known[b.finding_id] for b in record.bindings]
    notes = dict.fromkeys(note for finding in findings for note in (finding.uncertainty, finding.conflict) if note)
    if record.qualification:
        notes[record.qualification] = None
    chunks = dict.fromkeys(item.source.chunk_id for finding in findings for item in finding.evidence)
    disposition = assessed_disposition(record, known)
    sources = ", ".join(dict.fromkeys(f"{b.ticker} FY{b.filing_year} {b.section_id}" for b in record.bindings))
    return " ".join((f"[{disposition}, unverified] {record.statement}", *notes,
                     f"Sources {sources}. Evidence chunks " + ", ".join(chunks)))


def assessed_disposition(record: Conclusion, known: Mapping[str, Finding]) -> AnswerDisposition:
    if record.disposition == "answered" and any(
        finding.analysis_status != "analyzed" or finding.conflict or finding.uncertainty
        for finding in (known[b.finding_id] for b in record.bindings)
    ):
        return "partial"
    return record.disposition
