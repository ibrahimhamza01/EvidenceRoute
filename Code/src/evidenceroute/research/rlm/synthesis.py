"""Bound generic root conclusions to selected, single-section findings."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from evidenceroute.research.rlm.coverage import CoverageReport, SectionTarget
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.tools import FiscalYear, Identifier, StrictInput

AnswerDisposition = Literal["answered", "partial", "insufficient_evidence"]
Note = Annotated[str, Field(min_length=1, max_length=400)]


class SourceOperand(StrictInput):
    model_config = ConfigDict(frozen=True)
    ticker: Identifier
    filing_year: FiscalYear
    section_id: Identifier
    period_text: Note | None = Field(default=None, description="Exact reporting-period text in the bound excerpt, if needed.")


class DraftBinding(SourceOperand):
    ref: str = Field(pattern=r"^(s[1-8]|d[1-9][0-9]{0,8})$")


class BoundOperand(SourceOperand):
    finding_id: Identifier


class Assessment(StrictInput):
    model_config = ConfigDict(frozen=True)
    statement: str = Field(min_length=1, max_length=700)
    disposition: AnswerDisposition = Field(description="Investigator assessment of the requested answer within this root's scope.")
    qualification: Note | None = None


class ConclusionDraft(Assessment):
    bindings: list[DraftBinding] = Field(min_length=2, max_length=8)


class Conclusion(Assessment):
    semantic_status: Literal["unverified"] = "unverified"
    bindings: tuple[BoundOperand, ...] = Field(min_length=2, max_length=8)


def validate_conclusion(record: Conclusion, available: Mapping[str, Finding], required: Sequence[SourceOperand],
                        scope: Sequence[SectionTarget] | None) -> None:
    """Check source identity and literal period anchors. Meaning remains unverified."""
    operands = set()
    for binding in record.bindings:
        finding = available.get(binding.finding_id)
        if finding is None:
            raise ValueError("unknown_finding")
        operand = SourceOperand(**binding.model_dump(exclude={"finding_id"}))
        for item in finding.evidence:
            source = item.source
            if ((source.ticker, source.fiscal_year, source.section_id) !=
                (operand.ticker, operand.filing_year, operand.section_id)
                or scope is not None and SectionTarget(source.document_id, source.section_id) not in scope):
                raise ValueError("wrong_source")
        if operand.period_text and not any(operand.period_text in item.quote for item in finding.evidence):
            raise ValueError("missing_period_text")
        operands.add(operand)
        if operand.period_text:
            operands.add(operand.model_copy(update={"period_text": None}))
    if len({binding.finding_id for binding in record.bindings}) < 2 or not set(required).issubset(operands):
        raise ValueError("missing_operand")


def resolve_conclusions(proposals: object, references: Mapping[str, Finding], required: Sequence[SourceOperand],
                        scope: Sequence[SectionTarget] | None, selected: Sequence[Finding],
                        coverage: CoverageReport) -> tuple[tuple[Conclusion, ...], bool]:
    if proposals == []:
        return (), False
    if not isinstance(proposals, list) or len(proposals) > 3:
        return (), True
    known = {f.finding_id: f for f in references.values()}
    visible = {source.chunk_id for source in coverage.visible_chunks}
    cited = {item.source.chunk_id for f in selected for item in f.evidence}
    inherited = {f.finding_id for ref, f in references.items() if ref.startswith("d")}
    used = {f.finding_id for f in selected}
    records = []
    withheld = False
    for raw in proposals:
        try:
            draft = ConclusionDraft.model_validate(raw)
            bindings = tuple(BoundOperand(**b.model_dump(exclude={"ref"}), finding_id=references[b.ref].finding_id)
                             for b in draft.bindings)
            record = Conclusion(**draft.model_dump(exclude={"bindings"}), bindings=bindings)
            validate_conclusion(record, known, required, scope)
        except (ValueError, KeyError):
            withheld = True
            continue
        ids = {b.finding_id for b in bindings}
        chunks = {item.source.chunk_id for fid in ids for item in known[fid].evidence}
        if not chunks <= visible or len(cited | chunks) > 50 or len((used | ids) & inherited) > 32:
            withheld = True
            continue
        records.append(record)
        used.update(ids)
        cited.update(chunks)
    return tuple(records), withheld


def render_conclusion(record: Conclusion, known: Mapping[str, Finding]) -> str:
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
