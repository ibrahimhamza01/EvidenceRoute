"""Structured semantic-review records shared by verification and checkpoints."""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from evidenceroute.research.rlm.comparison import Comparison
from evidenceroute.research.rlm.findings import Finding
from evidenceroute.research.rlm.tools import Identifier, StrictInput, observation_json

VerificationStatus = Literal["supported", "partially_supported", "contradicted", "insufficient_evidence", "unverified"]
ComponentStatus = Literal["supported", "contradicted", "insufficient_evidence"]
RiskRelationship = Literal["same_risk", "more_specific_wording", "different_risk", "unclear"]
EarlierYearEvidence = Literal["supports_risk", "explicitly_opposes", "nonmention", "insufficient"]
UnverifiedReason = Literal[
    "attempt_limit", "token_limit", "deadline", "item_limit", "input_limit", "output_limit",
    "invalid_assessment", "missing_assessment", "invalid_response", "transport_error", "truncated_response",
    "not_requested",
]
CaseKind = Literal["finding", "comparison"]


class _Record(StrictInput):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


class VerificationLimits(_Record):
    max_provider_attempts: int = Field(default=8, ge=0, le=64)
    max_tokens: int = Field(default=32_000, ge=0)
    max_output_tokens: int = Field(default=1_024, ge=1, le=4_096)
    max_items: int = Field(default=16, ge=0, le=128)
    batch_size: int = Field(default=2, ge=1, le=4)
    max_input_bytes: int = Field(default=16_000, ge=256, le=48_000)
    deadline_seconds: float = Field(default=60.0, gt=0)


class VerificationUsage(_Record):
    provider_attempts: int = Field(default=0, ge=0)
    reported_tokens: int = Field(default=0, ge=0)
    charged_tokens: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    attempts_with_missing_usage: int = Field(default=0, ge=0)
    attempts_with_missing_estimate: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> VerificationUsage:
        if (self.charged_tokens < self.reported_tokens or self.attempts_with_missing_usage > self.provider_attempts
                or self.attempts_with_missing_estimate > self.provider_attempts):
            raise ValueError("Inconsistent verification usage.")
        return self


class EvidencePointer(_Record):
    finding_id: Identifier
    evidence_index: int = Field(ge=0, le=2)


class ComponentReview(_Record):
    statement: str = Field(min_length=1, max_length=220)
    status: ComponentStatus
    evidence: tuple[EvidencePointer, ...] = Field(max_length=4)
    gap: Annotated[str, Field(min_length=1, max_length=200)] | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> ComponentReview:
        if self.status == "insufficient_evidence" and self.gap is None:
            raise ValueError("Insufficient evidence requires a gap.")
        if self.status != "insufficient_evidence" and not self.evidence:
            raise ValueError("Support or contradiction requires selected evidence.")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("Use each evidence reference once per component.")
        return self


class ComparisonReview(_Record):
    relationship: RiskRelationship
    relationship_evidence: tuple[EvidencePointer, ...] = Field(max_length=6)
    earlier_year: EarlierYearEvidence
    earlier_evidence: tuple[EvidencePointer, ...] = Field(max_length=4)
    gap: Annotated[str, Field(min_length=1, max_length=200)] | None = None

    @property
    def novelty(self) -> Literal["shared_risk_in_selected_evidence", "possible_novelty_needs_review", "unresolved"]:
        if self.relationship in {"same_risk", "more_specific_wording"}:
            return "shared_risk_in_selected_evidence"
        if self.relationship == "different_risk":
            return "possible_novelty_needs_review"
        return "unresolved"

    @model_validator(mode="after")
    def validate_review(self) -> ComparisonReview:
        if self.earlier_year != "insufficient" and not self.earlier_evidence:
            raise ValueError("An earlier-year assessment requires selected earlier evidence.")
        if (self.relationship in {"different_risk", "unclear"} or self.earlier_year in {"nonmention", "insufficient"}) and not self.gap:
            raise ValueError("An unresolved relationship or earlier year requires an evidence gap.")
        return self


def component_status(components: tuple[ComponentReview, ...]) -> VerificationStatus:
    states = {item.status for item in components}
    if not states:
        return "unverified"
    if "contradicted" in states:
        return "contradicted"
    if states == {"supported"}:
        return "supported"
    if "supported" in states:
        return "partially_supported"
    return "insufficient_evidence"


class VerificationItem(_Record):
    kind: CaseKind
    target_id: Identifier
    finding_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=16)
    subject_digest: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    status: VerificationStatus
    components: tuple[ComponentReview, ...] = Field(default=(), max_length=8)
    comparison: ComparisonReview | None = None
    unverified_reason: UnverifiedReason | None = None

    @model_validator(mode="after")
    def validate_status(self) -> VerificationItem:
        if (self.status != component_status(self.components)
                or (self.unverified_reason is not None) != (self.status == "unverified")
                or (self.comparison is not None) != (self.kind == "comparison" and self.status != "unverified")
                or (self.kind == "finding" and self.finding_ids != (self.target_id,))
                or len(set(self.finding_ids)) != len(self.finding_ids)):
            raise ValueError("Inconsistent verification result.")
        return self


class VerificationRun(_Record):
    run_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    corpus_identity: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    research_run_ids: tuple[Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")], ...] = Field(min_length=1, max_length=64)
    limits: VerificationLimits
    usage: VerificationUsage
    items: tuple[VerificationItem, ...]

    @model_validator(mode="after")
    def validate_limits(self) -> VerificationRun:
        reviewed = sum(item.status != "unverified" for item in self.items)
        if (self.usage.provider_attempts > self.limits.max_provider_attempts
                or reviewed > min(self.limits.max_items, self.usage.provider_attempts * self.limits.batch_size)
                or len({(item.kind, item.target_id) for item in self.items}) != len(self.items)):
            raise ValueError("Inconsistent verification inventory or usage.")
        return self


def comparison_id(comparison: Comparison) -> str:
    """Stable identity of a specific historical conclusion and its evidence links."""
    return sha256(observation_json(comparison).encode("utf-8")).hexdigest()


def subject_digest(findings: tuple[Finding, ...], comparison: Comparison | None) -> str:
    """Bind a review to its original claim and exact evidence selection."""
    content = {"findings": [{"claim": item.claim, "evidence": item.evidence} for item in findings],
               "comparison_id": comparison_id(comparison) if comparison is not None else None}
    return sha256(observation_json(content).encode("utf-8")).hexdigest()


def validate_review_evidence(item: VerificationItem, findings: dict[str, Finding], comparison: Comparison | None) -> None:
    """Check evidence links and the structural limits of a comparative assessment."""
    if item.subject_digest != subject_digest(tuple(findings[fid] for fid in item.finding_ids), comparison):
        raise ValueError("Verification subject differs from its original claim or selected evidence.")

    def year(pointer: EvidencePointer) -> int:
        if pointer.finding_id not in item.finding_ids or pointer.evidence_index >= len(findings[pointer.finding_id].evidence):
            raise ValueError("Verification references evidence outside its case.")
        return findings[pointer.finding_id].evidence[pointer.evidence_index].source.fiscal_year

    component_years = [set(map(year, component.evidence)) for component in item.components]
    review = item.comparison
    if review is None:
        return
    assert comparison is not None
    earlier, later = comparison.years
    if any(year(pointer) != earlier for pointer in review.earlier_evidence):
        raise ValueError("Earlier-year assessment cites evidence from the wrong year.")
    relationship_years = set(map(year, review.relationship_evidence))
    if review.relationship != "unclear" and relationship_years != {earlier, later}:
        raise ValueError("A resolved relationship requires selected evidence from both years.")
    if (comparison.outcome == "comparable_both_years" and item.status == "supported"
            and review.relationship not in {"same_risk", "more_specific_wording"}):
        raise ValueError("Support for the same underlying risk requires a shared relationship.")
    if review.earlier_year in {"nonmention", "insufficient"} and (
        any(component.status == "contradicted" and years == {earlier}
            for component, years in zip(item.components, component_years))
        or (comparison.outcome == "conflicting" and item.status == "supported")
    ):
        raise ValueError("Earlier-year nonmention or insufficient evidence cannot establish contradiction.")
