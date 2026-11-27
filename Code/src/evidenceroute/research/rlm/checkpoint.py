"""Portable checkpoints and scoped continuation of settled research runs."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evidenceroute.research.rlm.checkpoint_validation import CHECKPOINT_VERSIONS, CheckpointError, validate_checkpoint
from evidenceroute.research.rlm.comparison import Comparison
from evidenceroute.research.rlm.corpus_research import (
    BatchOutcome, CorpusGroupOutcome, CorpusLimits, CorpusResearchResult, CorpusUsage, execute_corpus_research,
)
from evidenceroute.research.rlm.discovery import CorpusScan, DiscoverySpec
from evidenceroute.research.rlm.environment import CorpusEnvironment
from evidenceroute.research.rlm.findings import Finding, FindingEvidence
from evidenceroute.research.rlm.interpretation import InterpretationLimits
from evidenceroute.research.rlm.natural_language import QuestionResearchResult
from evidenceroute.research.rlm.planning import CompanyPartition, ResearchPlan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.research_coverage import ResearchCoverage, merge_research_coverage
from evidenceroute.research.rlm.tools import EvidenceRef, observation_json
from evidenceroute.research.rlm.verification_records import VerificationRun, VerificationUsage

MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_RUNS = 64
MAX_VERIFICATION_RUNS = 32


class _CheckpointModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)


class CheckpointInterpretation(_CheckpointModel):
    overall_limits: CorpusLimits
    limits: InterpretationLimits
    charged_tokens: int = Field(ge=0)
    usage: CorpusUsage


class InvocationMetrics(_CheckpointModel):
    invocation_id: str
    provider_attempts: int = Field(ge=0)
    reported_tokens: int = Field(ge=0)
    attempts_with_missing_usage: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    capture_calls: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    comparison_diagnostics: int = Field(ge=0)
    provider_seconds: float | None = Field(default=None, ge=0)
    empty_captures: int | None = Field(default=None, ge=0)


class CheckpointRun(_CheckpointModel):
    run_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    target_companies: tuple[str, ...]
    limits: CorpusLimits
    usage: CorpusUsage
    groups: tuple[CorpusGroupOutcome, ...]
    batches: tuple[BatchOutcome, ...]
    coverage: ResearchCoverage
    root_comparisons: tuple[Comparison, ...]
    acknowledged_child_comparisons: tuple[Comparison, ...]
    citations: tuple[EvidenceRef, ...]
    stop_reason: str
    execution_status: Literal["finished", "incomplete"]
    research_status: Literal["incomplete", "unverified"]
    comparative_capture: bool = False
    invocation_metrics: tuple[InvocationMetrics, ...] = ()

    @classmethod
    def from_result(cls, result: CorpusResearchResult, companies: tuple[str, ...]) -> CheckpointRun:
        return cls(
            run_id=result.run_id, target_companies=companies, limits=result.limits, usage=result.usage,
            groups=result.groups, batches=result.batches, coverage=result.coverage,
            root_comparisons=result.root_comparisons,
            acknowledged_child_comparisons=result.acknowledged_child_comparisons, citations=result.citations,
            stop_reason=result.stop_reason, execution_status=result.execution_status, research_status=result.research_status,
            comparative_capture=result.comparative_capture,
            invocation_metrics=tuple(InvocationMetrics(invocation_id=item["invocation_id"], **item["metrics"])
                                     for item in result.diagnostics),
        )


def _sum_usage(entries: Iterable[CorpusUsage]) -> CorpusUsage:
    entries = tuple(entries)
    return CorpusUsage(**{field.name: sum(getattr(entry, field.name) for entry in entries)
                         for field in fields(CorpusUsage)})


class ResearchCheckpoint(_CheckpointModel):
    version: Literal[1, 2, 3]
    corpus_identity: Annotated[str, Field(pattern=r"^sha256:[a-f0-9]{64}$")]
    plan: ResearchPlan
    runs: tuple[CheckpointRun, ...] = Field(min_length=1, max_length=MAX_CHECKPOINT_RUNS)
    interpretation: CheckpointInterpretation | None = None
    discovery: DiscoverySpec | None = None
    scan_terms: tuple[Annotated[str, Field(min_length=1, max_length=200)], ...] = Field(default=(), max_length=32)
    discovery_cache: CorpusScan | None = None
    verification_runs: tuple[VerificationRun, ...] = Field(default=(), max_length=MAX_VERIFICATION_RUNS)

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(finding for run in self.runs for node in run.coverage.invocations for finding in node.findings)

    @property
    def evidence(self) -> tuple[FindingEvidence, ...]:
        return tuple(dict.fromkeys(item for finding in self.findings for item in finding.evidence))

    @property
    def citations(self) -> tuple[EvidenceRef, ...]:
        return tuple(sorted({ref for run in self.runs for ref in run.citations}, key=lambda ref: ref.chunk_id))

    @property
    def comparisons(self) -> tuple[Comparison, ...]:
        # Key by stable finding identities and assessment content, preserving conflicting assessments.
        return tuple({observation_json(item): item for run in self.runs
                      for item in (*run.root_comparisons, *run.acknowledged_child_comparisons)}.values())

    @property
    def assignments(self) -> tuple[CorpusGroupOutcome, ...]:
        latest: dict[str, CorpusGroupOutcome] = {}
        for run in self.runs:
            selected = set(run.target_companies)
            for group in run.groups:
                if group.ticker not in selected:
                    continue
                previous = latest.get(group.ticker)
                if previous is None or previous.pending_reason is not None or group.pending_reason is None:
                    latest[group.ticker] = group
        return tuple(latest[ticker] for ticker in self.plan.requested_companies)

    @property
    def execution_status(self) -> Literal["finished", "incomplete"]:
        assignments = self.assignments
        return "finished" if assignments and all(group.pending_reason is None for group in assignments) else "incomplete"

    @property
    def research_status(self) -> Literal["incomplete", "unverified"]:
        # A focused continuation alone cannot certify multi-run analytical completeness.
        return self.runs[0].research_status if len(self.runs) == 1 else "incomplete"

    @property
    def historical_usage(self) -> CorpusUsage:
        entries = [run.usage for run in self.runs[:-1]]
        if entries and self.interpretation is not None:
            entries.insert(0, self.interpretation.usage)
        return _sum_usage(entries)

    @property
    def current_usage(self) -> CorpusUsage:
        if len(self.runs) == 1 and self.interpretation is not None:
            return _sum_usage((self.interpretation.usage, self.runs[0].usage))
        return self.runs[-1].usage

    @property
    def cumulative_usage(self) -> CorpusUsage:
        return _sum_usage((self.historical_usage, self.current_usage))

    @property
    def verification_usage(self) -> VerificationUsage:
        return VerificationUsage(**{
            field: sum(getattr(run.usage, field) for run in self.verification_runs)
            for field in VerificationUsage.model_fields
        })

    @property
    def combined_usage(self) -> CorpusUsage:
        research, verification = self.cumulative_usage, self.verification_usage
        fields = ("provider_attempts", "reported_tokens", "estimated_tokens", "attempts_with_missing_usage",
                  "attempts_with_missing_estimate", "elapsed_seconds")
        return replace(research, **{field: getattr(research, field) + getattr(verification, field) for field in fields})

    def merged_coverage(self, corpus: CorpusEnvironment) -> ResearchCoverage:
        if corpus.snapshot_identity != self.corpus_identity:
            raise CheckpointError("Checkpoint corpus identity does not match the loaded snapshot.")
        return merge_research_coverage(corpus, self.plan, (run.coverage for run in self.runs))


@dataclass(frozen=True)
class ContinuationPlan:
    groups: tuple[CompanyPartition, ...]
    blocked_companies: tuple[str, ...]


def create_checkpoint(
    corpus: CorpusEnvironment, result: CorpusResearchResult | QuestionResearchResult, *,
    discovery: DiscoverySpec | None = None, scan_terms: tuple[str, ...] = (),
) -> ResearchCheckpoint:
    """Capture settled structured state, including interpretation usage when available."""
    interpretation = None
    if isinstance(result, QuestionResearchResult):
        report = result.interpretation
        if result.research is None:
            raise CheckpointError("A question checkpoint requires a settled research result.")
        counters = {field.name: getattr(result.usage, field.name) - getattr(result.research.usage, field.name)
                    for field in fields(CorpusUsage)}
        interpretation = CheckpointInterpretation(overall_limits=result.limits, limits=result.interpretation_limits,
                                                  charged_tokens=report.charged_tokens, usage=CorpusUsage(**counters))
        discovery = discovery or report.outcome.discovery
        result = result.research
    if result.scan is not None:
        discovery = discovery or result.scan.discovery
        if discovery is None:
            scan_terms = scan_terms or result.scan.terms
    checkpoint = ResearchCheckpoint(
        version=3 if result.comparative_capture else 1, corpus_identity=corpus.snapshot_identity, plan=result.plan,
        runs=(CheckpointRun.from_result(result, result.plan.requested_companies),), interpretation=interpretation,
        discovery=discovery, scan_terms=scan_terms,
        discovery_cache=_complete_scan(result.scan, result.plan) if discovery is not None and result.comparative_capture else None,
    )
    validate_checkpoint(checkpoint, corpus)
    return checkpoint


def _complete_scan(scan: CorpusScan | None, plan: ResearchPlan) -> CorpusScan | None:
    if (scan is not None and scan.stop_reason is None and not scan.unvisited_targets
            and tuple(row.pair for row in scan.sections) == plan.resolved_targets
            and all(row.inventory_chunks == row.scanned_chunks for row in scan.sections)):
        return scan
    return None


def plan_continuation(checkpoint: ResearchCheckpoint, companies: tuple[str, ...] | None = None) -> ContinuationPlan:
    """Choose unresolved executions by default, with explicit company selection for evidence gaps."""
    plan = checkpoint.plan
    if companies is None:
        completed = {group.ticker for run in checkpoint.runs if run.comparative_capture
                     for group in run.groups if group.outcome and group.outcome.assignment_status == "completed"}
        selected = {group.ticker for group in checkpoint.assignments if group.pending_reason is not None} - completed
    else:
        selected = set(companies)
        if len(selected) != len(companies) or not selected.issubset(plan.requested_companies):
            raise CheckpointError("Select unique companies from the original research plan.")
    blocked = {row.company for row in plan.missing_requirements}
    return ContinuationPlan(tuple(group for group in plan.groups if group.ticker in selected - blocked and group.targets),
                            tuple(sorted(selected & blocked)))


def continue_research(
    checkpoint: ResearchCheckpoint, runner: RecursiveResearch, *, limits: CorpusLimits,
    companies: tuple[str, ...] | None = None, goal: str | None = None, max_groups: int = 2,
) -> ResearchCheckpoint:
    """Run one authorized continuation and append its settled usage and accepted state."""
    validate_checkpoint(checkpoint, runner.corpus)
    selection = plan_continuation(checkpoint, companies)
    if goal is not None and (not isinstance(goal, str) or not goal.strip() or len(goal) > 750):
        raise CheckpointError("Continuation goal must be nonblank and at most 750 characters.")
    if not selection.groups:
        return checkpoint
    if len(checkpoint.runs) >= MAX_CHECKPOINT_RUNS:
        raise CheckpointError("Checkpoint run history limit reached.")
    runner = copy(runner)
    runner._prior_findings = checkpoint.findings
    if goal:
        runner.child_guidance = "\n".join(filter(None, (runner.child_guidance, f"Continuation goal: {goal}")))
    targets = tuple(group.ticker for group in selection.groups)
    result = execute_corpus_research(
        checkpoint.plan, runner, limits=limits, max_groups=max_groups, discovery=checkpoint.discovery,
        scan_terms=checkpoint.scan_terms or None, root_guidance=goal, _target_companies=targets,
        _cached_scan=checkpoint.discovery_cache,
    )
    updated = checkpoint.model_copy(update={
        "version": 3 if result.comparative_capture else checkpoint.version,
        "runs": (*checkpoint.runs, CheckpointRun.from_result(result, targets)),
    })
    validate_checkpoint(updated, runner.corpus)
    return updated


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    properties = dict(pairs)
    if len(properties) != len(pairs):
        raise CheckpointError("Checkpoint JSON contains duplicate keys.")
    return properties


def _reject_nonfinite(_: str) -> None:
    raise CheckpointError("Checkpoint JSON contains a nonfinite number.")


def _decode_checkpoint(payload: bytes) -> ResearchCheckpoint:
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise CheckpointError("Checkpoint exceeds the input size limit.")
    try:
        data = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite)
    except CheckpointError:
        raise
    except (ValueError, RecursionError):
        raise CheckpointError("Malformed checkpoint JSON.") from None
    if not isinstance(data, dict) or type(data.get("version")) is not int:
        raise CheckpointError("Checkpoint schema version is missing or invalid.")
    if data["version"] not in CHECKPOINT_VERSIONS:
        raise CheckpointError("Unsupported checkpoint schema version.")
    del data
    try:
        return ResearchCheckpoint.model_validate_json(payload, strict=True)
    except (ValidationError, RecursionError, ValueError):
        raise CheckpointError("Checkpoint does not match its schema.") from None


def load_checkpoint(path: Path | str, corpus: CorpusEnvironment) -> ResearchCheckpoint:
    """Read a bounded UTF-8 JSON file and validate it against the canonical corpus."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CheckpointError("Checkpoint must be a regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            checkpoint = _decode_checkpoint(source.read(MAX_CHECKPOINT_BYTES + 1))
    finally:
        os.close(descriptor)
    validate_checkpoint(checkpoint, corpus)
    return checkpoint


def save_checkpoint(path: Path | str, checkpoint: ResearchCheckpoint) -> None:
    """Write one checkpoint atomically using a same-directory, private temporary file."""
    target = Path(path)
    exclude = {"verification_runs"} if not checkpoint.verification_runs else set()
    if checkpoint.discovery_cache is None:
        exclude.add("discovery_cache")
    payload = checkpoint.model_dump_json(exclude=exclude).encode("utf-8")
    _decode_checkpoint(payload)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
                                delete=False) as pending:
            temporary = Path(pending.name)
            pending.write(payload)
            pending.flush()
            os.fsync(pending.fileno())
        os.replace(temporary, target)
        temporary = None
        if os.name == "posix":
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
