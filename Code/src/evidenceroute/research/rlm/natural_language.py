"""Run interpretation and scoped corpus research under one overall allowance."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace

from evidenceroute.research.rlm.budget import BudgetExceeded
from evidenceroute.research.rlm.claude import ClaudeModelError
from evidenceroute.research.rlm.controller import ModelAttempt
from evidenceroute.research.rlm.corpus_research import (
    CorpusLimits, CorpusResearchResult, CorpusUsage, execute_corpus_research,
)
from evidenceroute.research.rlm.interpretation import (
    BedrockIntentInterpreter, InterpretationBudget, InterpretationLimits, InterpretationOutcome,
    InterpretationValidationFailure,
)
from evidenceroute.research.rlm.planning import ResearchPlan, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.schemas import ResearchResponse


@dataclass(frozen=True)
class InterpretationReport:
    outcome: InterpretationOutcome
    attempts: tuple[ModelAttempt, ...]
    reported_tokens: int
    estimated_tokens: int
    charged_tokens: int
    attempts_with_missing_usage: int
    elapsed_seconds: float
    validation_failures: tuple[InterpretationValidationFailure, ...] = ()
    correction_attempt: int | None = None


@dataclass(frozen=True)
class QuestionResearchResult:
    question: str
    limits: CorpusLimits
    interpretation_limits: InterpretationLimits
    interpretation: InterpretationReport
    plan: ResearchPlan | None
    research: CorpusResearchResult | None
    usage: CorpusUsage

    @property
    def response(self) -> ResearchResponse:
        if self.research is not None and self.research.response is not None:
            return self.research.response
        outcome = self.interpretation.outcome
        answer = outcome.explanation or (
            "Unable to interpret the research question." if outcome.status == "failed"
            else "The research run could not start within its available budget."
        )
        return ResearchResponse(question=self.question, answer=answer, strategy="rlm", citations=[],
                                verification_status="not_checked")


def _fit_inferred_sections(plan: ResearchPlan, runner: RecursiveResearch, limits: CorpusLimits,
                           max_groups: int) -> ResearchPlan:
    """Drop lower-priority section alternatives when needed to keep each filing in scope."""
    intent = plan.intent
    if intent.objective != "review" or "section_ids" not in intent.inferred_fields or len(intent.section_ids) < 2:
        return plan
    cap = min(limits.max_partitions, runner.shared_limits.max_partitions)
    together = bool(intent.required_operands) and len(plan.groups) <= max_groups
    section_ids = set(intent.section_ids)
    required = {operand.section_id for operand in intent.required_operands}
    original_documents = {pair.document_id for pair in plan.resolved_targets}

    def fits() -> bool:
        sizes = [sum(pair.section_id in section_ids for pair in group.targets) for group in plan.groups]
        return (sum(sizes) if together else max(sizes, default=0)) <= cap

    for section in reversed(intent.section_ids):
        if fits():
            break
        if section not in required and {
            pair.document_id for pair in plan.resolved_targets if pair.section_id in section_ids - {section}
        } == original_documents:
            section_ids.remove(section)
    omitted = [section for section in intent.section_ids if section not in section_ids]
    if not omitted or not section_ids:
        return plan
    bounded = intent.model_copy(update={
        "section_ids": tuple(section for section in intent.section_ids if section in section_ids),
        "scope_notes": f"{intent.scope_notes} Alternate sections outside the partition allowance: {', '.join(omitted)}.",
    })
    return resolve_research_plan(bounded, runner.corpus)


def generate_rlm_answer(
    question: str, runner: RecursiveResearch, *, limits: CorpusLimits,
    interpretation_limits: InterpretationLimits | None = None,
    interpreter_factory: Callable[[], BedrockIntentInterpreter] | None = None,
    max_groups: int = 2, detailed_trace: bool = False,
) -> ResearchResponse:
    """Return the shared answer contract. Use research_question for run diagnostics."""
    return research_question(
        question, runner, limits=limits, interpretation_limits=interpretation_limits,
        interpreter_factory=interpreter_factory, max_groups=max_groups, detailed_trace=detailed_trace,
    ).response


def research_question(
    question: str, runner: RecursiveResearch, *, limits: CorpusLimits,
    interpretation_limits: InterpretationLimits | None = None,
    interpreter_factory: Callable[[], BedrockIntentInterpreter] | None = None,
    max_groups: int = 2, detailed_trace: bool = False,
) -> QuestionResearchResult:
    """Interpret an original question and execute only its resolved, supported scope."""
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise ValueError("Supply a nonblank research question of at most 4,000 characters.")
    if type(max_groups) is not int or max_groups < 0:
        raise ValueError("max_groups must be a nonnegative integer.")
    if runner.comparison_scan is not None or runner.discovery_scan is not None:
        raise ValueError("The question run must build candidate discovery from its own interpreted plan.")
    caps = interpretation_limits if interpretation_limits is not None else InterpretationLimits()
    clock = runner.clock
    started = clock()
    deadline = started + limits.deadline_seconds
    budget = InterpretationBudget(
        max_attempts=min(caps.max_provider_attempts, limits.max_provider_attempts),
        token_limit=min(caps.max_tokens, limits.max_reported_tokens),
        deadline=min(deadline, started + caps.deadline_seconds), clock=clock,
    )
    interpreter = None
    plan = None
    try:
        budget.check_provider_attempt()
        try:
            interpreter = (interpreter_factory() if interpreter_factory is not None
                           else BedrockIntentInterpreter.from_env(clock=clock))
        except Exception:
            raise ClaudeModelError("interpretation_unavailable") from None
        outcome = interpreter.interpret(question, runner.corpus, budget=budget, limits=caps)
        if outcome.intent is not None:
            try:
                resolved = resolve_research_plan(outcome.intent, runner.corpus)
                plan = _fit_inferred_sections(resolved, runner, limits, max_groups)
            except Exception:
                raise ClaudeModelError("plan_resolution_failed") from None
        if clock() >= budget.deadline:
            raise BudgetExceeded("deadline")
    except (BudgetExceeded, ClaudeModelError) as exc:
        outcome = InterpretationOutcome("failed", str(exc))
        plan = None
    except Exception:
        outcome = InterpretationOutcome("failed", "interpretation_failed")
        plan = None

    attempts = (interpreter.last_attempts[-budget.provider_attempts:]
                if interpreter is not None and budget.provider_attempts else ())
    estimated = sum(attempt.estimated_tokens or 0 for attempt in attempts)
    settled = clock()
    interpretation = InterpretationReport(
        outcome, attempts, budget.reported_tokens, estimated, budget.charged_tokens,
        budget.attempts_with_missing_usage, settled - started,
        interpreter.validation_failures if attempts else (),
        interpreter.correction_attempt if attempts else None,
    )
    remaining_seconds = deadline - settled
    result = None
    if plan is not None and remaining_seconds > 0:
        scoped_runner = copy(runner)
        child_extra = f"Original research question: {plan.intent.question}\nResearch topic: {plan.intent.topic}"
        if plan.intent.objective != "year_over_year":
            child_extra += (
                " Preserve the requested metric, unit, period, and qualifiers. "
                "For each historical explanation, cite passages connecting the stated reason to that exact change "
                "in the report that gives it. Describe an evidence gap only within the reviewed passages. "
                "Investigate unresolved parts while tool slots remain."
            )
        scoped_runner.child_guidance = "\n".join(filter(None, (runner.child_guidance, child_extra)))
        root_guidance = (
            "Requested fiscal years " + ", ".join(map(str, sorted(set(plan.intent.fiscal_years)))) + ". "
            "Requested filing sections " + ", ".join(sorted(set(plan.intent.section_ids))) + ". "
            + ("The question requires every requested source. Unread section text does not by itself make supported evidence insufficient."
               if plan.intent.comprehensive else "The question requests a bounded, evidence-based answer.")
        )
        if plan.intent.objective != "year_over_year":
            root_guidance += (
                " Preserve the requested metric, unit, period, and qualifiers. "
                "Do not substitute a related measure, component, balance, or segment figure."
            )
        if plan.intent.required_operands:
            root_guidance += (
                " Required source operands must all support a cross-source answer. If a source is missing, "
                "report the gap in the reviewed passages."
            )
        remaining = limits.model_copy(update={
            "max_provider_attempts": limits.max_provider_attempts - budget.provider_attempts,
            "max_reported_tokens": limits.max_reported_tokens - budget.charged_tokens,
            "deadline_seconds": remaining_seconds,
        })
        result = execute_corpus_research(
            plan, scoped_runner, limits=remaining, max_groups=max_groups, discovery=outcome.discovery,
            root_guidance=root_guidance, detailed_trace=detailed_trace, _deadline=deadline,
        )
    research_usage = result.usage if result else CorpusUsage(
        provider_attempts=0, reported_tokens=0, attempts_with_missing_usage=0,
        estimated_tokens=0, attempts_with_missing_estimate=0, corpus_attempts=0, delegated_invocations=0,
        partition_assignments=0, lexical_pages_read=0, lexical_metadata_reads=0, lexical_chunks_scanned=0,
        elapsed_seconds=0.0,
    )
    usage = replace(
        research_usage, provider_attempts=research_usage.provider_attempts + budget.provider_attempts,
        reported_tokens=research_usage.reported_tokens + budget.reported_tokens,
        attempts_with_missing_usage=research_usage.attempts_with_missing_usage + budget.attempts_with_missing_usage,
        estimated_tokens=research_usage.estimated_tokens + estimated,
        attempts_with_missing_estimate=(research_usage.attempts_with_missing_estimate
                                        + budget.provider_attempts
                                        - sum(attempt.estimated_tokens is not None for attempt in attempts)),
        elapsed_seconds=clock() - started,
    )
    return QuestionResearchResult(question, limits, caps, interpretation, plan, result, usage)
