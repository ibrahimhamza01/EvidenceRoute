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
from evidenceroute.research.rlm.tools import observation_json


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
            except Exception:
                raise ClaudeModelError("plan_resolution_failed") from None
            plan = resolved
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
                "Do not substitute a related measure, component, balance, or segment figure. "
                "Before finishing, examine each unresolved part of the question. "
                "Unread chunks are not a reason to skip a part while tool slots remain."
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
                " Compare these required source operands and bind each one. A missing operand is insufficient "
                "evidence and does not establish filing-wide absence. Required source operands "
                + observation_json(plan.intent.required_operands) + "."
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
