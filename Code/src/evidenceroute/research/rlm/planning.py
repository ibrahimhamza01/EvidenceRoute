"""Interpret research intent and plan work from local SEC filing metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol, TypeVar

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, StrictInt, StringConstraints, ValidationError, field_validator,
    model_validator,
)

from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.environment import MAX_METADATA_ROWS, CorpusEnvironment, Page
from evidenceroute.research.rlm.tools import validation_message
from evidenceroute.schemas import Filing

_Text = Annotated[str, Field(min_length=1)]
_SectionID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
_FiscalYear = Annotated[StrictInt, Field(ge=1900, le=9999)]
_IntentField = Literal[
    "company_selection", "fiscal_years", "section_ids", "objective", "topic", "comprehensive",
]


class IntentRequirements(BaseModel):
    """Structured interpretation supplied by a caller or an offline model stub."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    company_selection: Literal["all", "explicit"] = Field(description="All tickers in the corpus, or named companies.")
    companies: tuple[_Text, ...] = Field(default=(), description="Exact company names or tickers for explicit scope.")
    fiscal_years: tuple[_FiscalYear, ...] = Field(min_length=1)
    section_ids: tuple[_SectionID, ...] = Field(min_length=1, description="Use canonical section IDs, such as item_1a.")
    objective: Literal["review", "year_over_year"]
    topic: _Text = Field(description="The topic or criteria for evidence selection during analysis.")
    comprehensive: StrictBool = Field(description="Whether answering requires all requested targets to be reviewed.")
    inferred_fields: tuple[_IntentField, ...] = Field(description="Mark inferred requirements. Use [] when all are explicit.")
    scope_notes: _Text | None = Field(default=None, description="Explain scope assumptions or uncertainty, if any.")

    @model_validator(mode="after")
    def check_scope(self) -> IntentRequirements:
        if bool(self.companies) != (self.company_selection == "explicit"):
            raise ValueError("Supply company names or tickers only for explicit selection.")
        if self.objective == "year_over_year" and len(set(self.fiscal_years)) < 2:
            raise ValueError("A year-over-year comparison requires at least two distinct fiscal years.")
        if self.inferred_fields and self.scope_notes is None:
            raise ValueError("Explain any inferred requirements in scope_notes.")
        return self


class ResearchIntent(IntentRequirements):
    question: Annotated[str, StringConstraints(strip_whitespace=False)]

    @field_validator("question")
    @classmethod
    def check_question(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("Research question must be a non-blank string.")
        return question


class IntentInterpreter(Protocol):
    def __call__(self, question: str, output_schema: dict, /) -> Mapping[str, object]:
        """Interpret requirements only. Leave canonical identity resolution to the planner."""
        ...


class IntentInterpretationError(ValueError):
    """The interpreter did not supply usable research requirements."""


def interpret_intent(question: str, interpreter: IntentInterpreter) -> ResearchIntent:
    """Validate an injected interpretation and retain the original question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Research question must be a non-blank string.")
    try:
        raw = interpreter(question, IntentRequirements.model_json_schema())
    except Exception:
        raise IntentInterpretationError("Research intent interpretation failed.") from None
    try:
        requirements = IntentRequirements.model_validate(raw)
    except ValidationError as exc:
        raise IntentInterpretationError(f"Invalid research intent. {validation_message(exc)}") from None
    return ResearchIntent(question=question, **requirements.model_dump())


@dataclass(frozen=True)
class MissingRequirement:
    company: str
    fiscal_year: int
    section_id: str
    reason: Literal["unknown_company", "ambiguous_company", "missing_filing", "ambiguous_filing", "missing_section"]
    candidate_tickers: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompanyPartition:
    ticker: str
    question: str
    targets: tuple[SectionTarget, ...]


@dataclass(frozen=True)
class ResearchPlan:
    intent: ResearchIntent
    requested_companies: tuple[str, ...]
    groups: tuple[CompanyPartition, ...]
    missing_requirements: tuple[MissingRequirement, ...]

    @property
    def resolved_targets(self) -> tuple[SectionTarget, ...]:
        return tuple(target for group in self.groups for target in group.targets)

    @property
    def required_target_count(self) -> int:
        return len(self.resolved_targets) + len(self.missing_requirements)

    def delegation_shortfalls(self, limits: SharedLimits) -> tuple[str, ...]:
        """Check structural limits for one child per resolved group in one run."""
        group_count = sum(bool(group.targets) for group in self.groups)
        if not group_count:
            return ("no_resolved_targets",)
        shortfalls = []
        if limits.max_depth < 1:
            shortfalls.append("max_depth")
        if group_count > limits.max_delegations:
            shortfalls.append("max_delegations")
        if len(self.resolved_targets) > limits.max_partitions:
            shortfalls.append("max_partitions")
        return tuple(shortfalls)


_T = TypeVar("_T")


def _items(fetch_page: Callable[..., Page[_T]]) -> Iterator[_T]:
    offset = 0
    while True:
        page = fetch_page(offset=offset, limit=MAX_METADATA_ROWS)
        yield from page.items
        if page.next_offset is None:
            return
        offset = page.next_offset


def _name_key(name: str) -> str:
    return " ".join(name.casefold().split())


def resolve_research_plan(intent: ResearchIntent, corpus: CorpusEnvironment) -> ResearchPlan:
    """Resolve required company/year/section slots against the snapshot inventory."""
    by_company_year: dict[tuple[str, int], list[Filing]] = defaultdict(list)
    tickers: dict[str, set[str]] = defaultdict(set)
    names: dict[str, set[str]] = defaultdict(set)
    for filing in _items(corpus.list_filings):
        tickers[_name_key(filing.ticker)].add(filing.ticker)
        names[_name_key(filing.company_name)].add(filing.ticker)
        by_company_year[filing.ticker, filing.fiscal_year].append(filing)

    years = sorted(set(intent.fiscal_years))
    section_ids = sorted({section.lower() for section in intent.section_ids})
    selected = set()
    missing = []
    unresolved = set()
    if intent.company_selection == "all":
        selected.update(ticker for matches in tickers.values() for ticker in matches)
    else:
        selectors = sorted({_name_key(company) for company in intent.companies})
        for selector in selectors:
            candidates = tuple(sorted(tickers.get(selector) or names.get(selector, ())))
            if len(candidates) == 1:
                selected.add(candidates[0])
                continue
            unresolved.add(selector)
            reason = "ambiguous_company" if candidates else "unknown_company"
            missing.extend(MissingRequirement(selector, year, section, reason, candidate_tickers=candidates)
                           for year in years for section in section_ids)

    groups = []
    for ticker in sorted(selected):
        targets = []
        for year in years:
            filings = by_company_year[ticker, year]
            if len(filings) != 1:
                reason = "ambiguous_filing" if filings else "missing_filing"
                document_ids = tuple(sorted(filing.document_id for filing in filings))
                missing.extend(MissingRequirement(ticker, year, section, reason, document_ids=document_ids)
                               for section in section_ids)
                continue
            document_id = filings[0].document_id
            available = {section.section_id for section in _items(
                lambda **page: corpus.list_sections(document_id, **page)
            )}
            for section in section_ids:
                if section in available:
                    targets.append(SectionTarget(document_id, section))
                else:
                    missing.append(MissingRequirement(ticker, year, section, "missing_section",
                                                      document_ids=(document_id,)))
        groups.append(CompanyPartition(ticker, intent.question, tuple(targets)))

    missing.sort(key=lambda entry: (entry.company, entry.fiscal_year, entry.section_id))
    return ResearchPlan(intent, tuple(sorted(selected | unresolved)), tuple(groups), tuple(missing))
