"""Shared state and validated research plans."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict, Field

from evidenceroute.schemas import RetrievalResult


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    tickers: list[str] | None = None
    fiscal_years: list[int] | None = None
    section_ids: list[str] | None = None


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[ResearchTask] = Field(
        min_length=1,
        max_length=6,
    )


class AgentState(TypedDict, total=False):
    question: str
    plan: ResearchPlan
    evidence: list[RetrievalResult]
    completed_tasks: list[int]
    errors: list[str]
    trace: list[dict]
    iteration: int
    answer: str
    status: str
    budget: dict
