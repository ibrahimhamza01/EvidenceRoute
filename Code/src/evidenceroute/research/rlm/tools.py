"""JSON tool contracts and dispatch for the local SEC corpus."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evidenceroute.research.rlm.environment import (
    MAX_CHUNK_ROWS,
    MAX_METADATA_ROWS,
    MAX_QUERY_CHARS,
    MAX_SECTION_CHARS,
    CorpusEnvironment,
)
from evidenceroute.retrieval import ChunkFilter
from evidenceroute.schemas import Chunk, RetrievalResult

MIN_RESPONSE_BYTES = 128
Identifier = Annotated[str, Field(min_length=1, max_length=160)]
FiscalYear = Annotated[int, Field(ge=1900, le=9999)]


class StrictInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)


class _Page(StrictInput):
    offset: int = Field(default=0, ge=0)


class _MetadataPage(_Page):
    limit: int = Field(default=10, ge=1, le=MAX_METADATA_ROWS)


class _Companies(_MetadataPage):
    limit: int = Field(default=MAX_METADATA_ROWS, ge=1, le=MAX_METADATA_ROWS)


class _Filings(_MetadataPage):
    ticker: Identifier | None = None
    fiscal_year: FiscalYear | None = None


class _Filing(StrictInput):
    document_id: Identifier


class _Sections(_MetadataPage, _Filing):
    pass


class _Section(_Page, _Filing):
    section_id: Identifier
    limit: int = Field(default=MAX_SECTION_CHARS, ge=1, le=MAX_SECTION_CHARS)


class _ChunkPage(_Page):
    limit: int = Field(default=MAX_CHUNK_ROWS, ge=1, le=MAX_CHUNK_ROWS)


class _Read(_ChunkPage, _Filing):
    section_id: Identifier | None = None


class _Chunk(StrictInput):
    chunk_id: Identifier


class _Filters(StrictInput):
    tickers: list[Identifier] | None = Field(default=None, min_length=1, max_length=25)
    fiscal_years: list[FiscalYear] | None = Field(default=None, min_length=1, max_length=25)
    document_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=25)
    section_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=25)


class _Search(_ChunkPage):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    filters: _Filters | None = None


_TOOLS: dict[str, tuple[str, type[StrictInput]]] = {
    "list_companies": ("List companies and available fiscal years.", _Companies),
    "list_filings": ("List filings, optionally filtered by ticker and fiscal year.", _Filings),
    "get_filing": ("Look up original metadata for one filing.", _Filing),
    "list_sections": ("List a filing's section metadata and chunk counts.", _Sections),
    "get_section": ("Read a character window from one original processed section.", _Section),
    "read_chunks": ("Read a page of whole original chunks from a filing or section.", _Read),
    "get_chunk": ("Look up one original chunk by its ID.", _Chunk),
    "search_chunks": ("Rank keyword matches with optional metadata filters.", _Search),
}


def tool_schemas() -> list[dict]:
    """Return fresh JSON-compatible tool definitions with JSON Schema inputs."""
    return [
        {"name": name, "description": description, "input_schema": args.model_json_schema()}
        for name, (description, args) in _TOOLS.items()
    ]


def _json_default(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return asdict(value)


def observation_json(observation: dict) -> str:
    return json.dumps(observation, default=_json_default, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def validation_message(error: ValidationError) -> str:
    first = error.errors(include_url=False, include_context=False, include_input=False)[0]
    location = ".".join(map(str, first["loc"]))[:40]
    return f"{location or 'input'} ({first['type'][:30]})"


def error_observation(code: str, message: str, max_bytes: int) -> str:
    content = observation_json({"ok": False, "error": {"code": code, "message": message}})
    if len(content.encode("utf-8")) > max_bytes:
        return observation_json({"ok": False, "error": {"code": code}})
    return content


@dataclass(frozen=True)
class EvidenceRef:
    document_id: str
    section_id: str
    chunk_id: str
    ticker: str
    fiscal_year: int
    source_url: str

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> EvidenceRef:
        return cls(chunk.document_id, chunk.section_id, chunk.chunk_id,
                   chunk.ticker, chunk.fiscal_year, chunk.source_url)


@dataclass(frozen=True)
class DispatchResult:
    arguments: dict | None
    content: str
    evidence: tuple[EvidenceRef, ...] = ()
    search_discovered: tuple[EvidenceRef, ...] = ()


class CorpusToolDispatcher:
    """Validate tool requests and serialize the exact model-facing observation."""

    def __init__(self, corpus: CorpusEnvironment, *, max_response_bytes: int):
        if type(max_response_bytes) is not int or max_response_bytes < MIN_RESPONSE_BYTES:
            raise ValueError(f"max_response_bytes must be at least {MIN_RESPONSE_BYTES}")
        self._corpus = corpus
        self._max_bytes = max_response_bytes

    def dispatch(self, name: str, arguments: dict) -> DispatchResult:
        if not isinstance(name, str) or name not in _TOOLS:
            return self._error("unknown_tool", "Tool is outside the SEC corpus allowlist.")
        try:
            validated = _TOOLS[name][1].model_validate(arguments).model_dump(mode="json")
        except ValidationError as exc:
            return self._error("invalid_arguments", validation_message(exc))

        call_args = validated.copy()
        if name == "search_chunks" and validated["filters"] is not None:
            filters = ChunkFilter.model_validate(validated["filters"])
            validated["filters"] = filters.model_dump(mode="json", exclude_none=True)
            call_args["filters"] = filters
        try:
            # Only names in the fixed registry reach attribute lookup.
            result = getattr(self._corpus, name)(**call_args)
            evidence = self._evidence(name, result)
            discovered = evidence if name == "search_chunks" else ()
            content = observation_json({"ok": True, "result": result})
            if len(content.encode("utf-8")) > self._max_bytes:
                return self._error(
                    "response_too_large", "Reduce the limit or narrow the request.",
                    validated, discovered,
                )
            return DispatchResult(validated, content, evidence, discovered)
        except KeyError:
            return self._error("unknown_id", "No corpus record matches the supplied IDs.", validated)
        except Exception as exc:
            return self._error("tool_error", type(exc).__name__, validated)

    @staticmethod
    def _evidence(name: str, result) -> tuple[EvidenceRef, ...]:
        if name == "get_chunk":
            return (EvidenceRef.from_chunk(result),)
        if name in {"read_chunks", "search_chunks"}:
            return tuple(
                EvidenceRef.from_chunk(item.chunk if isinstance(item, RetrievalResult) else item)
                for item in result.items
            )
        return ()

    def _error(
        self, code: str, message: str, arguments: dict | None = None,
        discovered: tuple[EvidenceRef, ...] = (),
    ) -> DispatchResult:
        return DispatchResult(arguments, error_observation(code, message, self._max_bytes),
                              search_discovered=discovered)
