"""JSON tool contracts and dispatch for the local SEC corpus."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evidenceroute.research.rlm.environment import (
    MAX_CHUNK_ROWS,
    MAX_METADATA_ROWS,
    MAX_QUERY_CHARS,
    MAX_SECTION_CHARS,
    CorpusEnvironment,
    Page,
    SectionWindow,
)
from evidenceroute.retrieval import ChunkFilter
from evidenceroute.research.rlm.spans import passage_spans
from evidenceroute.schemas import Chunk, RetrievalResult

if TYPE_CHECKING:
    from evidenceroute.research.rlm.scoped import ScopedCorpus

MIN_RESPONSE_BYTES = 128
MAX_TOOL_BATCH = 4
# This only bounds protocol recovery. A batch above MAX_TOOL_BATCH never executes.
MAX_PROTOCOL_TOOL_BATCH = 16
# Internal batches preserve native IDs, including Bedrock's period and colon.
TOOL_USE_ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


def trace_data(value):
    """Keep tool telemetry JSON-only and mask credential fields and AWS identifiers."""
    if isinstance(value, dict):
        return {
            trace_data(key): "<redacted>" if any(token in re.sub(r"[^a-z]", "", str(key).lower())
                                                 for token in ("authorization", "credential", "apikey",
                                                               "accesskey", "sessiontoken", "securitytoken",
                                                               "secretkey", "secretaccesskey", "accountid",
                                                               "reasoning", "thinking"))
            else trace_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [trace_data(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?<!\w)\d{12}(?!\w)|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bsk-ant-[\w-]+",
            "<redacted>", value,
        )
    return value
Identifier = Annotated[str, Field(min_length=1, max_length=160)]
FiscalYear = Annotated[int, Field(ge=1900, le=9999)]


class StrictInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)


class _Page(StrictInput):
    offset: int = Field(default=0, ge=0, description="Use next_offset from a successful response to continue.")


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
    limit: int = Field(default=MAX_SECTION_CHARS, ge=1, le=MAX_SECTION_CHARS,
                       description="Maximum text characters, measured separately from the response byte cap.")


class _ChunkPage(_Page):
    limit: int = Field(default=1, ge=1, le=MAX_CHUNK_ROWS,
                       description="Maximum whole chunks. Start with 1 for targeted questions.")


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
    "get_section": ("Read original section text by character offset and limit.", _Section),
    "read_chunks": ("Page through whole original chunks in order. Set section_id to scope the read.", _Read),
    "get_chunk": ("Look up one original chunk by its ID.", _Chunk),
    "search_chunks": ("Rank keyword matches as whole chunks. Use a short query and filter to the target "
                      "filing and section when known. Returned chunks include text and citation IDs.", _Search),
}


def tool_schemas(*, max_response_bytes: int | None = None) -> list[dict]:
    """Return fresh JSON-compatible tool definitions with JSON Schema inputs."""
    tools = [
        {"name": name, "description": description, "input_schema": args.model_json_schema()}
        for name, (description, args) in _TOOLS.items()
    ]
    if max_response_bytes is not None:
        for tool in tools:
            if tool["name"] in {"get_section", "read_chunks", "search_chunks"}:
                tool["description"] += f" This run caps each observation at {max_response_bytes} UTF-8 bytes."
    return tools


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


def error_observation(code: str, message: str, max_bytes: int, **details: int) -> str:
    error = {"code": code, **details}
    for candidate in ({**error, "message": message}, error):
        content = observation_json({"ok": False, "error": candidate})
        if len(content.encode("utf-8")) <= max_bytes:
            return content
    return observation_json({"ok": False, "error": {"code": code}})


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

    def __init__(self, corpus: CorpusEnvironment | ScopedCorpus, *, max_response_bytes: int, include_spans: bool = False):
        if type(max_response_bytes) is not int or max_response_bytes < MIN_RESPONSE_BYTES:
            raise ValueError(f"max_response_bytes must be at least {MIN_RESPONSE_BYTES}")
        self._corpus = corpus
        self._max_bytes = max_response_bytes
        self._include_spans = include_spans

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
            chunks = self._result_chunks(name, result)
            evidence = tuple(EvidenceRef.from_chunk(chunk) for chunk in chunks)
            discovered = evidence if name == "search_chunks" else ()
            body = {"ok": True, "result": result}
            if self._include_spans and chunks:
                body["span_ids"] = {chunk.chunk_id: list(passage_spans(chunk.text)) for chunk in chunks}
            content = observation_json(body)
            response_bytes = len(content.encode("utf-8"))
            if response_bytes > self._max_bytes:
                return self._oversized(name, validated, result, response_bytes, discovered)
            return DispatchResult(validated, content, evidence, discovered)
        except KeyError:
            return self._error("unknown_id", "No corpus record matches the supplied IDs.", validated)
        except Exception as exc:
            return self._error("tool_error", type(exc).__name__, validated)

    def _oversized(
        self, name: str, arguments: dict, result, response_bytes: int, discovered: tuple[EvidenceRef, ...],
    ) -> DispatchResult:
        details = {"max_response_bytes": self._max_bytes, "response_bytes": response_bytes}
        if isinstance(result, SectionWindow) and len(result.text) > 1:
            details["suggested_limit"] = len(result.text) // 2
        elif isinstance(result, Page) and len(result.items) > 1:
            details["suggested_limit"] = 1
        if "suggested_limit" in details:
            message = "No results delivered. Retry with suggested_limit and the same offset and filters."
        elif name in {"search_chunks", "read_chunks", "get_chunk"}:
            message = "No chunk delivered. Try get_section with a small character limit if the section is known."
        else:
            message = "No result delivered. This request is already at its smallest size."
        content = error_observation("response_too_large", message, self._max_bytes, **details)
        return DispatchResult(arguments, content, search_discovered=discovered)

    @staticmethod
    def _result_chunks(name: str, result) -> tuple[Chunk, ...]:
        if name == "get_chunk":
            return (result,)
        if name in {"read_chunks", "search_chunks"}:
            return tuple(item.chunk if isinstance(item, RetrievalResult) else item for item in result.items)
        return ()

    def _error(
        self, code: str, message: str, arguments: dict | None = None,
    ) -> DispatchResult:
        return DispatchResult(arguments, error_observation(code, message, self._max_bytes))
