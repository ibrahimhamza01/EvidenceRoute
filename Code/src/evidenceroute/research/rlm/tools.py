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
SUBMIT_FINDINGS = "submit_findings"
READ_PRIOR_FINDINGS = "read_prior_findings"


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


def trace_research_action(action):
    """Omit model finding inputs from saved requests. Resolved findings are reported separately."""
    if not isinstance(action, dict):
        return trace_data(action)
    name = action.get("name")
    if isinstance(name, str) and name.strip() in (SUBMIT_FINDINGS, "finish_research"):
        return trace_data({key: action[key] for key in ("type", "name", "tool_use_id") if key in action})
    if action.get("type") in ("finish", "synthesis"):
        return {"type": action["type"]}
    if action.get("type") == "tool_batch" and isinstance(action.get("calls"), list):
        return {"type": "tool_batch", "calls": [trace_research_action(call) for call in action["calls"]]}
    return trace_data(action)


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


class _Spans(_Chunk):
    span_ids: list[Annotated[str, Field(pattern=r"^\d{1,8}:\d{1,8}$")]] = Field(min_length=1, max_length=4)


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


def tool_schemas(
    *, max_response_bytes: int | None = None, include_spans: bool = False, comparative: bool = False,
) -> list[dict]:
    """Return fresh JSON-compatible tool definitions with JSON Schema inputs."""
    tools = [
        {"name": name, "description": description, "input_schema": args.model_json_schema()}
        for name, (description, args) in _TOOLS.items()
    ]
    if include_spans or comparative:
        tools.append({"name": "get_spans", "description": "Read up to four exact passages in one scoped chunk. "
                      "Use span IDs from the lexical shortlist, candidate page, or search results.",
                      "input_schema": _Spans.model_json_schema()})
        for tool in tools:
            if tool["name"] == "search_chunks" and comparative:
                tool["description"] = "Rank scoped keyword candidates. Returns IDs and matching span IDs. Use get_spans for text."
            elif tool["name"] == "get_chunk":
                tool["description"] += " Use get_spans for targeted excerpts. Already delivered chunks are referenced by ID."
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

    def __init__(
        self, corpus: CorpusEnvironment | ScopedCorpus, *, max_response_bytes: int, include_spans: bool = False,
        candidate_search: bool | None = None,
    ):
        if type(max_response_bytes) is not int or max_response_bytes < MIN_RESPONSE_BYTES:
            raise ValueError(f"max_response_bytes must be at least {MIN_RESPONSE_BYTES}")
        self._corpus = corpus
        self._max_bytes = max_response_bytes
        self._include_spans = include_spans
        self._candidate_search = include_spans if candidate_search is None else candidate_search

    def dispatch(self, name: str, arguments: dict, *, delivered_chunks: set[str] | None = None,
                 delivered_spans: dict[tuple[str, str, str], tuple[str, str]] | None = None) -> DispatchResult:
        if not isinstance(name, str):
            return self._error("unknown_tool", "Tool is outside the SEC corpus allowlist.")
        args_schema = _Spans if self._include_spans and name == "get_spans" else _TOOLS.get(name, (None, None))[1]
        if args_schema is None:
            return self._error("unknown_tool", "Tool is outside the SEC corpus allowlist.")
        try:
            validated = args_schema.model_validate(arguments).model_dump(mode="json")
        except ValidationError as exc:
            return self._error("invalid_arguments", validation_message(exc))

        call_args = validated.copy()
        if name == "search_chunks" and validated["filters"] is not None:
            filters = ChunkFilter.model_validate(validated["filters"])
            validated["filters"] = filters.model_dump(mode="json", exclude_none=True)
            call_args["filters"] = filters
        try:
            # Only names in the fixed registry reach attribute lookup.
            if name == "get_spans":
                return self._get_spans(validated, delivered_spans)
            if name == "get_chunk" and self._include_spans and validated["chunk_id"] in (delivered_chunks or ()):
                return DispatchResult(validated, observation_json({"ok": True, "result": {
                    "chunk_id": validated["chunk_id"], "already_delivered": True,
                }}))
            result = getattr(self._corpus, name)(**call_args)
            chunks = self._result_chunks(name, result)
            evidence = tuple(EvidenceRef.from_chunk(chunk) for chunk in chunks)
            discovered = evidence if name == "search_chunks" else ()
            body = {"ok": True, "result": result}
            if self._include_spans and chunks:
                if name == "search_chunks" and self._candidate_search:
                    words = set(re.findall(r"\w+", validated["query"].casefold()))
                    body["result"] = {"items": [self._candidate(chunk, ref, words) for chunk, ref in zip(chunks, evidence)],
                                      "total": result.total, "next_offset": result.next_offset}
                    evidence = ()
                else:
                    if name == "read_chunks" and delivered_chunks:
                        body["result"] = {"items": [
                            {"chunk_id": chunk.chunk_id, "already_delivered": True} if chunk.chunk_id in delivered_chunks else chunk
                            for chunk in chunks
                        ], "total": result.total, "next_offset": result.next_offset}
                        chunks = tuple(chunk for chunk in chunks if chunk.chunk_id not in delivered_chunks)
                        evidence = tuple(ref for ref in evidence if ref.chunk_id not in delivered_chunks)
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

    @staticmethod
    def _candidate(chunk: Chunk, ref: EvidenceRef, words: set[str]) -> dict:
        spans = passage_spans(chunk.text)
        ranked = sorted(spans, key=lambda key: -len(words.intersection(
            re.findall(r"\w+", chunk.text[slice(*spans[key])].casefold())
        )))
        return {"source": ref, "span_ids": ranked[:4]}

    def _get_spans(
        self, arguments: dict, delivered: dict[tuple[str, str, str], tuple[str, str]] | None,
    ) -> DispatchResult:
        chunk = self._corpus.get_chunk(arguments["chunk_id"])
        spans = passage_spans(chunk.text)
        selected = arguments["span_ids"]
        if any(key not in spans for key in selected):
            return self._error("unknown_span", "Use canonical span IDs for this chunk.", arguments)
        ref = EvidenceRef.from_chunk(chunk)
        excerpts = {}
        repeated = {}
        for key in dict.fromkeys(selected):
            exact = chunk.text[slice(*spans[key])]
            previous = (delivered or {}).get((chunk.document_id, chunk.section_id, exact))
            if previous is None:
                excerpts[key] = exact
            else:
                repeated[key] = {"chunk_id": previous[0], "span_id": previous[1]}
        result = {"source": ref, "spans": excerpts} if excerpts else {"chunk_id": chunk.chunk_id}
        if repeated:
            result["already_delivered"] = repeated
        content = observation_json({"ok": True, "result": result,
                                    "span_ids": {chunk.chunk_id: list(excerpts)} if excerpts else {}})
        if len(content.encode("utf-8")) > self._max_bytes:
            return self._error("response_too_large", "Try fewer spans in one request.", arguments)
        return DispatchResult(arguments, content, (ref,) if excerpts else ())

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
