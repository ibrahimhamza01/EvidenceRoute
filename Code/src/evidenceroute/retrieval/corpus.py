"""Validated, deterministically ordered collections of ``Chunk`` records.

Retrieval consumes the agreed ``Chunk`` schema only. Loaders accept JSONL now and
the processed Parquet export later, without depending on the HTML parser.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from evidenceroute.schemas import Chunk


class CorpusError(ValueError):
    """Chunks cannot form a consistent retrieval corpus."""


def _canonical(chunk: Chunk) -> str:
    return json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


class ChunkCorpus:
    """Immutable chunk collection sorted by ``chunk_id``.

    Exact duplicate records are collapsed. Two different records sharing a
    ``chunk_id`` are rejected, because results must point at one original chunk.
    """

    def __init__(self, chunks: Iterable[Chunk | Mapping[str, Any]]):
        by_id: dict[str, Chunk] = {}
        for raw in chunks:
            chunk = raw if isinstance(raw, Chunk) else Chunk.model_validate(raw)
            chunk = Chunk.model_validate(chunk.model_dump())  # re-validate copies
            if chunk.char_end < chunk.char_start:
                raise CorpusError(f"char_end < char_start for chunk {chunk.chunk_id}")
            existing = by_id.get(chunk.chunk_id)
            if existing is not None and existing != chunk:
                raise CorpusError(f"Conflicting records share chunk_id {chunk.chunk_id}")
            by_id[chunk.chunk_id] = chunk
        self._chunks: tuple[Chunk, ...] = tuple(by_id[k] for k in sorted(by_id))
        self._by_id = {c.chunk_id: c for c in self._chunks}

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self._chunks)

    def __getitem__(self, index: int) -> Chunk:
        return self._chunks[index]

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._chunks

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def fingerprint(self) -> str:
        """SHA-256 over canonical records; identical inputs give identical fingerprints."""
        digest = hashlib.sha256()
        for chunk in self._chunks:
            digest.update(_canonical(chunk).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()


def load_chunks_jsonl(path: Path | str) -> ChunkCorpus:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(Chunk.model_validate_json(line))
            except ValueError as exc:
                raise CorpusError(f"{path}:{line_no}: invalid Chunk record: {exc}") from exc
    return ChunkCorpus(records)


def load_chunks_parquet(path: Path | str) -> ChunkCorpus:
    """Load the processing pipeline's ``chunks.parquet`` (requires pyarrow)."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    return ChunkCorpus(table.to_pylist())


def load_chunks(path: Path | str) -> ChunkCorpus:
    suffix = Path(path).suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return load_chunks_jsonl(path)
    if suffix == ".parquet":
        return load_chunks_parquet(path)
    raise CorpusError(f"Unsupported chunk file type: {path}")


def as_corpus(chunks: ChunkCorpus | Iterable[Chunk | Mapping[str, Any]]) -> ChunkCorpus:
    return chunks if isinstance(chunks, ChunkCorpus) else ChunkCorpus(chunks)
