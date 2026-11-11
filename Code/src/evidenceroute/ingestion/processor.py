"""Deterministic, character-offset-safe SEC section chunking and Parquet export."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from evidenceroute.schemas import Chunk, Filing, Section


def chunk_sections(sections: list[Section], filing: Filing, *, max_chars: int = 1800,
                   overlap_chars: int = 200) -> list[Chunk]:
    if max_chars <= 0 or not 0 <= overlap_chars < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap_chars < max_chars")
    chunks: list[Chunk] = []
    for section in sections:
        if section.document_id != filing.document_id:
            raise ValueError("Section and filing document_id differ")
        text = section.text
        start = 0
        index = 0
        while start < len(text):
            limit = min(start + max_chars, len(text))
            end = limit
            if limit < len(text):
                # Prefer a paragraph boundary; otherwise a word boundary.
                cut = text.rfind("\n", start + max_chars // 2, limit)
                if cut < 0:
                    cut = text.rfind(" ", start + max_chars // 2, limit)
                if cut > start:
                    end = cut + (1 if text[cut] == "\n" else 0)
            if end <= start:
                raise ValueError("Chunking failed to advance")
            fragment = text[start:end]
            key = f"{filing.document_id}|{section.section_id}|{index}|{start}|{end}"
            chunk_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
            chunk = Chunk(chunk_id=chunk_id, document_id=filing.document_id,
                          company_name=filing.company_name, ticker=filing.ticker,
                          fiscal_year=filing.fiscal_year, section_id=section.section_id,
                          section_name=section.section_name, chunk_index=index,
                          text=fragment, char_start=start, char_end=end,
                          source_url=section.source_url)
            assert text[chunk.char_start:chunk.char_end] == chunk.text
            chunks.append(chunk)
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
            index += 1
    return chunks


def write_parquet(sections: list[Section], chunks: list[Chunk], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sections or not chunks:
        raise ValueError("Refusing to export empty sections or chunks")
    for name, records in (("sections", sections), ("chunks", chunks)):
        rows = [record.model_dump(mode="json") for record in records]
        # Explicit sorted columns and stable record order for reproducibility.
        columns = sorted(rows[0])
        table = pa.table({column: [row[column] for row in rows] for column in columns})
        dest = output_dir / f"{name}.parquet"
        pq.write_table(table, dest, compression="zstd")
        # Verify round-trip readability and row count.
        if pq.read_metadata(dest).num_rows != len(rows):
            raise ValueError(f"Parquet round-trip row count mismatch: {dest}")
