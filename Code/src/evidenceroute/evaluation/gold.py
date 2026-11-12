"""Locate gold evidence quotes in processed SEC sections and find the chunks that contain them."""

from __future__ import annotations

import re

# Characters treated as equivalent when matching quotes to section text.
# "|" is removed because the parser separates table cells with it.
_REPLACEMENTS = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "™": "", "\xa0": " ", "|": " ",
}


def normalize(text: str) -> str:
    """Normalize a quote the same way section text is normalized."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize text and remember, for each output character, its position in the original text."""
    chars: list[str] = []
    offsets: list[int] = []
    previous_was_space = True  # also drops leading whitespace
    for position, char in enumerate(text):
        char = _REPLACEMENTS.get(char, char)
        if char == "":
            continue
        if char.isspace():
            if previous_was_space:
                continue
            char = " "
            previous_was_space = True
        else:
            previous_was_space = False
        chars.append(char)
        offsets.append(position)
    return "".join(chars), offsets


def locate_quote(section_text: str, quote: str) -> tuple[int, int] | None:
    """Return (char_start, char_end) of the quote in the original section text, or None."""
    normalized_text, offsets = normalize_with_offsets(section_text)
    normalized_quote = normalize(quote)
    if not normalized_quote:
        return None
    found = normalized_text.find(normalized_quote)
    if found < 0:
        return None
    start = offsets[found]
    end = offsets[found + len(normalized_quote) - 1] + 1
    return start, end


def containing_chunks(chunks: list[dict], document_id: str, section_id: str, start: int, end: int) -> list[str]:
    """IDs of chunks in the same section that fully contain the character range [start, end)."""
    return [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["document_id"] == document_id
        and chunk["section_id"] == section_id
        and chunk["char_start"] <= start
        and chunk["char_end"] >= end
    ]
