"""Stable passage IDs based on exact character boundaries inside a canonical chunk."""

import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
_MAX_SPAN = 360


def passage_spans(text: str) -> dict[str, tuple[int, int]]:
    """Split on sentence-like boundaries, keeping source characters unchanged."""
    spans = {}
    start = 0
    boundaries = [(match.start(), match.end()) for match in _BOUNDARY.finditer(text)]
    for end, following in [*boundaries, (len(text), len(text))]:
        while start < end:
            if text[start].isspace():
                start += 1
                continue
            stop = min(end, start + _MAX_SPAN)
            if stop < end:
                space = text.rfind(" ", start, stop)
                if space > start:
                    stop = space
            trimmed = start + len(text[start:stop].rstrip())
            spans[f"{start}:{trimmed}"] = (start, trimmed)
            start = stop
        start = following
    return spans
