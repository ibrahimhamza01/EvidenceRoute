"""SEC 10-K HTML -> canonical Section records.

Pilot parser: structural block extraction, conservative heading recognition,
and explicit failure for missing or ambiguous item boundaries.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from evidenceroute.schemas import Section

TARGET_ITEMS = ("1", "1A", "7", "8", "15")
ITEM_ORDER = (
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A",
    "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
)
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "7": "Management's Discussion and Analysis",
    "8": "Financial Statements and Supplementary Data",
    "15": "Exhibits and Financial Statement Schedules",
}
# Full heading titles are essential: "Item 1A. Environmental Regulations" is not a boundary.
HEADING_TITLES = {
    "1": ("business",),
    "1A": ("risk factors",),
    "1B": ("unresolved staff comments",),
    "1C": ("cybersecurity",),
    "2": ("properties",),
    "3": ("legal proceedings",),
    "4": ("mine safety disclosures",),
    "5": ("market for registrant",),
    "6": ("reserved",),
    "7": ("management's discussion and analysis", "management’s discussion and analysis"),
    "7A": ("quantitative and qualitative",),
    "8": ("financial statements and supplementary data",),
    "9": ("changes in and disagreements with accountants",),
    "9A": ("controls and procedures",),
    "9B": ("other information",),
    "9C": ("disclosure regarding foreign jurisdictions",),
    "10": ("directors, executive officers and corporate governance",),
    "11": ("executive compensation",),
    "12": ("security ownership",),
    "13": ("certain relationships and related transactions",),
    "14": ("principal accountant fees and services",),
    "15": ("exhibit",),
    "16": ("form 10-k summary",),
}
ITEM_RE = re.compile(r"^item\s+(1[0-6]|[1-9])([abc])?\s*[.:\-–—]?\s*(.*)$", re.I)
WHITESPACE = re.compile(r"[\t\r\f\v \u00a0]+")


class ParsingError(ValueError):
    """Input cannot be parsed without risking incorrect evidence attribution."""


def clean_line(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        # Avoid duplicating text in nested tables.
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        values = [clean_line(c.get_text(" ", strip=True)) for c in cells]
        if any(values):
            rows.append(" | ".join(values))
    return "\n".join(rows)


def _blocks(root: Tag) -> list[str]:
    """Flatten terminal text blocks and tables in DOM order without duplicate ancestors."""
    out: list[str] = []

    def visit(node: Tag) -> None:
        if node.name in {"script", "style", "noscript", "svg", "ix:header", "header", "nav"}:
            return
        if node.name == "table":
            value = _table_text(node)
            if value:
                out.append(value)
            return
        if node.name in {"div", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            nested = node.find(["div", "p", "table", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
            if nested is None:
                value = clean_line(node.get_text(" ", strip=True))
                if value:
                    out.append(value)
                return
        children = [c for c in node.children if isinstance(c, Tag)]
        if children:
            for child in children:
                visit(child)
        else:
            value = clean_line(node.get_text(" ", strip=True))
            if value:
                out.append(value)

    visit(root)
    return out


def _item_heading(text: str) -> str | None:
    # A heading is a short standalone block; no TOC page number or paragraph.
    if len(text) > 220 or "\n" in text or " | " in text:
        return None
    match = ITEM_RE.match(text)
    if not match:
        return None
    item = match.group(1) + (match.group(2) or "").upper()
    title = clean_line(match.group(3)).lower().strip(" .:–—\"'“”")
    # Normalize odd SEC inline spacing and apostrophes.
    title = title.replace("’", "'")
    if item not in HEADING_TITLES:
        return None
    if re.search(r"\s+\d{1,3}$", title):  # table-of-contents page number
        return None
    prefixes = tuple(t.replace("’", "'") for t in HEADING_TITLES[item])
    if not any(title.startswith(p) for p in prefixes):
        return None
    # A title is not a paragraph. Only permit modest continuation of long headings.
    if item in {"1", "1A", "1B", "1C", "2", "3", "4", "6", "8", "9A", "9B", "11", "14", "16"}:
        if not any(title == p for p in prefixes):
            return None
    return item


@dataclass(frozen=True)
class Boundary:
    item: str
    block_index: int
    heading: str


def detect_item_boundaries(blocks: list[str]) -> list[Boundary]:
    """Select ordered body headings; reject ambiguous duplicate candidates."""
    candidates = [Boundary(item, i, text) for i, text in enumerate(blocks)
                  if (item := _item_heading(text)) is not None]
    # TOC may include title-only headings. Choose a full monotonic run beginning
    # at Item 1 and continuing through Item 9; require target and successors.
    starts = [i for i, b in enumerate(candidates) if b.item == "1"]
    runs: list[list[Boundary]] = []
    for start in starts:
        run = [candidates[start]]
        current_order = 0
        for candidate in candidates[start + 1:]:
            pos = ITEM_ORDER.index(candidate.item)
            if pos <= current_order:
                break
            run.append(candidate)
            current_order = pos
            if candidate.item == "16":
                break
        found = {b.item for b in run}
        if {"1", "1A", "1B", "7", "7A", "8", "9"}.issubset(found):
            runs.append(run)
    if not runs:
        raise ParsingError("Could not identify an ordered SEC Item 1–9 body heading sequence")
    # Prefer most complete run, then latest occurrence (TOC normally precedes body).
    run = max(runs, key=lambda r: (len(r), r[0].block_index))
    found = {b.item for b in run}
    if len(found) != len(run):
        raise ParsingError("Duplicate SEC Item headings in selected body sequence")
    return run


def extract_sections(html_path: Path, document_id: str, source_url: str) -> list[Section]:
    """Return four pilot Sections. Fail closed if any target is empty or misbounded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(Path(html_path).read_bytes(), "lxml")
    root = soup.body or soup
    for tag in list(root.find_all(["script", "style", "noscript", "svg", "ix:header", "nav"])):
        tag.decompose()
    blocks = _blocks(root)
    boundaries = detect_item_boundaries(blocks)
    by_item = {b.item: i for i, b in enumerate(boundaries)}
    sections: list[Section] = []
    for item in TARGET_ITEMS:
        if item not in by_item or by_item[item] + 1 >= len(boundaries):
            raise ParsingError(f"Missing start/end boundary for Item {item}")
        start = boundaries[by_item[item]]
        end = boundaries[by_item[item] + 1]
        if end.block_index <= start.block_index:
            raise ParsingError(f"Invalid Item {item} boundaries")
        text = normalize_text("\n".join(blocks[start.block_index:end.block_index]))
        if len(text) < 100:
            raise ParsingError(f"Item {item} suspiciously short ({len(text)} chars)")
        sections.append(Section(document_id=document_id, section_id=f"item_{item.lower()}",
                                section_name=ITEM_NAMES[item], text=text, source_url=source_url))
    return sections
