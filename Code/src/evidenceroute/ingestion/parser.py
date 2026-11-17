"""SEC 10-K HTML -> canonical Section records.

Canonical parser: structural block extraction, conservative heading recognition,
and explicit failure for missing or ambiguous item boundaries.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from evidenceroute.schemas import Section

TARGET_ITEMS = ("1", "1A", "1C", "7", "8", "15")
ITEM_ORDER = (
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A",
    "8", "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
)
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1C": "Cybersecurity",
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
    """Recognize SEC headings despite table cells and split words."""
    if len(text) > 220:
        return None

    # Some 10-Ks place the Item 7 or Item 8 heading across
    # multiple lines, ending in "Table of Contents".
    # Only flatten this verified structural pattern.
    if "\n" in text:
        lines = [
            clean_line(line)
            for line in text.splitlines()
            if clean_line(line)
        ]

        is_financial_contents_heading = (
            len(lines) in (2, 3)
            and lines[-1].casefold() == "table of contents"
            and re.match(r"^item\s+(?:7|8)\.", lines[0], re.I)
            is not None
        )

        if not is_financial_contents_heading:
            return None

        text = " ".join(lines)

    # Some 10-Ks combine Items 1 and 2 into one body heading.
    # Normalize punctuation and whitespace before checking.
    combined = re.sub(r"[^a-z0-9]", "", text.casefold())

    if combined == "items1and2businessandproperties":
        return "1"

    match = ITEM_RE.match(text)
    if not match:
        return None

    item = match.group(1) + (match.group(2) or "").upper()
    if item not in HEADING_TITLES:
        return None

    title = clean_line(match.group(3))

    # Reject common table-of-contents page-number suffixes.
    if re.search(r"\s+\d{1,3}\s*$", title):
        return None

    # Compare only letters/digits. This handles:
    # B USINESS, FINANC IAL, and table-cell separators (|).
    def signature(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    normalized = signature(title)
    prefixes = tuple(signature(t) for t in HEADING_TITLES[item])

    # Some reports prefix Item 7's title with the registrant name
    # and append "Table of Contents" to the actual body heading.
    if item == "7":
        md_title = signature("management's discussion and analysis")
        position = normalized.find(md_title)
        if (
            0 < position <= 80
            and normalized[:position].endswith("andsubsidiaries")
            and normalized.endswith("tableofcontents")
        ):
            return item

    # An Item 8 body may begin with a financial-section contents page.
    if item == "8" and any(
        normalized == prefix + "tableofcontents"
        for prefix in prefixes
    ):
        return item

    if not any(normalized.startswith(prefix) for prefix in prefixes):
        return None

    strict_items = {
        "1", "1A", "1B", "1C", "2", "3", "4",
        "6", "8", "9A", "9B", "11", "14", "16",
    }
    if item in strict_items and normalized not in prefixes:
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
        if (
            {"1", "1A", "7", "7A", "8"}.issubset(found)
            and bool({"9", "9A"} & found)
        ):
            runs.append(run)
    if not runs:
        raise ParsingError("Could not identify an ordered SEC Item 1–9 body heading sequence")
    # Prefer most complete run, then latest occurrence (TOC normally precedes body).
    run = max(runs, key=lambda r: (len(r), r[0].block_index))
    found = {b.item for b in run}
    if len(found) != len(run):
        raise ParsingError("Duplicate SEC Item headings in selected body sequence")
    return run




def _standalone_heading_key(text: str) -> str | None:
    """Normalize short standalone headings, excluding tables and paragraphs."""
    if len(text) > 160 or "\n" in text or "|" in text:
        return None
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _short_financial_reference(text: str) -> bool:
    """Identify short Item 7/8 sections that defer to another section."""
    return (
        len(text) < 2000
        and re.search(r"\b(?:see|reference is made)\b", text, re.I) is not None
        and ("financial" in text.casefold() or "management" in text.casefold())
    )


def _resolve_referenced_financial_sections(
    blocks: list[str],
    boundaries: list[Boundary],
    sections: list[Section],
) -> list[Section]:
    """Recover substantive Items 7/8 from a financial section after Item 16.

    Only resolves filings where BOTH Items 7 and 8 contain short references.
    Otherwise preserves normal extraction or fails for unsupported layouts.
    """
    from evidenceroute.ingestion.financial_attribution import (
        resolve_special_financial_references,
    )

    resolved = resolve_special_financial_references(
        blocks, boundaries, sections
    )
    if resolved is not None:
        return resolved

    by_id = {section.section_id: section for section in sections}
    needs_7 = _short_financial_reference(by_id["item_7"].text)
    needs_8 = _short_financial_reference(by_id["item_8"].text)

    if not needs_7 and not needs_8:
        return sections

    if needs_7 != needs_8:
        raise ParsingError(
            "Only one of Items 7/8 is a short reference; manual review required"
        )

    item_16 = next((b for b in boundaries if b.item == "16"), None)
    if item_16 is None:
        raise ParsingError(
            "Cannot resolve referenced financial section without Item 16"
        )

    first = item_16.block_index + 1
    keys = {
        i: _standalone_heading_key(blocks[i])
        for i in range(first, len(blocks))
    }

    md_titles = {
        "managementsdiscussionandanalysis",
        "managementsdiscussionandanalysisoffinancialconditionandresultsofoperations",
    }

    financial_starts = {
        "managementsreportoninternalcontroloverfinancialreporting",
        "reportofindependentregisteredpublicaccountingfirm",
        "financialstatements",
        "consolidatedstatementofincome",
        "statementsofconsolidatedincome",
    }

    md_start = next(
        (i for i in range(first, len(blocks)) if keys[i] in md_titles),
        None,
    )
    if md_start is None:
        raise ParsingError("Substantive MD&A heading not found after Item 16")

    financial_start = next(
        (
            i for i in range(md_start + 21, len(blocks))
            if keys[i] in financial_starts
        ),
        None,
    )
    if financial_start is None:
        raise ParsingError("Financial statement start not found after MD&A")

    # An explicit exhibit/signature section ends the financial material.
    # Otherwise the remaining tail is provisionally retained for manual QA.
    ending_titles = {"indextoexhibits", "exhibitindex", "signatures"}
    financial_end = next(
        (
            i for i in range(financial_start + 1, len(blocks))
            if keys[i] in ending_titles
        ),
        len(blocks),
    )

    # Do not accept a financial section without all three core statements.
    financial_keys = {
        keys[i] for i in range(financial_start, financial_end)
    }
    required = {
        "income statement": {
            "consolidatedstatementofincome",
            "statementsofconsolidatedincome",
        },
        "balance sheet": {
            "consolidatedbalancesheet",
            "consolidatedbalancesheets",
        },
        "cash flow statement": {
            "consolidatedstatementofcashflows",
            "statementsofconsolidatedcashflows",
        },
    }
    missing = [
        label for label, headings in required.items()
        if not (financial_keys & headings)
    ]
    if missing:
        raise ParsingError(
            f"Referenced financial section missing statements: {missing}"
        )

    spans = {
        "item_7": (md_start, financial_start),
        "item_8": (financial_start, financial_end),
    }
    resolved = []

    for section in sections:
        if section.section_id not in spans:
            resolved.append(section)
            continue

        start, end = spans[section.section_id]

        # Retain the original numbered Item heading, but replace the
        # cross-reference paragraph with actual text from the same filing.
        item_heading = section.text.splitlines()[0]
        text = normalize_text(
            item_heading + "\n" + "\n".join(blocks[start:end])
        )

        if len(text) < 10000:
            raise ParsingError(
                f"Resolved {section.section_id} suspiciously short"
            )

        resolved.append(
            Section.model_validate({
                **section.model_dump(),
                "text": text,
            })
        )

    return resolved


def extract_sections(html_path: Path, document_id: str, source_url: str) -> list[Section]:
    """Extract canonical SEC sections, including available Item 1C. Fail closed if any target is empty or misbounded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(Path(html_path).read_bytes(), "lxml")
    root = soup.body or soup
    for tag in list(root.find_all(["script", "style", "noscript", "svg", "ix:header", "nav"])):
        tag.decompose()
    blocks = _blocks(root)

    from evidenceroute.ingestion.topical_parser import (
        is_topical_form10k,
        extract_topical_sections,
    )

    if is_topical_form10k(blocks):
        return extract_topical_sections(
            blocks, document_id, source_url,
        )

    boundaries = detect_item_boundaries(blocks)

    # Some filings have Item 15 but no Item 16.
    # Stop before the standalone SIGNATURES section.
    if boundaries[-1].item == "15":
        start_index = boundaries[-1].block_index
        signatures = [
            i for i in range(start_index + 1, len(blocks))
            if re.fullmatch(r"signatures?", blocks[i].strip(), re.I)
        ]
        if len(signatures) != 1:
            raise ParsingError(
                "Cannot safely terminate Item 15: "
                f"found {len(signatures)} SIGNATURES headings"
            )
        end_index = signatures[0]
        boundaries.append(
            Boundary("END", end_index, blocks[end_index])
        )

    by_item = {b.item: i for i, b in enumerate(boundaries)}
    sections: list[Section] = []
    for item in TARGET_ITEMS:
        if item == "1C" and item not in by_item:
            continue
        if item not in by_item or by_item[item] + 1 >= len(boundaries):
            raise ParsingError(f"Missing start/end boundary for Item {item}")
        start = boundaries[by_item[item]]
        end = boundaries[by_item[item] + 1]
        if end.block_index <= start.block_index:
            raise ParsingError(f"Invalid Item {item} boundaries")
        text = normalize_text("\n".join(blocks[start.block_index:end.block_index]))
        short_but_valid = (
            item == "1C"
            and any(
                clean_line(line).lower().rstrip(".") == "not applicable"
                for line in text.splitlines()[1:]
            )
        )
        if len(text) < 100 and not short_but_valid:
            raise ParsingError(f"Item {item} suspiciously short ({len(text)} chars)")
        sections.append(Section(document_id=document_id, section_id=f"item_{item.lower()}",
                                section_name=ITEM_NAMES[item], text=text, source_url=source_url))
    sections = _resolve_referenced_financial_sections(
        blocks, boundaries, sections
    )
    return sections
