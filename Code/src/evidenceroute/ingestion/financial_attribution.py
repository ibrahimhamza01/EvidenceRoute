"""Resolve verified SEC 10-K internal financial cross-references.

Used by the canonical SEC extraction pipeline.
"""
from __future__ import annotations

import re

from evidenceroute.schemas import Section


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _heading(block: str) -> str:
    """Recognize short standalone headings or headings atop tables."""
    first = block.splitlines()[0].strip() if block else ""
    return _key(first) if len(first) <= 160 else ""


def _first(
    blocks: list[str], start: int, end: int, heading: str
) -> int:
    hits = [
        i for i in range(start, end)
        if _heading(blocks[i]) == heading
    ]
    if not hits:
        raise ValueError(
            f"Financial reference: missing {heading!r} "
            f"in blocks {start}-{end}"
        )
    return hits[0]


def _item_span(boundaries: list, item: str) -> tuple[int, int]:
    for i, boundary in enumerate(boundaries[:-1]):
        if boundary.item == item:
            return (
                boundary.block_index,
                boundaries[i + 1].block_index,
            )
    raise ValueError(
        f"Financial reference: Item {item} lacks an ending boundary"
    )


def _normalize(text: str) -> str:
    lines = [
        re.sub(r"[\t\r\f\v \u00a0]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def _slice(blocks: list[str], start: int, end: int) -> str:
    if start >= end:
        raise ValueError(
            f"Invalid financial reference span {start}:{end}"
        )
    return _normalize("\n".join(blocks[start:end]))


def _append(
    reference: Section, blocks: list[str], start: int, end: int
) -> str:
    return _normalize(
        reference.text + "\n" + _slice(blocks, start, end)
    )


def _check_financial_statements(
    blocks: list[str], start: int, end: int
) -> None:
    keys = [_heading(b) for b in blocks[start:end]]

    families = {
        "income statement": (
            "consolidatedstatementofincome",
            "consolidatedstatementsofincome",
            "statementsofconsolidatedincome",
            "consolidatedstatementsofoperations",
        ),
        "balance sheet": (
            "consolidatedbalancesheet",
            "consolidatedbalancesheets",
        ),
        "cash flow statement": (
            "consolidatedstatementofcashflows",
            "consolidatedstatementsofcashflows",
            "statementsofconsolidatedcashflows",
        ),
        "financial notes": (
            "notestoconsolidatedfinancialstatements",
            "notestotheconsolidatedfinancialstatements",
        ),
    }

    missing = [
        name
        for name, prefixes in families.items()
        if not any(
            key.startswith(prefix)
            for key in keys
            for prefix in prefixes
        )
    ]

    if missing:
        raise ValueError(
            f"Financial reference span lacks {', '.join(missing)}"
        )

    if len(_slice(blocks, start, end)) < 10000:
        raise ValueError(
            "Resolved financial statements suspiciously short"
        )


def _updated(
    sections: list[Section],
    replacements: dict[str, str],
) -> list[Section]:
    for section_id, value in replacements.items():
        if len(value) < 100:
            raise ValueError(
                f"Resolved {section_id} suspiciously short"
            )

    return [
        Section.model_validate({
            **section.model_dump(),
            "text": replacements[section.section_id],
        })
        if section.section_id in replacements
        else section
        for section in sections
    ]


def resolve_special_financial_references(
    blocks: list[str],
    boundaries: list,
    sections: list[Section],
) -> list[Section] | None:
    """Reattribute nonoverlapping financial spans to canonical SEC Items."""

    by_id = {s.section_id: s for s in sections}

    if not {"item_7", "item_8", "item_15"}.issubset(by_id):
        return None

    t7 = by_id["item_7"].text.casefold()
    t8 = by_id["item_8"].text.casefold()

    # Layout A: Item 8 points to statements physically inside Item 15.
    # Observed in NVIDIA FY2022-FY2024.
    if (
        len(t7) > 2000
        and len(t8) < 2000
        and "set forth" in t8
        and "consolidated financial statements" in t8
    ):
        item15_start, item15_end = _item_span(boundaries, "15")

        audit = _first(
            blocks, item15_start + 1, item15_end,
            "reportofindependentregisteredpublicaccountingfirm",
        )
        exhibit = _first(
            blocks, audit + 1, item15_end,
            "exhibitindex",
        )

        _check_financial_statements(blocks, audit, exhibit)

        return _updated(sections, {
            "item_8": _append(
                by_id["item_8"], blocks, audit, exhibit
            ),
            "item_15": _normalize(
                _slice(blocks, item15_start, audit)
                + "\n"
                + _slice(blocks, exhibit, item15_end)
            ),
        })

    # Layout B: Item 7/8 point to the Financial Table of Contents.
    # Observed in Chevron FY2022-FY2024.
    if (
        len(t7) < 2000
        and len(t8) < 2000
        and "financial table of contents" in t7
        and "financial table of contents" in t8
    ):
        item14_start, _ = _item_span(boundaries, "14")
        item15_start, _ = _item_span(boundaries, "15")

        md = _first(
            blocks, item14_start + 1, item15_start,
            "managementsdiscussionandanalysisoffinancialconditionandresultsofoperations",
        )
        audit = _first(
            blocks, md + 1, item15_start,
            "reportofindependentregisteredpublicaccountingfirm",
        )

        _check_financial_statements(
            blocks, audit, item15_start
        )

        if len(_slice(blocks, md, audit)) < 10000:
            raise ValueError(
                "Resolved MD&A suspiciously short"
            )

        return _updated(sections, {
            "item_7": _append(
                by_id["item_7"], blocks, md, audit
            ),
            "item_8": _append(
                by_id["item_8"], blocks, audit, item15_start
            ),
        })

    # Layout C: Items 7/8 refer to page ranges inside a report
    # following the original Item 15 exhibit list.
    # Observed in JPMorgan FY2022-FY2024.
    if (
        len(t7) < 2000
        and len(t8) < 2000
        and "appears on pages" in t7
        and "appear on pages" in t8
    ):
        item15_start, item15_end = _item_span(boundaries, "15")

        md = _first(
            blocks, item15_start + 1, item15_end,
            "managementsdiscussionandanalysis",
        )
        audit = _first(
            blocks, md + 1, item15_end,
            "reportofindependentregisteredpublicaccountingfirm",
        )

        _check_financial_statements(
            blocks, audit, item15_end
        )

        if len(_slice(blocks, md, audit)) < 10000:
            raise ValueError(
                "Resolved MD&A suspiciously short"
            )

        return _updated(sections, {
            "item_7": _append(
                by_id["item_7"], blocks, md, audit
            ),
            "item_8": _append(
                by_id["item_8"], blocks, audit, item15_end
            ),
            "item_15": _slice(
                blocks, item15_start, md
            ),
        })

    return None
