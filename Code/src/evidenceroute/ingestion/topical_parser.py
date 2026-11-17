"""Primary-section extraction for topic-organized SEC 10-K filings.

The SEC cross-reference index may map Items to noncontiguous pages.
These sections capture primary topical ranges, not every indexed page.
"""
from __future__ import annotations

import re

from evidenceroute.schemas import Section


SECTION_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1C": "Cybersecurity",
    "7": "Management's Discussion and Analysis",
    "8": "Financial Statements and Supplementary Data",
    "15": "Exhibits and Financial Statement Schedules",
}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _short_key(text: str) -> str:
    return _key(text) if len(text) <= 160 else ""


def is_topical_form10k(blocks: list[str]) -> bool:
    """Recognize the verified topical-heading/cross-reference layout."""
    labels = {_short_key(block) for block in blocks}
    return {
        "form10kcrossreferenceindex",
        "managementsdiscussionandanalysis",
        "financialstatementsandsupplementaldetails",
    }.issubset(labels)


def _unique_heading(blocks: list[str], label: str) -> int:
    matches = [
        i for i, text in enumerate(blocks)
        if _short_key(text) == label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Topical heading {label!r}: expected 1, found {len(matches)}"
        )
    return matches[0]


def _normalize(text: str) -> str:
    lines = [
        re.sub(r"[\t\r\f\v \u00a0]+", " ", line).strip()
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def extract_topical_sections(
    blocks: list[str],
    document_id: str,
    source_url: str,
) -> list[Section]:
    """Extract verified primary spans without overlapping their text."""
    if not is_topical_form10k(blocks):
        raise ValueError("Not a recognized topical Form 10-K")

    md = _unique_heading(blocks, "managementsdiscussionandanalysis")
    risks = _unique_heading(blocks, "riskfactors")
    finance = _unique_heading(
        blocks, "financialstatementsandsupplementaldetails"
    )
    exhibits = _unique_heading(blocks, "exhibits")
    index = _unique_heading(blocks, "form10kcrossreferenceindex")

    # Business headings can contain page-header decorations.
    business_candidates = [
        i for i, text in enumerate(blocks[:md])
        if len(text) <= 160
        and (
            _short_key(text).startswith("fundamentalsofourbusiness")
            or _short_key(text).startswith("fundamentalofourbusiness")
        )
    ]
    if not business_candidates:
        raise ValueError("Primary Business heading not found")
    business = business_candidates[0]

    cybers = [
        i for i, text in enumerate(blocks)
        if _short_key(text) == "cybersecurity"
    ]
    if len(cybers) > 1:
        raise ValueError(
            f"Ambiguous cybersecurity headings: {cybers}"
        )
    cyber = cybers[0] if cybers else None

    anchors = [business, md, risks]
    if cyber is not None:
        anchors.append(cyber)
    anchors += [finance, exhibits, index]

    if anchors != sorted(set(anchors)):
        raise ValueError(f"Unordered topical headings: {anchors}")

    # Confirm the end-of-document SEC cross-reference index is real.
    if index + 1 >= len(blocks) or not all(
        term in blocks[index + 1].casefold()
        for term in ("item 1.", "item 1a.", "item 7.", "item 8.")
    ):
        raise ValueError("SEC cross-reference index incomplete")

    # Reject financial sections missing core statements.
    financial_keys = {
        _short_key(b) for b in blocks[finance:exhibits]
    }

    income = {
        "consolidatedstatementsofincome",
        "consolidatedstatementsofoperations",
    }
    required = {
        "consolidatedbalancesheets",
        "consolidatedstatementsofcashflows",
    }

    if not (financial_keys & income) or not required.issubset(
        financial_keys
    ):
        raise ValueError(
            "Financial section missing core statement headings"
        )

    spans = [
        ("1", business, md),
        ("1A", risks, cyber if cyber is not None else finance),
    ]

    if cyber is not None:
        spans.append(("1C", cyber, finance))

    spans.extend([
        ("7", md, risks),
        ("8", finance, exhibits),
        ("15", exhibits, index),
    ])

    sections = []

    for item, start, end in spans:
        text = _normalize("\n".join(blocks[start:end]))

        minimum = 300 if item == "15" else 2000
        if len(text) < minimum:
            raise ValueError(
                f"Topical Item {item} suspiciously short: {len(text)}"
            )

        sections.append(
            Section(
                document_id=document_id,
                section_id=f"item_{item.lower()}",
                section_name=SECTION_NAMES[item],
                text=text,
                source_url=source_url,
            )
        )

    return sections
