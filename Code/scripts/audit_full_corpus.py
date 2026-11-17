"""Audit the 75-filing EvidenceRoute full SEC corpus."""

from collections import Counter
from pathlib import Path
import csv

from evidenceroute.ingestion.downloader import (
    load_manifest,
    relative_filing_path,
)
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.ingestion.processor import chunk_sections


def main():
    manifest = Path("data/full_filings_manifest.csv")
    output = Path("data/processed/full_parsing_audit_v3.csv")

    filings = load_manifest(manifest)
    assert len(filings) == 75, (
        f"Expected 75 filings, found {len(filings)}"
    )

    required = {
        "item_1", "item_1a", "item_7", "item_8", "item_15"
    }

    records = []
    all_chunk_ids = set()

    for number, filing in enumerate(filings, 1):
        row = {
            "ticker": filing.ticker,
            "fiscal_year": filing.fiscal_year,
            "document_id": filing.document_id,
            "status": "FAIL",
            "sections": 0,
            "chunks": 0,
            "item_1c_present": False,
            "item_7_chars": 0,
            "item_8_chars": 0,
            "item_8_table_lines": 0,
            "warnings": "",
            "error": "",
        }

        try:
            path = (
                Path("data/raw") /
                relative_filing_path(filing)
            )

            sections = extract_sections(
                path,
                filing.document_id,
                filing.source_url,
                include_item_1c=True,
                resolve_financial_references=True,
            )

            by_id = {
                section.section_id: section
                for section in sections
            }

            assert len(by_id) == len(sections)
            assert required <= set(by_id)

            for section in sections:
                assert section.document_id == filing.document_id
                assert section.source_url == filing.source_url

            chunks = chunk_sections(sections, filing)
            current_ids = set()

            for chunk in chunks:
                assert chunk.chunk_id not in current_ids
                assert chunk.chunk_id not in all_chunk_ids
                assert chunk.section_id in by_id

                section = by_id[chunk.section_id]

                assert (
                    section.text[
                        chunk.char_start:chunk.char_end
                    ] == chunk.text
                )

                current_ids.add(chunk.chunk_id)

            # Count IDs only after this filing has passed validation.
            all_chunk_ids.update(current_ids)

            item7 = by_id["item_7"].text
            item8 = by_id["item_8"].text

            table_lines = sum(
                " | " in line
                for line in item8.splitlines()
            )

            warnings = []

            if len(item7) < 2000:
                warnings.append("Short Item 7")

            if len(item8) < 2000:
                warnings.append("Short Item 8")

            if table_lines == 0:
                warnings.append("No Item 8 table-like lines")

            row.update({
                "status": "REVIEW" if warnings else "PASS",
                "sections": len(sections),
                "chunks": len(chunks),
                "item_1c_present": "item_1c" in by_id,
                "item_7_chars": len(item7),
                "item_8_chars": len(item8),
                "item_8_table_lines": table_lines,
                "warnings": "; ".join(warnings),
            })

        except Exception as exc:
            row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        records.append(row)

        print(
            f"[{number:02d}/{len(filings)}] "
            f"{filing.ticker:5s} FY{filing.fiscal_year} "
            f"{row['status']}",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(
        row["status"] for row in records
    )

    print("\n" + "=" * 55)
    print("FULL SEC CORPUS AUDIT — V3")
    print("=" * 55)
    print(f"Filings:       {len(records)}")
    print(f"PASS:          {counts['PASS']}")
    print(f"REVIEW:        {counts['REVIEW']}")
    print(f"FAIL:          {counts['FAIL']}")
    print(f"Unique chunks: {len(all_chunk_ids):,}")
    print(
        "Item 1C:      "
        f"{sum(row['item_1c_present'] for row in records)}/75"
    )
    print(f"Saved:         {output}")

    print("\nFILINGS REQUIRING ATTENTION:")
    outstanding = [
        row for row in records
        if row["status"] != "PASS"
    ]

    if not outstanding:
        print("None")

    for row in outstanding:
        print(
            f"{row['ticker']} FY{row['fiscal_year']}: "
            f"{row['status']} — "
            f"{row['warnings'] or row['error']}"
        )

    if outstanding:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
