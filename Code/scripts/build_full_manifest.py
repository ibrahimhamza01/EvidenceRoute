"""Build a frozen FY2022-FY2024 Form 10-K manifest from official SEC submissions.

Run from the repo root with PYTHONPATH=Code/src and SEC_USER_AGENT set.
No filing HTML is downloaded by this script.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

from evidenceroute.ingestion.downloader import MANIFEST_COLUMNS, load_manifest, validate_filing
from evidenceroute.schemas import Filing

ROOT = Path("data")
SELECTION = ROOT / "companies.csv"
PILOT = ROOT / "filings_manifest.csv"
OUTPUT = ROOT / "full_filings_manifest.csv"


class SECClient:
    def __init__(self, user_agent: str):
        if "@" not in user_agent or "example." in user_agent.lower():
            raise ValueError("Set SEC_USER_AGENT to your real project name and contact email")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.last_request = 0.0

    def json(self, url: str) -> dict:
        for attempt in range(4):
            time.sleep(max(0.0, 0.5 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, timeout=(10, 40))
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == 3:
                        response.raise_for_status()
                    time.sleep(min(2 ** (attempt + 1), 20))
                    continue
                if response.status_code == 403:
                    raise RuntimeError(f"SEC returned HTTP 403 for {url}; check User-Agent and access policy")
                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** (attempt + 1), 20))
        raise RuntimeError(f"Exhausted retries: {url}")


def fiscal_year(report_date: str) -> int:
    """January 1-7 year ends can belong to the prior labeled fiscal year.

    Verify unusual boundaries against the filing cover page during QA.
    """
    d = date.fromisoformat(report_date)
    return d.year - 1 if d.month == 1 and d.day <= 7 else d.year


def submissions_rows(columns: dict):
    accessions = columns.get("accessionNumber", [])
    for idx, accession in enumerate(accessions):
        if columns.get("form", [])[idx] != "10-K":
            continue
        report_date = columns.get("reportDate", [])[idx]
        if not report_date:
            continue
        yield {
            "accession_number": accession,
            "report_period_end": report_date,
            "filing_date": columns["filingDate"][idx],
            "primary_document": columns["primaryDocument"][idx],
            "fiscal_year": fiscal_year(report_date),
        }


def read_selection() -> list[tuple[str, str, str, tuple[int, ...]]]:
    """Read frozen company identifiers and fiscal years."""
    with SELECTION.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"ticker", "company_name", "cik", "fiscal_years"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing company columns: {required}")

        companies = []
        for row in reader:
            ticker = row["ticker"].strip().upper()
            name = row["company_name"].strip()
            cik = row["cik"].strip()
            years = tuple(
                int(year.strip())
                for year in row["fiscal_years"].split(";")
            )

            if (
                not ticker or not name
                or len(cik) != 10 or not cik.isdigit()
                or len(years) != 3 or len(set(years)) != 3
            ):
                raise ValueError(f"Invalid company selection: {row}")

            companies.append((ticker, cik, name, years))

    if len(companies) != 25 or len({c[0] for c in companies}) != 25:
        raise ValueError("Expected 25 distinct companies")

    return companies


def main() -> int:
    sec = SECClient(os.getenv("SEC_USER_AGENT", ""))
    companies = read_selection()
    pilot = {(row.ticker, row.fiscal_year): row for row in load_manifest(PILOT)}
    selected: list[Filing] = []
    problems: list[str] = []

    for ticker, cik, company_name, years in companies:
        for y in years:
            if (ticker, y) in pilot and pilot[ticker, y].cik != cik:
                problems.append(f"{ticker} FY{y}: CIK disagrees with validated pilot")
        try:
            payload = sec.json(f"https://data.sec.gov/submissions/CIK{cik}.json")
            filing_data = payload["filings"]
            rows = list(submissions_rows(filing_data["recent"]))
            available = {row["fiscal_year"] for row in rows}
            needed = set(years) - available
            if needed:
                for extra in filing_data.get("files", []):
                    older = sec.json(f'https://data.sec.gov/submissions/{extra["name"]}')
                    rows.extend(submissions_rows(older))
                    available.update(row["fiscal_year"] for row in rows)
                    if set(years) <= available:
                        break
        except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
            problems.append(f"{ticker}: SEC metadata lookup failed: {exc}")
            continue

        by_year: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in rows:
            if row["fiscal_year"] in years:
                by_year[row["fiscal_year"]][row["accession_number"]] = row

        for year in years:
            if (ticker, year) in pilot:
                selected.append(pilot[ticker, year])
                print(f"{ticker} FY{year}: preserved original pilot record")
                continue
            options = list(by_year[year].values())
            if len(options) != 1:
                problems.append(f"{ticker} FY{year}: expected 1 original 10-K, found {len(options)}")
                continue
            filing = options[0]
            accession = filing["accession_number"]
            document = filing["primary_document"]
            fields = {
                "document_id": f"{cik}_{accession}",
                "company_name": company_name,
                "ticker": ticker,
                "cik": cik,
                "accession_number": accession,
                "form_type": "10-K",
                "fiscal_year": year,
                "report_period_end": filing["report_period_end"],
                "filing_date": filing["filing_date"],
                "primary_document": document,
                "source_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession.replace('-', '')}/{document}"
                ),
            }
            try:
                record = Filing.model_validate(fields)
                validate_filing(record)
            except (ValueError, Exception) as exc:
                problems.append(f"{ticker} FY{year}: invalid metadata: {exc}")
                continue
            selected.append(record)
            print(f"{ticker} FY{year}: {record.accession_number} ({record.report_period_end})")

    if problems:
        print("\nMANIFEST NOT WRITTEN; issues to investigate:", file=sys.stderr)
        for problem in problems:
            print(" -", problem, file=sys.stderr)
        print(f"Valid records so far: {len(selected)}/75", file=sys.stderr)
        return 1

    if len(selected) != 75:
        raise ValueError(f"Expected 75 filings, found {len(selected)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in selected:
            values = record.model_dump(mode="json")
            writer.writerow({col: values[col] for col in MANIFEST_COLUMNS})
    validated = load_manifest(OUTPUT)
    print(f"\nSuccess: wrote and validated {len(validated)} original 10-K filings in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
