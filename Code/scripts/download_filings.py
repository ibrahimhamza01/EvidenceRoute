#!/usr/bin/env python3
"""Run from repository root: PYTHONPATH=Code/src python Code/scripts/download_filings.py ..."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from evidenceroute.ingestion.downloader import (
    AcquisitionError,
    FilingDownloader,
    load_manifest,
    relative_filing_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire exact SEC 10-K HTML files in a frozen CSV manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/filings_manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results", type=Path, default=Path("data/raw/acquisition_results.jsonl"))
    parser.add_argument("--ticker", action="append", help="Optional ticker filter (repeatable)")
    parser.add_argument("--year", type=int, action="append", help="Optional fiscal year filter (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print plan without network or writes")
    parser.add_argument("--s3", action="store_true", help="Publish into your allocated S3 prefix")
    parser.add_argument("--s3-bucket", default=os.environ.get("EVIDENCEROUTE_S3_BUCKET", "dats-nlp"))
    parser.add_argument(
        "--s3-prefix",
        default=os.environ.get("EVIDENCEROUTE_S3_PREFIX", "syedibrahim.hamza@gwu.edu/EvidenceRoute"),
    )
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""))
    args = parser.parse_args(argv)

    try:
        filings = load_manifest(args.manifest)
        if args.ticker:
            wanted = {t.upper().strip() for t in args.ticker}
            filings = [f for f in filings if f.ticker.upper() in wanted]
        if args.year:
            filings = [f for f in filings if f.fiscal_year in args.year]
        if not filings:
            raise AcquisitionError("Filters matched zero filings")
        print(f"Validated {len(filings)} filing(s) from {args.manifest}")
        if args.dry_run:
            for filing in filings:
                relative = relative_filing_path(filing).as_posix()
                print(f"  {filing.ticker} FY{filing.fiscal_year}: {filing.source_url}")
                print(f"    local: {args.raw_dir / relative}")
                if args.s3:
                    print(f"    s3: s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/raw/{relative}")
            return 0

        downloader = FilingDownloader(
            user_agent=args.user_agent,
            raw_dir=args.raw_dir,
            results_path=args.results,
        )
        s3_client = None
        if args.s3:
            import boto3

            s3_client = boto3.client("s3")  # EC2 instance role / normal AWS credential chain
        errors = 0
        for filing in filings:
            print(f"[{filing.ticker} FY{filing.fiscal_year}]", flush=True)
            try:
                result = downloader.acquire(
                    filing,
                    s3_client=s3_client,
                    bucket=args.s3_bucket if args.s3 else None,
                    prefix=args.s3_prefix,
                )
                print(
                    f"  {result['origin']}: {result['size_bytes']:,} bytes, "
                    f"sha256={result['sha256']} (S3: {result['s3_status']})"
                )
            except Exception as exc:
                errors += 1
                print(f"  ERROR: {exc}", file=sys.stderr)
                # Preserve per-filing failures without modifying the frozen selection manifest.
                from evidenceroute.ingestion.downloader import append_result

                append_result(args.results, {
                    "status": "error", "document_id": filing.document_id,
                    "ticker": filing.ticker, "fiscal_year": filing.fiscal_year,
                    "source_url": filing.source_url, "error": str(exc),
                })
        print(f"Finished: {len(filings) - errors} succeeded; {errors} failed")
        return 1 if errors else 0
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
