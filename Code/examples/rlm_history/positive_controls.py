"""Compare a saved Amazon/Alphabet result with known canonical filing controls."""
import argparse
import json
from pathlib import Path


def main():
    controls = {
        "AMZN": {2023: "b79aa2561327a953e141e2aa", 2024: "4d1474550060251037c5f981"},
        "GOOGL": {2023: "15787429383b3e9839f69716", 2024: "62bad042e2e9d8334c2e6267"},
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("result") is None:
        parser.error("Use the initial Amazon/Alphabet research report with its candidate scan.")
    result = report["result"]
    scan = result["scan"]["sections"]
    retained = {entry["source"]["chunk_id"] for finding in report["final_report"]["original_findings"]
                for entry in finding["evidence"]}
    targets = {(entry["ticker"], entry["fiscal_year"]): entry for entry in report["coverage"]["targets"]}
    rows = []
    for row in scan:
        ticker, year = row["ticker"], row["fiscal_year"]
        if ticker not in controls or year not in controls[ticker]:
            continue
        chunk_id = controls[ticker][year]
        target = targets[ticker, year]
        candidates = [hit["chunk_id"] for hit in row["candidates"]]
        rows.append(dict(ticker=ticker, fiscal_year=year, control_chunk=chunk_id,
                         candidate_rank=candidates.index(chunk_id) + 1 if chunk_id in candidates else None,
                         acknowledged_text=chunk_id in target["direct_span_chunk_ids"] + target["whole_chunk_ids"],
                         retained=chunk_id in retained))
    print(json.dumps({"controls": rows, "conclusions": report["final_report"]["conclusions"]}, indent=2))


if __name__ == "__main__":
    main()
