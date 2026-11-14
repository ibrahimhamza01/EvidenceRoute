"""Build the verified development chunk fixture used by the retrieval tests (Issue #4).

The fixture is a small set of genuine passages copied from the four pilot SEC 10-K
filings. Each passage is selected by an anchor phrase, located inside the parsed
section text, and verified twice:

1. ``section.text[char_start:char_end] == passage`` (exact offsets in the section).
2. The passage appears in the visible text of the original HTML (whitespace-insensitive),
   independently of the section parser.

The retrieval package never imports this script or the ingestion code. Only the
resulting JSONL fixture is consumed by retrieval tests and the demo.

Usage (from the repository root, after downloading the pilot filings):

    PYTHONPATH=Code/src python Code/scripts/build_retrieval_dev_chunks.py
    PYTHONPATH=Code/src python Code/scripts/build_retrieval_dev_chunks.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from evidenceroute.ingestion.downloader import load_manifest, relative_filing_path
from evidenceroute.ingestion.parser import extract_sections
from evidenceroute.schemas import Chunk, Filing

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/filings_manifest.csv"
DEFAULT_RAW_DIR = ROOT / "data/raw"
DEFAULT_OUTPUT = ROOT / "Code/tests/fixtures/retrieval_dev_chunks.jsonl"

MAX_PASSAGE_CHARS = 1500
SENTENCE_END = re.compile(r"[.;:](?=\s|$)")
NO_SPACE = re.compile(r"\s+")

# (ticker, fiscal_year, section_id, anchor). The anchor must start exactly one
# paragraph (line) of the parsed section text.
SELECTIONS: tuple[tuple[str, int, str, str], ...] = (
    # NVIDIA FY2023
    ("NVDA", 2023, "item_1", "NVIDIA pioneered accelerated computing"),
    ("NVDA", 2023, "item_1", "In fiscal year 2023, we introduced the Hopper architecture"),
    ("NVDA", 2023, "item_1a", "We depend on foundries to manufacture our semiconductor wafers"),
    ("NVDA", 2023, "item_1a", "Security breaches, computer malware, social-engineering attacks"),
    ("NVDA", 2023, "item_1a", "During the third quarter of fiscal year 2023, the U.S. government, or USG"),
    ("NVDA", 2023, "item_1a", "For example, in response to the war in Ukraine"),
    ("NVDA", 2023, "item_7", "Data Center revenue for fiscal year 2023 was $15.01 billion"),
    ("NVDA", 2023, "item_15", "We derive our revenue from product sales"),
    # NVIDIA FY2024
    ("NVDA", 2024, "item_1", "NVIDIA pioneered accelerated computing"),
    ("NVDA", 2024, "item_1", "During the third quarter of fiscal year 2023, the U.S. government, or the USG"),
    ("NVDA", 2024, "item_1", "We believe that our employees are our greatest assets"),
    ("NVDA", 2024, "item_1a", "We depend on foundries to manufacture our semiconductor wafers"),
    ("NVDA", 2024, "item_1a", "Security breaches, computer malware, social-engineering attacks"),
    ("NVDA", 2024, "item_1a", "Export controls could disrupt our supply chain"),
    ("NVDA", 2024, "item_7", "Data Center revenue for fiscal year 2024 was up 217%"),
    ("NVDA", 2024, "item_15", "We derive our revenue from product sales"),
    # AMD FY2023
    ("AMD", 2023, "item_1", "We are a global semiconductor company. Semiconductors are"),
    ("AMD", 2023, "item_1", "Data Center GPUs. Our AMD Instinct"),
    ("AMD", 2023, "item_1", "As of December 30, 2023, we had approximately 26,000 employees"),
    ("AMD", 2023, "item_1a", "We utilize third-party wafer foundries"),
    ("AMD", 2023, "item_1a", "Our business relies on technology hardware, software, cloud services"),
    ("AMD", 2023, "item_1a", "In October 2023, BIS issued new requirements"),
    ("AMD", 2023, "item_7", "Data Center net revenue of $6.5 billion in 2023"),
    ("AMD", 2023, "item_8", "Revenue is recognized when a customer obtains control"),
    # AMD FY2024
    ("AMD", 2024, "item_1", "One of our priorities in 2024 was to accelerate growth"),
    ("AMD", 2024, "item_1", "During the year, we completed the acquisition of Silo AI"),
    ("AMD", 2024, "item_1", "As of December 28, 2024, we had approximately 28,000 employees"),
    ("AMD", 2024, "item_1a", "We utilize third-party wafer foundries"),
    ("AMD", 2024, "item_1a", "Our business relies on technology hardware, software, cloud services"),
    ("AMD", 2024, "item_1a", "United States export control regulations include restrictions"),
    ("AMD", 2024, "item_7", "Data Center net revenue of $12.6 billion in 2024"),
    ("AMD", 2024, "item_8", "Revenue is recognized when a customer obtains control"),
)


class FixtureError(ValueError):
    """A selected passage could not be located or verified."""


def dev_chunk_id(document_id: str, section_id: str, start: int, end: int) -> str:
    """Stable ID, namespaced with ``dev`` so it never collides with pipeline chunk IDs."""
    key = f"dev|{document_id}|{section_id}|{start}|{end}"
    return "dev-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def locate_passage(section_text: str, anchor: str,
                   max_chars: int = MAX_PASSAGE_CHARS) -> tuple[int, int]:
    """Return [start, end) of the paragraph starting with ``anchor``.

    Paragraphs longer than ``max_chars`` are cut at the last sentence end before the
    limit so the passage is still an exact, readable substring of the section.
    """
    starts = [m.start() for m in re.finditer(re.escape(anchor), section_text)
              if m.start() == 0 or section_text[m.start() - 1] == "\n"]
    if len(starts) != 1:
        raise FixtureError(f"Anchor must start exactly one paragraph, found {len(starts)}: {anchor!r}")
    start = starts[0]
    end = section_text.find("\n", start)
    end = len(section_text) if end < 0 else end
    if end - start > max_chars:
        window = section_text[start:start + max_chars]
        cuts = [m.end() for m in SENTENCE_END.finditer(window)]
        if not cuts:
            raise FixtureError(f"No sentence boundary within {max_chars} chars: {anchor!r}")
        end = start + cuts[-1]
    return start, end


def html_visible_text(html_path: Path) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_path.read_bytes(), "lxml")
    for tag in soup.find_all(["script", "style", "noscript", "svg", "ix:header"]):
        tag.decompose()
    return NO_SPACE.sub("", soup.get_text(" "))


def build_chunks(manifest: Path, raw_dir: Path) -> list[Chunk]:
    filings: dict[tuple[str, int], Filing] = {
        (f.ticker, f.fiscal_year): f for f in load_manifest(manifest)
    }
    sections_cache: dict[tuple[str, int], dict] = {}
    html_cache: dict[tuple[str, int], str] = {}
    located: list[tuple[Filing, object, int, int]] = []

    for ticker, year, section_id, anchor in SELECTIONS:
        filing = filings.get((ticker, year))
        if filing is None:
            raise FixtureError(f"Filing {ticker} FY{year} is not in the manifest")
        html_path = raw_dir / relative_filing_path(filing)
        if not html_path.is_file():
            raise FixtureError(f"Raw filing is missing: {html_path}")
        if (ticker, year) not in sections_cache:
            sections = extract_sections(html_path, filing.document_id, filing.source_url)
            sections_cache[(ticker, year)] = {s.section_id: s for s in sections}
            html_cache[(ticker, year)] = html_visible_text(html_path)
        section = sections_cache[(ticker, year)].get(section_id)
        if section is None:
            raise FixtureError(f"Section {section_id} missing for {ticker} FY{year}")
        start, end = locate_passage(section.text, anchor)
        passage = section.text[start:end]
        if NO_SPACE.sub("", passage) not in html_cache[(ticker, year)]:
            raise FixtureError(f"Passage not found in original HTML: {anchor!r}")
        located.append((filing, section, start, end))

    # chunk_index is the passage order within its (document, section).
    located.sort(key=lambda item: (item[0].document_id, item[1].section_id, item[2]))
    chunks: list[Chunk] = []
    counters: dict[tuple[str, str], int] = {}
    for filing, section, start, end in located:
        key = (filing.document_id, section.section_id)
        index = counters.get(key, 0)
        counters[key] = index + 1
        chunk = Chunk(
            chunk_id=dev_chunk_id(filing.document_id, section.section_id, start, end),
            document_id=filing.document_id,
            company_name=filing.company_name,
            ticker=filing.ticker,
            fiscal_year=filing.fiscal_year,
            section_id=section.section_id,
            section_name=section.section_name,
            chunk_index=index,
            text=section.text[start:end],
            char_start=start,
            char_end=end,
            source_url=filing.source_url,
        )
        chunks.append(chunk)
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise FixtureError("Duplicate chunk IDs in selection")
    return chunks


def serialize(chunks: list[Chunk]) -> str:
    lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
             for c in chunks]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="Rebuild in memory and fail if the committed fixture differs.")
    args = parser.parse_args(argv)

    content = serialize(build_chunks(args.manifest, args.raw_dir))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
            print(f"Fixture is out of date: {args.output}", file=sys.stderr)
            return 1
        print(f"Fixture verified: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {content.count(chr(10))} verified chunks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
