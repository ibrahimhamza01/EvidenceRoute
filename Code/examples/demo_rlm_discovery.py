"""Compare candidate discovery with the literal baseline on canonical SEC data, offline."""

import json

from demo_rlm_comparative import DISCOVERY_PHRASES, TERMS
from demo_rlm_plan import QUESTION, scripted_interpretation
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.discovery import DiscoverySpec, discover_research_plan, scan_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan


# Known assessed chunks from the live Amazon–Alphabet run and earlier Microsoft investigation.
KNOWN_TARGETS = {
    "0982db825d8177d57ab2c33e", "15787429383b3e9839f69716", "4d1474550060251037c5f981",
    "62bad042e2e9d8334c2e6267", "b79aa2561327a953e141e2aa", "e09b0898eb296a45490cd615",
    "e0ec543a580b908a6602bba5", "4815881d3fdefcf0a17e31d7", "4f9a71476fee75c3bb848dd2",
}


def _summary(scan, known=()):
    candidates = {hit.chunk_id for row in scan.sections for hit in row.candidates}
    return {
        "scanned_chunks": sum(row.scanned_chunks for row in scan.sections),
        "page_reads": scan.pages_read, "metadata_reads": scan.metadata_reads, "stop_reason": scan.stop_reason,
        "matched_chunks": sum(row.matched_chunks for row in scan.sections), "candidate_chunks": len(candidates),
        "known_targets_found": sorted(candidates.intersection(known)), "known_targets_missing": sorted(set(known) - candidates),
        "by_section": [{"ticker": row.ticker, "fiscal_year": row.fiscal_year, "section_id": row.pair.section_id,
                        "scanned_chunks": row.scanned_chunks, "candidate_chunks": len(row.candidates)}
                       for row in scan.sections],
    }


def main() -> None:
    corpus = CorpusEnvironment()
    ai = resolve_research_plan(interpret_intent(QUESTION, lambda *args: dict(
        scripted_interpretation(*args), company_selection="explicit", companies=["AMZN", "GOOGL", "MSFT"],
    )), corpus)
    spec = DiscoverySpec.from_intent(ai.intent, DISCOVERY_PHRASES)
    compare = resolve_research_plan(interpret_intent("Which filings discuss exposure to foreign currency fluctuations?", lambda *_: {
        "company_selection": "explicit", "companies": ["AAPL", "NVDA"], "fiscal_years": [2024],
        "section_ids": ["item_1a"], "objective": "review", "topic": "Foreign currency exposure",
        "comprehensive": False, "inferred_fields": [],
    }), corpus)
    foreign = DiscoverySpec.from_intent(compare.intent, ("foreign currency", "exchange rate fluctuations", "foreign exchange"))
    legacy = scan_research_plan(corpus, ai, TERMS)
    normalized = discover_research_plan(corpus, ai, spec)
    unrelated = discover_research_plan(corpus, compare, foreign)
    print(json.dumps({
        "provider_attempts": 0, "model_assessment_performed": False,
        "literal_ai_baseline": _summary(legacy, KNOWN_TARGETS),
        "normalized_ai_phrases": _summary(normalized, KNOWN_TARGETS),
        "foreign_currency_review": _summary(unrelated),
    }, indent=2))


if __name__ == "__main__":
    main()
