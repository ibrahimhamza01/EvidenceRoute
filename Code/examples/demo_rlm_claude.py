"""Opt-in, bounded Claude demonstration against one SEC disclosure."""

import argparse
import json
from dataclasses import asdict

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.claude import ClaudeRootClient
from evidenceroute.research.rlm.controller import RootController, RootLimits
from evidenceroute.research.rlm.coverage import ResearchScope, SectionTarget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize live model API requests.")
    parser.add_argument("--provider", choices=("bedrock", "anthropic"), default="bedrock")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to opt in to provider requests.")
    try:
        factory = BedrockRootClient if args.provider == "bedrock" else ClaudeRootClient
        client = factory.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    corpus = CorpusEnvironment()
    filing, = corpus.list_filings(ticker="NVDA", fiscal_year=2024).items
    section_id = "item_1a"
    question = (
        "Identify one supply-chain risk disclosure from NVIDIA's FY2024 Item 1A. "
        f"Use document_id {filing.document_id}, section_id {section_id}, and cite a returned chunk ID."
    )
    scope = ResearchScope((filing.document_id,), (SectionTarget(filing.document_id, section_id),))
    trace = RootController(
        corpus, client,
        limits=RootLimits(max_iterations=6, max_tool_calls=4, max_response_bytes=6_000, deadline_seconds=45),
    ).run(question, scope=scope)
    attempts = trace.model_attempts
    token_fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    totals = {}
    for field in token_fields:
        counts = [getattr(attempt, field) for attempt in attempts if getattr(attempt, field) is not None]
        totals[field] = {"reported_total": sum(counts) if counts else None, "reporting_calls": len(counts)}
    citations = [
        asdict(ref) for ref in trace.coverage.visible_chunks if ref.chunk_id in trace.cited_chunk_ids
    ]
    print(json.dumps({
        "question": trace.question,
        "answer": trace.answer,
        "stop_reason": trace.stop_reason,
        "citations": citations,
        "coverage": asdict(trace.coverage),
        "provider": args.provider,
        "model": client.config.model,
        "api_calls": len(attempts),
        "tool_calls": trace.tool_calls,
        "tokens": totals,
        "elapsed_seconds": trace.elapsed_seconds,
        "provider_attempts": [asdict(attempt) for attempt in attempts],
    }, indent=2))
    return 0 if trace.stop_reason == "finished" and citations else 1


if __name__ == "__main__":
    raise SystemExit(main())
