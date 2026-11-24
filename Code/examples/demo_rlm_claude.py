"""Opt-in, bounded Claude demonstration against one SEC disclosure."""

import argparse
import json
from dataclasses import asdict

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.claude import ClaudeRootClient
from evidenceroute.research.rlm.controller import RootController, RootLimits, StepTrace
from evidenceroute.research.rlm.coverage import ResearchScope, SectionTarget


def _step_details(step: StepTrace) -> dict:
    return {
        "iteration": step.iteration,
        "requested_action": step.requested_action,
        "validated_arguments": step.validated_arguments,
        "observation_error": step.observation.get("error") if step.observation is not None else None,
        "step_error": step.error if step.observation is None else None,
        "observation_sent": step.observation_sent,
        "observation_chunk_ids": [ref.chunk_id for ref in step.evidence],
        "search_discovered_chunk_ids": [ref.chunk_id for ref in step.search_discovered],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize live model API requests.")
    parser.add_argument("--provider", choices=("bedrock", "anthropic"), default="bedrock")
    parser.add_argument("--max-iterations", type=int, default=6, help="Model decision limit (default: 6).")
    parser.add_argument("--max-tool-calls", type=int, default=4, help="Corpus tool attempt limit (default: 4).")
    parser.add_argument("--max-response-bytes", type=int, default=6_000,
                        help="UTF-8 bytes per tool observation (default: 6000).")
    parser.add_argument("--deadline-seconds", type=float, default=45, help="Overall deadline (default: 45s).")
    parser.add_argument("--trace", action="store_true", help="Include decisions, errors, and evidence IDs.")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("Use --live to opt in to provider requests.")
    try:
        limits = RootLimits(
            max_iterations=args.max_iterations, max_tool_calls=args.max_tool_calls,
            max_response_bytes=args.max_response_bytes, deadline_seconds=args.deadline_seconds,
        )
        factory = BedrockRootClient if args.provider == "bedrock" else ClaudeRootClient
        client = factory.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    corpus = CorpusEnvironment()
    filing, = corpus.list_filings(ticker="NVDA", fiscal_year=2024).items
    section_id = "item_1a"
    question = (
        "Identify one supply-chain risk disclosure from NVIDIA's FY2024 Item 1A. "
        f"Start with search_chunks using limit=1 and filters for document_id {filing.document_id} "
        f"and section_id {section_id}. Cite a returned chunk ID."
    )
    scope = ResearchScope((filing.document_id,), (SectionTarget(filing.document_id, section_id),))
    trace = RootController(corpus, client, limits=limits).run(question, scope=scope)
    attempts = trace.model_attempts
    token_fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    totals = {}
    for field in token_fields:
        counts = [getattr(attempt, field) for attempt in attempts if getattr(attempt, field) is not None]
        totals[field] = {"reported_total": sum(counts) if counts else None, "reporting_calls": len(counts)}
    citations = [
        asdict(ref) for ref in trace.coverage.visible_chunks if ref.chunk_id in trace.cited_chunk_ids
    ]
    report = {
        "question": trace.question,
        "answer": trace.answer,
        "stop_reason": trace.stop_reason,
        "citations": citations,
        "coverage": asdict(trace.coverage),
        "provider": args.provider,
        "model": client.config.model,
        "api_calls": len(attempts),
        "tool_calls": trace.tool_calls,
        "limits": limits.model_dump(mode="json"),
        "tokens": totals,
        "elapsed_seconds": trace.elapsed_seconds,
        "provider_attempts": [asdict(attempt) for attempt in attempts],
    }
    if args.trace:
        report["steps"] = [_step_details(step) for step in trace.steps]
    print(json.dumps(report, indent=2))
    return 0 if trace.stop_reason == "finished" and citations else 1


if __name__ == "__main__":
    raise SystemExit(main())
