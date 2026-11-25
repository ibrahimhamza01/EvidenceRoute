"""Run two planned company groups offline or explicitly opt in to Bedrock."""

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from functools import cache
from time import monotonic

from demo_rlm_plan import QUESTION, scripted_interpretation
from demo_rlm_recursive import _invocation_report, _usage, _walk
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedLimits
from evidenceroute.research.rlm.controller import RootLimits, ScriptedModelClient
from evidenceroute.research.rlm.coverage import SectionTarget
from evidenceroute.research.rlm.planned_research import execute_research_plan
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import trace_data
from evidenceroute.retrieval import ChunkFilter

QUERY = "artificial intelligence infrastructure"
COMPANIES = ("AAPL", "AMD")
ROOT_GUIDANCE = (
    "Delegate each company group once before synthesizing. Each child should investigate both assigned fiscal years. "
    "Use acknowledged findings from each company, declare their IDs in used_finding_ids, and cite an original "
    "chunk for each company and year where supported. Distinguish evidence from uncertainty and do not generalize "
    "to unexamined companies. Distinguish a supported FY2024 risk, comparable FY2023 evidence, possible changes "
    "needing verification, and insufficient evidence to determine novelty. Omit unsupported findings or abstain. "
    "Copy any quote in a finding exactly from a short contiguous span of delivered text. Keep punctuation and "
    "spacing, and avoid ellipses, paraphrases, and joined passages."
)
CHILD_GUIDANCE = (
    "Investigate comparable AI-related infrastructure risks in both authorized fiscal years. Choose your own "
    "focused queries and start with one chunk per request. Reserve investigation for both years. Return compact "
    "structured findings, one per year where supported, with original chunk IDs and quotes under 150 characters. "
    "Copy a contiguous passage exactly, preserving punctuation, capitalization and spacing. Do not insert ellipses, "
    "paraphrase, or join passages. Distinguish observed FY2024 risks from comparable FY2023 evidence, possible change, "
    "and insufficient evidence. Disclose partial coverage. General AI references alone do not establish an "
    "infrastructure risk. A search miss does not show a disclosure was absent or a risk was newly introduced. "
    "Omit unsupported findings or abstain."
)


def _search_arguments(pair):
    return {"query": QUERY, "limit": 1, "filters": {"document_ids": [pair.document_id],
                                                   "section_ids": [pair.section_id]}}


def _excerpt(chunk):
    match = re.search(r"artificial intelligence|infrastructure", chunk.text, re.IGNORECASE)
    start = max(0, match.start() - 45) if match else 0
    return chunk.text[start:start + 150].strip()


def _script_factory(corpus, plan):
    def factory(invocation):
        if invocation.parent_id is None:
            groups = [g for g in plan.groups if g.targets and set(g.targets).issubset(invocation.partitions)]
            actions = [{
                "type": "tool", "name": "delegate_research", "arguments": {
                    "question": f"Search Item 1A for both assigned years of {group.ticker} using the same query.",
                    "goal": "Return one original excerpt per year. Report the limits of this sample.",
                    "partitions": [asdict(pair) for pair in group.targets],
                },
            } for group in groups]
            hits = [candidates(group.targets) for group in groups]
            cited = [chunk.chunk_id for company_hits in hits for chunk in company_hits]
            used = [f"child-{index}:finding-{i}" for index, company_hits in enumerate(hits, 1)
                    for i in range(1, len(company_hits) + 1)]
            return ScriptedModelClient([*actions, {
                "type": "finish", "answer": "Cited candidate Item 1A excerpts from the same search across the "
                "assigned filings. The excerpts leave the original question unanswered, including whether any risk was new.",
                "analysis_status": "incomplete", "cited_chunk_ids": cited, "used_finding_ids": used,
            }])

        chunks = candidates(invocation.partitions)
        reads = [{"type": "tool", "name": "search_chunks", "arguments": _search_arguments(pair)}
                 for pair in invocation.partitions]
        return ScriptedModelClient([*reads, {
            "type": "finish", "answer": "Ran a targeted Item 1A search for each assigned year.",
            "analysis_status": "incomplete", "findings": [{
                "claim": f"The query returned this Item 1A excerpt from the FY{chunk.fiscal_year} filing.",
                "analysis_status": "incomplete", "uncertainty": "A sample cannot establish absence or novelty.",
                "evidence": [{"chunk_id": chunk.chunk_id, "quote": _excerpt(chunk)}],
            } for chunk in chunks],
        }])

    @cache
    def first_chunk(pair):
        # Seed scripted evidence references. The recursive tool path supplies and validates each observation.
        args = _search_arguments(pair)
        hits = corpus.search_chunks(QUERY, limit=1, filters=ChunkFilter.model_validate(args["filters"])).items
        return hits[0].chunk if hits else None

    def candidates(pairs):
        return [chunk for pair in pairs if (chunk := first_chunk(pair)) is not None]

    return factory


def _structural_criteria(result):
    trace = result.trace
    used = set(trace.used_finding_ids) if trace else set()
    cited = {ref.chunk_id for ref in result.citations}
    outcomes = {item.invocation.invocation_id: item for item in result.coverage.invocations}
    groups = {group.ticker: group for group in result.groups}
    required = {group.ticker: set(group.targets) for group in result.plan.groups}
    by_company = {}
    for ticker in COMPANIES:
        group = groups.get(ticker)
        children = [outcomes[key] for key in group.root_delegate_ids] if group else []
        findings = [finding for child in children if child.stop_reason == "finished" and child.parent_acknowledged
                    for finding in child.findings if finding.finding_id in child.findings_acknowledged_by_parent]
        targets = required.get(ticker, set())
        acknowledged_pairs = {SectionTarget(item.source.document_id, item.source.section_id)
                              for finding in findings for item in finding.evidence}
        used_citations = {SectionTarget(item.source.document_id, item.source.section_id)
                          for finding in findings if finding.finding_id in used
                          for item in finding.evidence if item.source.chunk_id in cited}
        by_company[ticker] = {
            "whole_group_delegated": bool(children),
            "child_finished": any(child.stop_reason == "finished" for child in children),
            "both_years_acknowledged": len(targets) == 2 and targets <= acknowledged_pairs,
            "both_years_cited_from_declared_used_findings": len(targets) == 2 and targets <= used_citations,
        }
    return {
        "expected_groups_admitted": tuple(g.ticker for g in result.groups if g.pending_reason is None) == COMPANIES,
        "root_finished_with_synthesis": bool(trace and trace.stop_reason == "finished" and trace.answer and used),
        "companies": by_company,
    }


def _live_limits(args):
    shared = SharedLimits(
        max_depth=args.max_depth, max_delegations=args.max_delegations, max_partitions=args.max_partitions,
        max_corpus_attempts=args.max_corpus_attempts, max_provider_attempts=args.max_provider_attempts,
        max_reported_tokens=args.max_reported_tokens, deadline_seconds=args.deadline_seconds,
    )
    finalization = FinalizationPolicy(
        child_provider_attempts=args.child_provider_attempts, finalization_attempts=2, finalization_tool_choice="required",
        finalization_max_output_tokens=args.finalization_max_output_tokens,
    )
    root_attempts = shared.max_provider_attempts - shared.max_delegations * finalization.child_provider_attempts
    local = RootLimits(
        max_iterations=max(root_attempts, finalization.child_provider_attempts), max_tool_calls=args.max_tool_calls,
        max_response_bytes=args.max_response_bytes, deadline_seconds=shared.deadline_seconds,
    )
    return shared, local, finalization


def _report(result, args, runner, config, elapsed):
    trace = result.trace
    targets = result.coverage.targets
    criteria = _structural_criteria(result)
    success = (criteria["expected_groups_admitted"] and criteria["root_finished_with_synthesis"]
               and all(all(checks.values()) for checks in criteria["companies"].values()))
    return {
        "mode": "live Bedrock planned research" if args.live else "scripted offline planned research",
        "intent_interpretation": "scripted",
        "intent": result.plan.intent.model_dump(mode="json"),
        "requested_companies": len(result.plan.requested_companies),
        "required_targets": result.coverage.required_target_count,
        "admitted_scope": [{**asdict(t.pair), "ticker": t.ticker, "fiscal_year": t.fiscal_year}
                           for t in targets if t.admitted],
        "answer": trace.answer if trace else None,
        "answer_status": result.answer_status,
        "semantic_claims_verified": False,
        "structural_status": "success" if success else "incomplete",
        "structural_criteria": criteria,
        "stop_reason": trace.stop_reason if trace else None,
        "budget_stop": trace.budget_stop if trace else None,
        "wall_clock_seconds": elapsed,
        "access_counts_by_target": dict(Counter(t.status for t in targets)),
        "unique_search_discovered_chunks": len({chunk for t in targets for chunk in t.discovered_chunk_ids}),
        "unique_whole_chunks_delivered": len({chunk for t in targets for chunk in t.whole_chunk_ids}),
        "resolved_chunk_inventory": sum(t.chunk_inventory_count for t in targets),
        "section_characters_delivered": sum(end - start for t in targets for start, end in t.character_ranges),
        "resolved_section_characters": sum(t.section_character_count for t in targets),
        "groups": [asdict(group) for group in result.groups],
        "ledger": asdict(result.coverage),
        "original_citations": [asdict(citation) for citation in result.citations],
        "shared_usage": asdict(trace.shared_budget) if trace else None,
        "limits": {
            "max_groups": len(COMPANIES), "shared": runner.shared_limits.model_dump(), "local": runner.local_limits.model_dump(),
            "finalization": runner.finalization.model_dump(),
        },
        "provider": {"name": "bedrock", **config.model_dump()} if config else None,
        "provider_usage": _usage(tuple(_walk(trace)) if trace else ()),
        "trace": _invocation_report(trace, args.trace) if trace and (args.live or args.trace) else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize Bedrock requests instead of scripted execution.")
    parser.add_argument("--trace", action="store_true", help="Include detailed provider and tool steps.")
    live = parser.add_argument_group("live experiment limits")
    for name, default in (
        ("max-depth", 1), ("max-delegations", 2), ("max-partitions", 4), ("max-corpus-attempts", 12),
        ("max-provider-attempts", 18), ("max-reported-tokens", 100_000), ("child-provider-attempts", 6),
        ("max-tool-calls", 6), ("max-response-bytes", 6_000), ("finalization-max-output-tokens", 2_048),
    ):
        live.add_argument(f"--{name}", type=int, default=default, help="Default: %(default)s.")
    live.add_argument("--deadline-seconds", type=float, default=180.0, help="Shared and local deadline in seconds.")
    args = parser.parse_args(argv)
    config = None

    def live_factory(invocation):
        nonlocal config
        client = BedrockRootClient.from_env(disable_retries=True)
        if invocation.depth == 0:
            config = client.config
        return client

    try:
        corpus = CorpusEnvironment()
        plan = resolve_research_plan(interpret_intent(QUESTION, scripted_interpretation), corpus)
        if args.live:
            shared, local, finalization = _live_limits(args)
            runner = RecursiveResearch(corpus, live_factory, shared_limits=shared, local_limits=local,
                                       finalization=finalization, child_guidance=CHILD_GUIDANCE)
        else:
            runner = RecursiveResearch(
                corpus, _script_factory(corpus, plan),
                shared_limits=SharedLimits(max_depth=1, max_delegations=2, max_partitions=4,
                                           max_corpus_attempts=4, max_provider_attempts=12),
                local_limits=RootLimits(max_iterations=6, max_tool_calls=4, max_response_bytes=6_000),
                finalization=FinalizationPolicy(),
            )
        started = monotonic()
        result = execute_research_plan(plan, runner, max_groups=len(COMPANIES),
                                       root_guidance=ROOT_GUIDANCE if args.live else None)
        report = _report(result, args, runner, config, monotonic() - started)
    except FileNotFoundError:
        parser.error("The demo requires the local canonical corpus.")
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(trace_data(report), indent=2))
    return 0 if report["structural_status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
