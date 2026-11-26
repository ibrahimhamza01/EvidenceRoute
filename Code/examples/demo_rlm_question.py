"""Run a natural-language SEC question with Bedrock or one of two offline fixtures."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits
from evidenceroute.research.rlm.interpretation import BedrockIntentInterpreter, InterpretationLimits
from evidenceroute.research.rlm.natural_language import research_question
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import observation_json, trace_data


OFFLINE_CASES = {
    "What AI infrastructure supply risks does Amazon disclose in FY2024 Item 1A?": {
        "company": "AMZN", "section_inferred": False, "topic": "AI infrastructure supply risks",
        "phrases": ["AI infrastructure", "semiconductor products", "graphics processing units", "supply constraints"],
        "chunk_id": "4d1474550060251037c5f981",
        "quote": "For example, we rely on a limited group of suppliers for semiconductor products, "
                 "including products related to artificial intelligence infrastructure such as graphics processing units.",
        "claim": "Amazon discloses reliance on a limited group of semiconductor suppliers for products including GPUs.",
    },
    "What foreign exchange risks does Apple disclose in FY2024?": {
        "company": "AAPL", "section_inferred": True, "topic": "Foreign exchange exposure",
        "phrases": ["foreign currency", "exchange rate fluctuations", "foreign exchange"],
        "chunk_id": "cd7988623c9a3bf545f06045",
        "quote": "The weakening of foreign currencies relative to the U.S. dollar adversely affects the U.S. dollar value "
                 "of the Company’s foreign currency–denominated sales and earnings, and generally leads the Company to "
                 "raise international pricing, potentially reducing demand for the Company’s products.",
        "claim": "A stronger U.S. dollar can reduce the translated value of Apple's foreign-currency sales and earnings.",
    },
}


class ScriptedInterpretationTransport:
    def send(self, payload, *, timeout_seconds):
        question = json.loads(payload["messages"][0]["content"][0]["text"])["question"]
        case = OFFLINE_CASES[question]
        inferred = ["objective", "comprehensive"]
        if case["section_inferred"]:
            inferred.append("section_ids")
        return {
            "stopReason": "tool_use",
            "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                "toolUseId": "fixture.interpretation",
                "name": payload["toolConfig"]["toolChoice"]["tool"]["name"],
                "input": {
                    "mode": "review", "explanation": None, "discovery_phrases": case["phrases"],
                    "requirements": {"company_selection": "explicit", "companies": [case["company"]],
                                     "fiscal_years": [2024], "section_ids": ["item_1a"], "objective": "review",
                                     "topic": case["topic"], "comprehensive": False, "inferred_fields": inferred,
                                     "scope_notes": "Bounded disclosure review."
                                                    + (" Use Item 1A for this risk question." if case["section_inferred"] else "")},
                },
            }}]}},
        }


def _scripted_research(question):
    case = OFFLINE_CASES[question]
    chunk_id = case["chunk_id"]

    def factory(invocation):
        finish = {"type": "finish", "answer": "Scripted review of one source passage.",
                  "analysis_status": "incomplete", "cited_chunk_ids": [chunk_id]}
        if invocation.depth:
            read = {"type": "tool", "name": "get_chunk", "arguments": {"chunk_id": chunk_id}}
            finish["findings"] = [{"claim": case["claim"], "analysis_status": "analyzed",
                                   "evidence": [{"chunk_id": chunk_id, "quote": case["quote"]}]}]
            return ScriptedModelClient([read, finish])
        assign = {"type": "tool", "name": "delegate_research", "arguments": {
            "question": question, "goal": "Examine one passage relevant to the question.",
            "partitions": [asdict(pair) for pair in invocation.partitions],
        }}
        namespace = invocation.invocation_id.removesuffix("root")
        finish["used_finding_ids"] = [f"{namespace}child-1:finding-1"]
        return ScriptedModelClient([assign, finish])

    return factory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The original SEC research question.")
    parser.add_argument("--live", action="store_true", help="Authorize Bedrock interpretation and research calls.")
    parser.add_argument("--limits-file", type=Path, help="Overall corpus limits, including interpretation.")
    parser.add_argument("--interpretation-limits-file", type=Path, help="Optional interpretation-only limits JSON.")
    parser.add_argument("--profile", choices=tuple(COMPARATIVE_PROFILES), default="diagnostic", help="Per-tree profile.")
    parser.add_argument("--trace", action="store_true", help="Include detailed sanitized research diagnostics.")
    args = parser.parse_args(argv)
    if args.live and not args.limits_file:
        parser.error("--live requires --limits-file with overall allowances.")
    if not args.live and args.question not in OFFLINE_CASES:
        parser.error("Offline mode has scripted responses for these exact questions:\n" + "\n".join(OFFLINE_CASES))
    profile = COMPARATIVE_PROFILES[args.profile]
    shared = profile.shared_limits
    try:
        limits = (CorpusLimits.model_validate_json(args.limits_file.read_bytes()) if args.limits_file else CorpusLimits(
            max_batches=1, max_provider_attempts=shared.max_provider_attempts + 2,
            max_reported_tokens=shared.max_reported_tokens + 16_000, max_corpus_attempts=shared.max_corpus_attempts,
            max_delegations=shared.max_delegations, max_partitions=shared.max_partitions, deadline_seconds=180.0,
        ))
        caps = (InterpretationLimits.model_validate_json(args.interpretation_limits_file.read_bytes())
                if args.interpretation_limits_file else InterpretationLimits())
        factory = ((lambda _: BedrockRootClient.from_env(disable_retries=True)) if args.live
                   else _scripted_research(args.question))
        runner = RecursiveResearch(
            CorpusEnvironment(), factory, shared_limits=shared, local_limits=profile.local_limits,
            finalization=profile.finalization, max_root_reinvestigation_calls=profile.max_root_reinvestigation_calls,
        )
        interpreter_factory = None
        if not args.live:
            def interpreter_factory():
                return BedrockIntentInterpreter(ScriptedInterpretationTransport(), config=BedrockConfig(model="offline-fixture"))

        result = research_question(args.question, runner, limits=limits, interpretation_limits=caps,
                                   interpreter_factory=interpreter_factory, max_groups=profile.max_groups,
                                   detailed_trace=args.trace)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "mode": "live Bedrock" if args.live else "offline scripted verification",
        "live_provider_attempts": result.usage.provider_attempts if args.live else 0,
        "independent_semantic_verification": False,
        "result": result,
    }
    print(json.dumps(trace_data(json.loads(observation_json(report))), indent=2))
    return 0 if result.research is not None and result.research.execution_status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
