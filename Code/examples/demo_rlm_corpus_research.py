"""Run a resolved SEC comparison across sequential, independently bounded trees."""

import argparse
import json
from pathlib import Path

from demo_rlm_comparative import DISCOVERY_PHRASES, GUIDANCE, ROOT_GUIDANCE, TERMS, _offline_factory
from demo_rlm_plan import QUESTION, scripted_interpretation
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockRootClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import observation_json, trace_data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(COMPARATIVE_PROFILES), default="diagnostic")
    parser.add_argument("--company", action="append", help="Request a ticker or company name. Repeat for multiple companies.")
    parser.add_argument("--term", action="append", help="Replace the example phrases. Repeat for variants.")
    parser.add_argument("--fixed-terms", action="store_true", help="Reproduce the literal-term baseline.")
    budget_options = parser.add_mutually_exclusive_group()
    budget_options.add_argument("--limited", action="store_true", help="Stop after the first allotted tree's partition work.")
    budget_options.add_argument("--limits-file", type=Path, help="JSON corpus-level limits.")
    parser.add_argument("--live", action="store_true", help="Authorize Bedrock calls with an explicit limits file.")
    parser.add_argument("--trace", action="store_true", help="Include detailed sanitized decision and delivery telemetry.")
    args = parser.parse_args(argv)
    if args.live and not args.limits_file:
        parser.error("--live requires --limits-file with explicit overall allowances.")

    profile = COMPARATIVE_PROFILES[args.profile]
    shared = profile.shared_limits
    try:
        limits = (CorpusLimits.model_validate_json(args.limits_file.read_bytes()) if args.limits_file else CorpusLimits(
            max_batches=13, max_provider_attempts=shared.max_provider_attempts, max_reported_tokens=shared.max_reported_tokens,
            max_corpus_attempts=shared.max_corpus_attempts + 2, max_delegations=26,
            max_partitions=shared.max_partitions if args.limited else 52, deadline_seconds=180.0,
        ))
        corpus = CorpusEnvironment()

        def interpret(question, schema):
            intent = scripted_interpretation(question, schema)
            if args.company:
                intent.update(company_selection="explicit", companies=args.company)
            return intent

        question = QUESTION if not args.company else QUESTION.replace("Which companies", "Which requested companies", 1)
        plan = resolve_research_plan(interpret_intent(question, interpret), corpus)
        factory = (lambda _: BedrockRootClient.from_env(disable_retries=True)) if args.live else _offline_factory(corpus, plan)
        runner = RecursiveResearch(corpus, factory, shared_limits=shared, local_limits=profile.local_limits,
                                   finalization=profile.finalization, child_guidance=GUIDANCE,
                                   max_root_reinvestigation_calls=profile.max_root_reinvestigation_calls)
        scan_terms = tuple(args.term or TERMS) if args.fixed_terms else None
        discovery = None if args.fixed_terms else DiscoverySpec.from_intent(plan.intent, tuple(args.term or DISCOVERY_PHRASES))
        result = execute_corpus_research(plan, runner, limits=limits, max_groups=profile.max_groups,
                                         scan_terms=scan_terms, discovery=discovery, root_guidance=ROOT_GUIDANCE,
                                         detailed_trace=args.trace)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "mode": "live Bedrock" if args.live else "offline scripted",
        "assessment_source": "model-reported" if args.live else "scripted example with other companies unassessed",
        "profile": args.profile, "tree_configuration": profile, "required_targets": plan.required_target_count,
        "resolved_targets": len(plan.resolved_targets), "independent_semantic_verification": False,
        "group_counts": result.group_counts, "result": result,
    }
    print(json.dumps(trace_data(json.loads(observation_json(report))), indent=2))
    return 0 if result.execution_status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
