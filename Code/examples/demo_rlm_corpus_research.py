"""Run a resolved SEC comparison across sequential, independently bounded trees."""

import argparse
import json
from pathlib import Path


from demo_rlm_comparative import DISCOVERY_PHRASES, GUIDANCE, ROOT_GUIDANCE, TERMS, _offline_factory
from demo_rlm_plan import QUESTION, scripted_interpretation
from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockRootClient
from evidenceroute.research.rlm.checkpoint import continue_research, create_checkpoint, load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan
from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.verification import BedrockSemanticVerifier, review_checkpoint, verification_report
from evidenceroute.research.rlm.verification_records import VerificationLimits
from evidenceroute.research.rlm.tools import observation_json, trace_data


class _OfflineVerifierTransport:
    def send(self, payload, *, timeout_seconds):
        cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
        reviews = []
        for case in cases:
            # Script a partial judgment of the known AMD power/water example.
            refs = [item["ref"] for item in case["selected_evidence"]]
            review = {"ref": case["ref"], "components": [
                {"statement": "The filing describes a general data-center power or water constraint.",
                 "status": "supported", "evidence": refs[:1]},
                {"statement": "The constraint specifically concerns AI.",
                 "status": "insufficient_evidence", "evidence": [], "gap": "Find exact evidence of an AI dependency."},
            ]}
            if case["kind"] == "comparison":
                review["comparison"] = {"relationship": "same_risk", "relationship_evidence": refs[:2],
                                        "earlier_year": "supports_risk", "earlier_evidence": [
                                            item["ref"] for item in case["selected_evidence"] if item["year"] == min(case["years"])
                                        ]}
            reviews.append(review)
        return {"stopReason": "tool_use", "usage": {"inputTokens": 500, "outputTokens": 250},
                "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                    "toolUseId": "scripted-verification",
                    "name": payload["toolConfig"]["toolChoice"]["tool"]["name"], "input": {"reviews": reviews},
                }}]}}}


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
    parser.add_argument("--save-checkpoint", type=Path, help="Atomically save the settled pass.")
    parser.add_argument("--resume", type=Path, help="Continue a validated checkpoint with a new explicit limits file.")
    parser.add_argument("--verification-limits-file", type=Path, help="Authorize one separate semantic verification pass.")
    parser.add_argument("--goal", help="Targeted reinvestigation goal with --resume and optional repeated --company.")
    parser.add_argument("--fail-amd-finish", action="store_true", help="Reproduce two invalid AMD finishes offline.")
    args = parser.parse_args(argv)
    if args.resume and not args.limits_file:
        parser.error("--resume requires --limits-file with a newly authorized allowance.")
    if args.resume and (args.term or args.fixed_terms):
        parser.error("A continuation uses saved discovery settings.")
    if args.goal and not args.resume:
        parser.error("--goal requires --resume.")
    if args.fail_amd_finish and (args.live or not COMPARATIVE_PROFILES[args.profile].comparative_capture):
        parser.error("--fail-amd-finish requires an offline profile with comparative capture.")
    if args.live and not args.limits_file:
        parser.error("--live requires --limits-file with explicit overall allowances.")

    profile = COMPARATIVE_PROFILES[args.profile]
    shared = profile.shared_limits
    try:
        if args.limits_file:
            limits = CorpusLimits.model_validate_json(args.limits_file.read_bytes())
        elif profile.comparative_capture:
            limits = CorpusLimits(
                max_batches=13, max_provider_attempts=13 * shared.max_provider_attempts,
                max_reported_tokens=13 * shared.max_reported_tokens, max_corpus_attempts=13 * shared.max_corpus_attempts,
                max_delegations=25, max_partitions=4 if args.limited else 50,
                deadline_seconds=13 * shared.deadline_seconds,
            )
        else:
            limits = CorpusLimits(
                max_batches=13, max_provider_attempts=shared.max_provider_attempts,
                max_reported_tokens=shared.max_reported_tokens, max_corpus_attempts=shared.max_corpus_attempts + 2,
                max_delegations=26, max_partitions=shared.max_partitions if args.limited else 52, deadline_seconds=180.0,
            )
        verification_limits = (VerificationLimits.model_validate_json(args.verification_limits_file.read_bytes())
                               if args.verification_limits_file else None)
        corpus = CorpusEnvironment()

        def interpret(question, schema):
            intent = scripted_interpretation(question, schema)
            if args.company:
                intent.update(company_selection="explicit", companies=args.company)
            return intent

        question = QUESTION if not args.company else QUESTION.replace("Which companies", "Which requested companies", 1)
        checkpoint = load_checkpoint(args.resume, corpus) if args.resume else None
        plan = checkpoint.plan if checkpoint else resolve_research_plan(interpret_intent(question, interpret), corpus)
        factory = ((lambda _: BedrockRootClient.from_env(disable_retries=True)) if args.live else
                   _offline_factory(corpus, plan, compact=profile.comparative_capture, fail_amd_finish=args.fail_amd_finish))
        runner = RecursiveResearch(
            corpus, factory, shared_limits=shared, local_limits=profile.local_limits,
            finalization=profile.finalization, child_guidance=GUIDANCE,
            max_root_reinvestigation_calls=profile.max_root_reinvestigation_calls,
            comparative_capture=profile.comparative_capture,
        )
        result = None
        if checkpoint:
            checkpoint = continue_research(checkpoint, runner, limits=limits, max_groups=profile.max_groups,
                                           companies=tuple(args.company) if args.company else None, goal=args.goal)
        else:
            scan_terms = tuple(args.term or TERMS) if args.fixed_terms else None
            discovery = None if args.fixed_terms else DiscoverySpec.from_intent(
                plan.intent, tuple(args.term or DISCOVERY_PHRASES),
            )
            result = execute_corpus_research(plan, runner, limits=limits, max_groups=profile.max_groups,
                                             scan_terms=scan_terms, discovery=discovery, root_guidance=ROOT_GUIDANCE,
                                             detailed_trace=args.trace)
            if args.save_checkpoint or verification_limits or profile.comparative_capture:
                checkpoint = create_checkpoint(corpus, result, discovery=discovery, scan_terms=scan_terms or ())
        if args.save_checkpoint:
            save_checkpoint(args.save_checkpoint, checkpoint)
        if verification_limits:
            verifier = (BedrockSemanticVerifier.from_env(disable_retries=True) if args.live else
                        BedrockSemanticVerifier(_OfflineVerifierTransport(), config=BedrockConfig(model="offline-fixture")))
            checkpoint = review_checkpoint(checkpoint, corpus, verifier, limits=verification_limits)
            if args.save_checkpoint:
                save_checkpoint(args.save_checkpoint, checkpoint)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "mode": "live Bedrock" if args.live else "offline scripted",
        "assessment_source": "model-reported" if args.live else "scripted example with other companies unassessed",
        "profile": args.profile, "tree_configuration": profile, "required_targets": plan.required_target_count,
        "resolved_targets": len(plan.resolved_targets), "independent_semantic_verification": bool(checkpoint and checkpoint.verification_runs),
        "group_counts": result.group_counts if result else None,
        "result": result, "final_report": verification_report(checkpoint) if checkpoint else None,
        "coverage": checkpoint.merged_coverage(corpus) if checkpoint else result.coverage,
    }
    print(json.dumps(trace_data(json.loads(observation_json(report))), indent=2))
    return 0 if (checkpoint or result).execution_status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
