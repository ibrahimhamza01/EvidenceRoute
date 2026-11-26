"""Review checkpointed SEC claims using offline fixtures or an authorized Bedrock invocation."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.bedrock import BedrockConfig
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.checkpoint import create_checkpoint, load_checkpoint, save_checkpoint
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import ResearchIntent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import observation_json
from evidenceroute.research.rlm.verification import (
    BedrockSemanticVerifier, VerificationLimits, record_verification, verification_report, verify_checkpoint,
)

APPLE_CHUNK = "cd7988623c9a3bf545f06045"
ALPHABET_EARLIER = "e09b0898eb296a45490cd615"
ALPHABET_LATER = "e0ec543a580b908a6602bba5"
ALPHABET_RECALLS = "0982db825d8177d57ab2c33e"
STRENGTHENING = (
    "When foreign currencies strengthen relative to the U.S. dollar, while generally beneficial to Apple's foreign "
    "currency-denominated sales and earnings, it could cause the Company to reduce international pricing or incur losses "
    "on foreign currency derivative instruments, thereby limiting the benefit, and may increase the cost of product "
    "components denominated in those currencies, adversely affecting gross margins."
)
HEDGING = (
    "Apple uses derivative instruments, such as foreign currency forward and option contracts, to hedge certain exposures "
    "to fluctuations in foreign exchange rates, but such hedging activities may not be effective to offset any, or more "
    "than a portion, of the adverse financial effects of unfavorable movements in foreign exchange rates over the "
    "limited time the hedges are in place."
)
LIMITED = "Apple's hedging activities may offset only a portion of adverse foreign-exchange effects."
GUARANTEED = "Apple's hedging always fully offsets adverse foreign-exchange effects."
SUPPLY = "Availability of components for technical infrastructure can delay data-center upgrades or expansions."
ACCELERATORS = "Alphabet describes competitive AI accelerator supply and warns that significant supply interruptions could delay data-center expansion."
RECALLS = "Defective products or recalls could harm Alphabet's business and financial results."


def _tool(name, **arguments):
    return {"type": "tool", "name": name, "arguments": arguments}


def _draft(claim, chunk_id, *spans):
    return {"claim": claim, "analysis_status": "analyzed",
            "evidence": [{"chunk_id": chunk_id, "span_id": span} for span in spans]}


def fixture_checkpoint(corpus, ticker):
    compare = ticker == "GOOGL"
    intent = ResearchIntent(
        question=("How did Alphabet describe infrastructure supply risks across FY2023 and FY2024?"
                  if compare else "What foreign exchange risks did Apple disclose in FY2024?"),
        company_selection="explicit", companies=(ticker,), section_ids=("item_1a",),
        fiscal_years=(2023, 2024) if compare else (2024,), objective="year_over_year" if compare else "review",
        topic="Infrastructure supply" if compare else "Foreign exchange", comprehensive=compare, inferred_fields=(),
    )
    plan = resolve_research_plan(intent, corpus)
    discovery = DiscoverySpec.from_intent(intent, ("data center", "AI accelerators") if compare else ("foreign exchange", "hedging"))

    def factory(invocation):
        if invocation.depth == 0:
            assign = _tool("delegate_research", question=intent.question, goal="Record the selected disclosure excerpts.",
                           partitions=[asdict(pair) for pair in invocation.partitions])
            if not compare:
                return ScriptedModelClient([assign, {"type": "synthesis", "synthesis": {"cite_refs": ["d9"]}}])
            comparisons = []
            for risk, same, earlier in (("Accelerator supply specificity", "uncertain", "contradicts"),
                                        ("Technical infrastructure supply constraints", "yes", "supports")):
                comparisons.append({"risk": risk, "same_risk": same, "assessments": [
                    {"ref": "d1", "relevance": "infrastructure", "entailment": earlier, "rationale": "Investigator judgment."},
                    {"ref": "d2", "relevance": "infrastructure", "entailment": "supports", "rationale": "Investigator judgment."},
                ]})
            comparisons.append({
                "risk": "Infrastructure supply interruptions versus product recalls", "same_risk": "no",
                "assessments": [
                    {"ref": "d1", "relevance": "infrastructure", "entailment": "supports", "rationale": "Infrastructure supply."},
                    {"ref": "d3", "relevance": "other", "entailment": "uncertain", "rationale": "Product quality and recalls."},
                ],
            })
            return ScriptedModelClient([assign, {"type": "synthesis", "synthesis": {"comparisons": comparisons}}])

        if compare:
            reads = [_tool("get_spans", chunk_id=ALPHABET_EARLIER, span_ids=["1304:1465"]),
                     _tool("get_spans", chunk_id=ALPHABET_LATER, span_ids=["417:578", "1504:1686"]),
                     _tool("get_spans", chunk_id=ALPHABET_RECALLS, span_ids=["454:671"])]
            drafts = [_draft(SUPPLY, ALPHABET_EARLIER, "1304:1465"),
                      _draft(ACCELERATORS, ALPHABET_LATER, "417:578", "1504:1686"),
                      _draft(RECALLS, ALPHABET_RECALLS, "454:671")]
        else:
            reads = [_tool("get_spans", chunk_id=APPLE_CHUNK, span_ids=["848:1098", "1442:1667"])]
            drafts = [_draft(STRENGTHENING, APPLE_CHUNK, "848:1098"),
                      *(_draft(claim, APPLE_CHUNK, "1442:1667") for claim in (HEDGING, LIMITED, GUARANTEED))]
        finish = {"type": "synthesis", "synthesis": {"cite_refs": [f"s{i}" for i in range(1, len(drafts) + 1)]}}
        return ScriptedModelClient([*reads, _tool("submit_findings", findings=drafts), finish])

    runner = RecursiveResearch(corpus, factory, shared_limits=SharedLimits(
        max_depth=1, max_provider_attempts=8, max_corpus_attempts=6, max_delegations=1, max_partitions=2,
    ))
    limits = CorpusLimits(max_batches=1, max_provider_attempts=8, max_reported_tokens=60_000, max_corpus_attempts=6,
                         max_delegations=1, max_partitions=2, deadline_seconds=60.0)
    result = execute_corpus_research(plan, runner, limits=limits, max_groups=1, discovery=discovery)
    return create_checkpoint(corpus, result)


def _component(statement, status, *evidence, gap=None):
    result = {"statement": statement, "status": status, "evidence": list(evidence)}
    if gap:
        result["gap"] = gap
    return result


def _fixture_review(case):
    claim = case["claim"]
    reply = {"ref": case["ref"]}
    if claim == STRENGTHENING:
        components = [
            _component("The trigger is foreign currency strengthening.", "insufficient_evidence",
                       gap="Select the sentence opening that identifies the currency movement."),
            _component("The move can benefit translated sales and earnings.", "supported", "e1"),
            _component("International price cuts or derivative losses can limit the benefit.", "supported", "e1"),
            _component("Stronger currencies increase component costs and hurt margins.", "insufficient_evidence",
                       gap="Select the separate component-cost and margin sentence."),
        ]
    elif claim == HEDGING:
        components = [
            _component("Apple uses forwards and options to hedge FX exposures.", "insufficient_evidence",
                       gap="Select text naming the derivative instruments and exposures."),
            _component("Hedging may offset none or only part of adverse effects during its term.", "supported", "e1"),
        ]
    elif claim == LIMITED:
        components = [_component("Hedging may offset only part of adverse FX effects.", "supported", "e1")]
    elif claim == GUARANTEED:
        components = [_component("Hedging always offsets all adverse FX effects.", "contradicted", "e1")]
    elif claim == SUPPLY:
        components = [_component("Supply interruptions may delay data-center upgrades or expansions.", "supported", "e1")]
    elif claim == ACCELERATORS:
        components = [_component("Accelerator supply is competitive.", "supported", "e2"),
                      _component("Significant supply interruptions can delay data-center expansion.", "supported", "e1")]
    elif claim == RECALLS:
        components = [_component("A recall could harm business or financial results.", "supported", "e1")]
    elif claim.startswith("Infrastructure supply interruptions versus product recalls."):
        components = [_component("These excerpts leave filing-wide novelty unresolved.", "supported", "e1", "e2")]
        reply["comparison"] = {
            "relationship": "different_risk", "relationship_evidence": ["e1", "e2"],
            "earlier_year": "nonmention", "earlier_evidence": ["e1"],
            "gap": "Review earlier product-quality and recall disclosures before drawing a novelty conclusion.",
        }
    elif case["kind"] == "comparison" and case["company"] == "GOOGL":
        conflict = claim.startswith("Accelerator supply specificity.")
        components = ([_component("The earlier wording opposes the later accelerator risk.", "insufficient_evidence", "e1",
                                  gap="Look for an earlier statement explicitly addressing accelerator supply.")]
                      if conflict else [_component("Both excerpts identify infrastructure supply interruption risk.",
                                                     "supported", "e1", "e2")])
        reply["comparison"] = {
            "relationship": "more_specific_wording", "relationship_evidence": ["e1", "e2", "e3"],
            "earlier_year": "nonmention" if conflict else "supports_risk", "earlier_evidence": ["e1"],
        }
        if conflict:
            reply["comparison"]["gap"] = "The earlier excerpt leaves accelerator-specific disclosure unresolved."
    else:
        return None
    return {**reply, "components": components}


class OfflineVerificationTransport:
    """Replay explicit assessments of the regression excerpts through native Converse."""

    def send(self, payload, *, timeout_seconds):
        cases = json.loads(payload["messages"][0]["content"][0]["text"])["cases"]
        reviews = [review for case in cases if (review := _fixture_review(case)) is not None]
        return {
            "stopReason": "tool_use",
            "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                "toolUseId": "fixture.verification", "name": "submit_verification", "input": {"reviews": reviews},
            }}]}},
            "usage": {"inputTokens": 800, "outputTokens": 240},
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize paid Bedrock verification requests.")
    parser.add_argument("--checkpoint", type=Path, help="Source research checkpoint.")
    parser.add_argument("--limits-file", type=Path, help="Explicit JSON verification limits.")
    parser.add_argument("--kind", choices=("all", "findings", "comparisons"), default="all", help="Cases to review.")
    parser.add_argument("--save-checkpoint", type=Path, help="Save the recorded review of a supplied checkpoint.")
    args = parser.parse_args(argv)
    if args.live and not (args.checkpoint and args.limits_file):
        parser.error("--live requires --checkpoint and --limits-file.")
    if args.save_checkpoint and not args.checkpoint:
        parser.error("--save-checkpoint requires --checkpoint.")
    corpus = CorpusEnvironment()
    try:
        if args.limits_file:
            with args.limits_file.open("rb") as source:
                raw = source.read(16_385)
            if len(raw) > 16_384:
                raise ValueError("Verification limits file exceeds 16 KiB.")
            limits = VerificationLimits.model_validate_json(raw)
        else:
            limits = VerificationLimits()
        inputs = ([load_checkpoint(args.checkpoint, corpus)] if args.checkpoint
                  else [fixture_checkpoint(corpus, ticker) for ticker in ("AAPL", "GOOGL")])
        reports = []
        for checkpoint in inputs:
            verifier = (BedrockSemanticVerifier.from_env(disable_retries=True) if args.live else
                        BedrockSemanticVerifier(OfflineVerificationTransport(), config=BedrockConfig(model="offline-verifier")))
            run = verify_checkpoint(checkpoint, corpus, verifier, limits=limits,
                                    finding_ids=() if args.kind == "comparisons" else None,
                                    comparison_ids=() if args.kind == "findings" else None)
            updated = record_verification(checkpoint, corpus, run)
            if args.save_checkpoint:
                save_checkpoint(args.save_checkpoint, updated)
            report = verification_report(updated)
            reports.append({**report, "status_counts": dict(Counter(item.status for item in report["items"]))})
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = {"mode": "live Bedrock" if args.live else "scripted verification fixtures", "reports": reports,
              "live_provider_attempts": sum(report["current_verification_usage"].provider_attempts for report in reports) if args.live else 0}
    print(json.dumps(json.loads(observation_json(output)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
