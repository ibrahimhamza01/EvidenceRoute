"""Plan issue #16's broad SEC question locally using a scripted interpretation."""

import argparse
import json
from dataclasses import asdict

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.planning import interpret_intent, resolve_research_plan

QUESTION = "Which companies introduced a new AI-related infrastructure risk in FY2024 that was not present in FY2023?"


def scripted_interpretation(_question, _output_schema):
    return {
        "company_selection": "all",
        "fiscal_years": [2023, 2024],
        "section_ids": ["item_1a"],
        "objective": "year_over_year",
        "topic": "AI-related infrastructure risk. Compare the same risk across both years before inferring change.",
        "comprehensive": True,
        "inferred_fields": ["company_selection", "section_ids", "comprehensive"],
        "scope_notes": "Treat the corpus as the company universe and Item 1A as the risk-factor scope. "
                       "The question does not specify a filing item. A search miss does not establish absence.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-delegations", type=int, default=8)
    parser.add_argument("--max-partitions", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        limits = SharedLimits(**vars(args))
        corpus = CorpusEnvironment()
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    plan = resolve_research_plan(interpret_intent(QUESTION, scripted_interpretation), corpus)
    shortfalls = plan.delegation_shortfalls(limits)
    print(json.dumps({
        "mode": "offline planning demonstration",
        "evidence_assessed": False,
        "provider_attempts": 0,
        "intent": plan.intent.model_dump(mode="json"),
        "corpus_companies": corpus.list_companies(limit=1).total,
        "corpus_filings": corpus.list_filings(limit=1).total,
        "requested_companies": len(plan.requested_companies),
        "required_targets": plan.required_target_count,
        "resolved_targets": len(plan.resolved_targets),
        "missing_requirements": [asdict(entry) for entry in plan.missing_requirements],
        "company_partition_groups": len(plan.groups),
        "representative_groups": [{
            "ticker": group.ticker,
            "targets": [{**asdict(pair), "fiscal_year": corpus.get_filing(pair.document_id).fiscal_year}
                        for pair in group.targets],
        } for group in plan.groups[:3]],
        "example_delegation_admission": {
            "limits": vars(args),
            "fits": not shortfalls,
            "blocking_limits": shortfalls,
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
