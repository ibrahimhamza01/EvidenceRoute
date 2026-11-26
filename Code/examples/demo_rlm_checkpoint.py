"""Save and resume a two-company SEC investigation with scripted local decisions."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evidenceroute.research.rlm import CorpusEnvironment
from evidenceroute.research.rlm.budget import SharedLimits
from evidenceroute.research.rlm.checkpoint import continue_research, create_checkpoint, load_checkpoint, plan_continuation, save_checkpoint
from evidenceroute.research.rlm.controller import ScriptedModelClient
from evidenceroute.research.rlm.corpus_research import CorpusLimits, execute_corpus_research
from evidenceroute.research.rlm.discovery import DiscoverySpec
from evidenceroute.research.rlm.planning import ResearchIntent, resolve_research_plan
from evidenceroute.research.rlm.recursive import RecursiveResearch
from evidenceroute.research.rlm.tools import observation_json


QUESTION = "What FY2024 Item 1A disclosures describe risks involving foreign-currency costs or constrained component supply?"
DISCLOSURES = {
    "AAPL": ("cd7988623c9a3bf545f06045", "1099:1275",
             "Stronger foreign currencies may increase component costs and adversely affect gross margins."),
    "AMZN": ("4d1474550060251037c5f981", "16:200",
             "Amazon relies on a limited group of semiconductor suppliers, including suppliers of GPUs."),
}
AMAZON_GAP = ("4d1474550060251037c5f981", "201:369",
              "Constraints on these products could impair Amazon's ability to develop and operate AI technologies.")


def _tool(name, **arguments):
    return {"type": "tool", "name": name, "arguments": arguments}


def _submit(disclosure):
    chunk, span, claim = disclosure
    return [
        _tool("get_spans", chunk_id=chunk, span_ids=[span]),
        _tool("submit_findings", findings=[{"claim": claim, "analysis_status": "analyzed",
                                           "evidence": [{"chunk_id": chunk, "span_id": span}]}]),
    ]


def _finish(*refs):
    return {"type": "synthesis", "synthesis": {"cite_refs": list(refs), "analysis_status": "incomplete"}}


def _runner(corpus, *, continuation=False):
    def factory(invocation):
        ticker = corpus.get_filing(invocation.partitions[0].document_id).ticker
        if invocation.depth == 0:
            return ScriptedModelClient([
                _tool("delegate_research", question=QUESTION, goal="Find scoped disclosures and record supporting spans.",
                      partitions=[asdict(pair) for pair in invocation.partitions]),
                _finish(*(("d1", "d2") if continuation else ("d1",))),
            ])
        if continuation:
            return ScriptedModelClient([
                _tool("read_prior_findings", limit=1), *_submit(AMAZON_GAP), _finish("d1", "s1"),
            ])
        return ScriptedModelClient([*_submit(DISCLOSURES[ticker]), _finish("s2" if ticker == "AMZN" else "s1")])

    return RecursiveResearch(corpus, factory, shared_limits=SharedLimits(
        max_depth=1, max_delegations=1, max_partitions=1, max_provider_attempts=6, max_corpus_attempts=5,
    ))


def _limits(batches):
    return CorpusLimits(max_batches=batches, max_provider_attempts=6 * batches, max_reported_tokens=60_000 * batches,
                        max_corpus_attempts=5 * batches, max_delegations=batches, max_partitions=batches,
                        deadline_seconds=60.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--continued-checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    corpus = CorpusEnvironment()
    intent = ResearchIntent(question=QUESTION, company_selection="explicit", companies=("AAPL", "AMZN"),
                            fiscal_years=(2024,), section_ids=("item_1a",), objective="review",
                            topic="FX costs and component supply", comprehensive=False, inferred_fields=())
    plan = resolve_research_plan(intent, corpus)
    discovery = DiscoverySpec.from_intent(intent, ("foreign currency", "semiconductor products"))
    result = execute_corpus_research(plan, _runner(corpus), limits=_limits(2), max_groups=1, discovery=discovery)
    initial = create_checkpoint(corpus, result)
    save_checkpoint(args.initial_checkpoint, initial)
    loaded = load_checkpoint(args.initial_checkpoint, corpus)
    selection = plan_continuation(loaded)
    updated = continue_research(loaded, _runner(corpus, continuation=True), limits=_limits(1),
                                goal="Check how constrained semiconductor availability could affect AI operations.")
    save_checkpoint(args.continued_checkpoint, updated)
    restored = load_checkpoint(args.continued_checkpoint, corpus)
    coverage = restored.merged_coverage(corpus)
    report = {
        "initial_execution_status": initial.execution_status,
        "continued_execution_status": restored.execution_status,
        "research_status": restored.research_status,
        "targeted_companies": [group.ticker for group in selection.groups],
        "required_target_count": coverage.required_target_count,
        "findings": restored.findings,
        "citations": restored.citations,
        "historical_usage": restored.historical_usage,
        "current_usage": restored.current_usage,
        "cumulative_usage": restored.cumulative_usage,
        "live_provider_attempts": restored.cumulative_usage.provider_attempts,
    }
    print(json.dumps(json.loads(observation_json(report)), indent=2))
    return 0 if restored.execution_status == "finished" else 1


if __name__ == "__main__":
    raise SystemExit(main())
