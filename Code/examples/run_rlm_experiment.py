"""Preflight or explicitly authorize one recorded RLM experiment configuration."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CATALOG = HERE / "rlm_experiments.json"


def git(root, *args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), *args], text=True).strip()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def sha256_file(path):
    return sha256(path.read_bytes()).hexdigest()


def output_directory(path, source_root):
    directory = path.resolve()
    if source_root != ROOT and directory.is_relative_to(source_root):
        raise ValueError("Use an output directory outside the historical checkout.")
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / ".rlm-runs"):
        raise ValueError("Use an output directory outside the repository or under .rlm-runs/.")
    if directory == ROOT / ".rlm-runs":
        raise ValueError("Use a new per-run directory inside .rlm-runs/.")
    return directory


def provider_environment(provider):
    env = os.environ.copy()
    env.update(BEDROCK_MODEL_ID=provider["model"], BEDROCK_REGION=provider["region"],
               CLAUDE_TIMEOUT_SECONDS=str(provider["timeout_seconds"]),
               CLAUDE_MAX_OUTPUT_TOKENS=str(provider["max_output_tokens"]),
               CLAUDE_MAX_RETRIES=str(provider["max_retries"]))
    for key in ("temperature", "top_p"):
        name = "CLAUDE_" + key.upper()
        if provider[key] is None:
            env.pop(name, None)
        else:
            env[name] = str(provider[key])
    return env


def question_setup(config):
    from evidenceroute.research.rlm.budget import FinalizationPolicy, SharedBudget, SharedLimits
    from evidenceroute.research.rlm.controller import RootLimits
    from evidenceroute.research.rlm.corpus_research import CorpusLimits
    from evidenceroute.research.rlm.interpretation import InterpretationLimits

    overall = CorpusLimits.model_validate(config["overall_limits"])
    interpretation = InterpretationLimits.model_validate(config["interpretation_limits"])
    shared = SharedLimits.model_validate(config["shared_limits"])
    local = RootLimits.model_validate(config["local_limits"])
    finalization = FinalizationPolicy.model_validate(config["finalization"])
    budget = SharedBudget(shared, finalization=finalization)
    with budget.invocation_allocation(root=True):
        if budget.active_allocation.investigation_left < config["max_groups"]:
            raise ValueError("The tree allowance cannot admit the configured number of groups.")
    if (shared.max_provider_attempts + interpretation.max_provider_attempts > overall.max_provider_attempts
            or shared.max_reported_tokens + interpretation.max_tokens > overall.max_reported_tokens):
        raise ValueError("The overall allowance does not cover interpretation and the configured tree.")
    return overall, interpretation, shared, local, finalization


def run_question(config, corpus, setup, provider, directory):
    import boto3

    from evidenceroute.research.rlm.bedrock import BedrockConfig, BedrockConverseTransport, BedrockRootClient
    from evidenceroute.research.rlm.checkpoint import create_checkpoint, save_checkpoint
    from evidenceroute.research.rlm.interpretation import BedrockIntentInterpreter
    from evidenceroute.research.rlm.natural_language import research_question
    from evidenceroute.research.rlm.recursive import RecursiveResearch
    from evidenceroute.research.rlm.tools import observation_json, trace_data

    overall, interpretation, shared, local, finalization = setup
    transport = BedrockConverseTransport(boto3.Session(), region=provider["region"])
    bedrock = BedrockConfig.model_validate(provider)
    runner = RecursiveResearch(
        corpus, lambda _: BedrockRootClient(transport, config=bedrock),
        shared_limits=shared, local_limits=local, finalization=finalization,
        max_root_reinvestigation_calls=config["max_root_reinvestigation_calls"],
        comparative_capture=config["comparative_capture"],
    )
    result = research_question(
        config["question"], runner, limits=overall, interpretation_limits=interpretation,
        interpreter_factory=lambda: BedrockIntentInterpreter(transport, config=bedrock),
        max_groups=config["max_groups"], detailed_trace=config["detailed_trace"],
    )
    report = {"mode": "live Bedrock", "independent_semantic_verification": False, "result": result}
    if hasattr(result, "response"):
        report["response"] = result.response
    write_json(directory / "report.json", trace_data(json.loads(observation_json(report))))
    if result.research is not None:
        save_checkpoint(directory / "checkpoint.json", create_checkpoint(corpus, result))
    return 0 if result.research is not None and result.research.execution_status == "finished" else 1


def corpus_setup(config, stage, limits):
    from demo_rlm_comparative import DISCOVERY_PHRASES, GUIDANCE, ROOT_GUIDANCE
    from demo_rlm_plan import QUESTION, scripted_interpretation
    from evidenceroute.research.rlm.corpus_research import CorpusLimits
    from evidenceroute.research.rlm.discovery import DiscoverySpec
    from evidenceroute.research.rlm.planning import ResearchIntent, interpret_intent
    from evidenceroute.research.rlm.profiles import COMPARATIVE_PROFILES
    from evidenceroute.research.rlm.verification_records import VerificationLimits

    profile = COMPARATIVE_PROFILES[config["profile"]]
    if profile.model_dump(mode="json") != config["tree_configuration"]:
        raise ValueError("The selected revision's demo profile differs from the recorded configuration.")
    intent = interpret_intent(QUESTION, scripted_interpretation)
    discovery = DiscoverySpec.from_intent(intent, DISCOVERY_PHRASES)
    if (intent != ResearchIntent.model_validate(config["intent"])
            or discovery != DiscoverySpec.model_validate_json(json.dumps(config["discovery"]))):
        raise ValueError("The selected revision's demo scope or discovery settings differ from the historical settings.")
    limit_type = VerificationLimits if stage == "corpus-verify" else CorpusLimits
    return limit_type.model_validate(limits).model_dump(mode="json"), {
        "intent": intent.model_dump(mode="json"), "discovery": discovery.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"), "child_guidance": GUIDANCE, "root_guidance": ROOT_GUIDANCE,
    }


def run_corpus_demo(stage, settings, source, directory, checkpoint, provider):
    verification = stage == "corpus-verify"
    limits_path = directory / "limits.json"
    write_json(limits_path, settings["pass_limits"])
    arguments = ["--live", "--limits-file", str(limits_path), "--save-checkpoint", str(directory / "checkpoint.json")]
    if verification:
        demo = "demo_rlm_verification.py"
        arguments += ["--outstanding", "--checkpoint", str(checkpoint)]
    else:
        demo = "demo_rlm_corpus_research.py"
        arguments += ["--profile", settings["profile_name"]]
        if settings["detailed_trace"]:
            arguments.append("--trace")
        if checkpoint:
            arguments += ["--resume", str(checkpoint)]
    env = provider_environment(provider)
    env["PYTHONPATH"] = str(source / "Code/src")
    with (directory / "report.json").open("x") as output, (directory / "stderr.log").open("x") as errors:
        return subprocess.run([sys.executable, str(source / "Code/examples" / demo), *arguments],
                              cwd=source, env=env, stdout=output, stderr=errors).returncode


def main(argv=None):
    catalog = read_json(CATALOG)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("limits", help="Print an archived limits preset as JSON.")
    export.add_argument("name", choices=tuple(catalog["limits"]))
    replay = commands.add_parser("replay", help="Publish a saved checkpoint locally with zero provider requests.")
    replay.add_argument("checkpoint", type=Path)
    replay.add_argument("--source-root", type=Path, default=ROOT)
    run = commands.add_parser("run", help="Validate settings and local inputs. Add --live to execute one pass.")
    stages = catalog["full_corpus"]["stages"]
    run.add_argument("experiment", choices=(*catalog["questions"], *stages))
    run.add_argument("--source-root", type=Path, default=ROOT, help="Use this Git checkout's code and full-v1 corpus.")
    run.add_argument("--checkpoint", type=Path, help="Input checkpoint for continuation or outstanding review.")
    run.add_argument("--output-dir", type=Path, help="New per-run directory, required with --live.")
    run.add_argument("--live", action="store_true", help="Authorize this pass's Bedrock requests.")
    args = parser.parse_args(argv)
    if args.command == "limits":
        print(json.dumps(catalog["limits"][args.name], indent=2))
        return 0
    if args.command == "run":
        stage = args.experiment
        if (stage in {"corpus-continue", "corpus-verify"}) != bool(args.checkpoint):
            parser.error("--checkpoint is required only for corpus-continue and corpus-verify.")
        if args.live != bool(args.output_dir):
            parser.error("--live and --output-dir must be supplied together.")

    source = args.source_root.resolve()
    try:
        if Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source:
            raise ValueError("--source-root must be a Git checkout root.")
        if not (source / "Code/src/evidenceroute/research/rlm").is_dir():
            raise ValueError("The selected checkout has no RLM implementation.")
        sys.path[:0] = [str(source / "Code/src"), str(source / "Code/examples")]
        from evidenceroute.research.rlm import CorpusEnvironment
        from evidenceroute.research.rlm.bedrock import BedrockConfig
        from evidenceroute.research.rlm.checkpoint import load_checkpoint
        from evidenceroute.research.rlm.tools import observation_json

        revision = git(source, "rev-parse", "HEAD")
        dirty = bool(git(source, "status", "--porcelain", "--untracked-files=no", "--", "Code/src", "Code/examples"))
        corpus = CorpusEnvironment()
        if corpus.snapshot_identity != catalog["corpus_identity"]:
            raise ValueError("The local corpus identity does not match the recorded full-v1 snapshot.")
        input_path = args.checkpoint.resolve() if args.checkpoint else None
        previous = load_checkpoint(input_path, corpus) if input_path else None
        if args.command == "replay":
            replayed = {
                "mode": "offline checkpoint replay", "provider_attempts": 0,
                "git_head": revision, "runtime_tracked_changes": dirty, "corpus_identity": corpus.snapshot_identity,
                "checkpoint": {"name": input_path.name, "sha256": sha256_file(input_path)},
                "recorded_usage": previous.combined_usage, "publication": previous.publication,
            }
            print(json.dumps(json.loads(observation_json(replayed)), indent=2, ensure_ascii=False))
            return 0
        provider = BedrockConfig.model_validate(catalog["provider"]).model_dump(mode="json")
        config = catalog["questions"].get(stage)
        if config:
            setup = question_setup(config)
            settings = {key: value for key, value in config.items()
                        if key not in {"git_head", "model", "region", "max_retries", "corpus_identity"}}
            budget_fields = ("overall_limits", "interpretation_limits", "shared_limits", "local_limits", "finalization")
            settings.update((field, limits.model_dump(mode="json")) for field, limits in zip(budget_fields, setup))
            if (config["model"], config["region"], config["max_retries"], config["corpus_identity"]) != (
                provider["model"], provider["region"], provider["max_retries"], corpus.snapshot_identity,
            ):
                raise ValueError("The question preset's provider or corpus differs from the catalog.")
            historical_head = config["git_head"]
            policy = "none"
        else:
            suite = catalog["full_corpus"]
            stage_config = stages[stage]
            limits, effective = corpus_setup(suite, stage, catalog["limits"][stage_config["limits"]])
            settings = {"pass_limits": limits, "profile_name": suite["profile"],
                        "detailed_trace": stage_config.get("detailed_trace", False), **effective}
            historical_head = suite["historical_git_head"]
            policy = stage_config.get("verification_policy", "none")
            if previous and (previous.plan.intent.model_dump(mode="json") != effective["intent"]
                             or previous.discovery is None
                             or previous.discovery.model_dump(mode="json") != effective["discovery"]):
                raise ValueError("The input checkpoint is not from the configured full-corpus experiment.")
        metadata = {
            "mode": "live Bedrock" if args.live else "offline preflight",
            "experiment": stage, "git_head": revision, "runtime_tracked_changes": dirty,
            "historical_git_head": historical_head,
            "revision_matches_historical_reference": revision == historical_head and not dirty,
            "harness_sha256": sha256_file(Path(__file__)), "catalog_sha256": sha256_file(CATALOG),
            "corpus_identity": corpus.snapshot_identity,
            "provider": provider, "sdk_total_max_attempts": 1, "verification_policy": policy,
            "settings": settings,
            "input_checkpoint": {"name": input_path.name, "sha256": sha256_file(input_path)} if input_path else None,
            "python": sys.version.split()[0],
            "dependencies": {name: version(name) for name in ("boto3", "botocore", "pydantic", "pyarrow")},
        }
        if not args.live:
            print(json.dumps({**metadata, "provider_attempts": 0}, indent=2))
            return 0
        directory = output_directory(args.output_dir, source)
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        metadata["started_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "configuration.json", metadata)
        started = monotonic()
        exit_code = None
        try:
            exit_code = (run_question(config, corpus, setup, provider, directory) if config
                         else run_corpus_demo(stage, settings, source, directory, input_path, provider))
        finally:
            wall = monotonic() - started
            checkpoint_path = directory / "checkpoint.json"
            report_path = directory / "report.json"
            summary = {"exit_code": exit_code, "elapsed_wall_seconds": wall,
                       "report": report_path.name if report_path.is_file() and report_path.stat().st_size else None,
                       "checkpoint": checkpoint_path.name if checkpoint_path.is_file() else None}
            if checkpoint_path.is_file():
                checkpoint = load_checkpoint(checkpoint_path, corpus)
                runs = checkpoint.verification_runs if stage == "corpus-verify" else checkpoint.runs
                start = (len(previous.verification_runs if stage == "corpus-verify" else previous.runs)
                         if previous else 0)
                summary["pass_usage"] = [item.usage for item in runs[start:]]
                summary["combined_usage"] = checkpoint.combined_usage
            if config and summary["report"]:
                result = read_json(report_path)["result"]
                summary["pass_usage"] = [result["usage"]]
            write_json(directory / "run-summary.json", json.loads(observation_json(summary)))
        print(directory)
        return exit_code
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
