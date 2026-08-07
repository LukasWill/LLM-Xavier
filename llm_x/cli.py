"""Command-line interface for reproducible offline evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import __version__
from .backends import OllamaChatBackend, OpenAIChatBackend, SequenceBackend
from .data import Episode
from .evaluation import EvaluationConfig, METRICS, evaluate_episode
from .questions import available_questions


DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "indexed_history": True,
    "include_prompts": True,
    "state_threshold": 1e-4,
    "fetch_drop_last_feature": True,
    "action_bins": 10,
    "timeout": 60.0,
    "retries": 2,
    "api_key_env": "OPENAI_API_KEY",
    "overwrite": False,
    "dry_run": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-x", description="LLM-X offline RL-agent evaluator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-questions", help="list supported metric/question variants")

    inspect_parser = subparsers.add_parser("inspect-data", help="validate an episode and print its schema")
    inspect_parser.add_argument("--data-path", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one offline episode")
    evaluate.add_argument("--config", help="optional JSON or YAML configuration")
    evaluate.add_argument("--data-path")
    evaluate.add_argument("--task-name")
    evaluate.add_argument("--metric", choices=METRICS)
    evaluate.add_argument("--question-name")
    evaluate.add_argument("--history-size", type=int)
    evaluate.add_argument("--backend", choices=("fixture", "openai", "openai-compatible", "ollama"))
    evaluate.add_argument("--model")
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--fixture-responses")
    evaluate.add_argument("--endpoint")
    evaluate.add_argument("--api-key-env")
    evaluate.add_argument("--timeout", type=float)
    evaluate.add_argument("--retries", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--state-threshold", type=float)
    evaluate.add_argument("--action-min", type=float)
    evaluate.add_argument("--action-max", type=float)
    evaluate.add_argument("--action-bins", type=int)
    evaluate.add_argument("--max-queries", type=int)
    evaluate.add_argument("--max-steps", type=int)
    evaluate.add_argument(
        "--indexed-history",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    evaluate.add_argument(
        "--include-prompts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    evaluate.add_argument(
        "--fetch-drop-last-feature",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    evaluate.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    evaluate.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-questions":
            print(json.dumps(available_questions(), indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-data":
            episode = Episode.load(args.data_path)
            print(json.dumps(_episode_schema(episode), indent=2, sort_keys=True))
            return 0
        return _evaluate(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _evaluate(args: argparse.Namespace) -> int:
    values = _resolved_values(args)
    required = (
        "data_path",
        "task_name",
        "metric",
        "question_name",
        "history_size",
        "backend",
        "output_dir",
    )
    missing = [name.replace("_", "-") for name in required if values.get(name) is None]
    if missing:
        raise ValueError(f"missing required evaluation settings: {', '.join(missing)}")
    if values["backend"] != "fixture" and not values.get("model"):
        raise ValueError("model is required for non-fixture backends")
    if values["backend"] == "fixture" and not values.get("fixture_responses"):
        raise ValueError("fixture-responses is required for the fixture backend")
    if values["backend"] in {"openai-compatible", "ollama"} and not values.get("endpoint"):
        raise ValueError(f"endpoint is required for backend {values['backend']}")
    if values.get("endpoint"):
        from .backends import _reject_endpoint_credentials

        _reject_endpoint_credentials(values["endpoint"])

    episode = Episode.load(values["data_path"])
    config = EvaluationConfig(
        task_name=values["task_name"],
        metric=values["metric"],
        question_name=values["question_name"],
        history_size=values["history_size"],
        seed=values["seed"],
        indexed_history=values["indexed_history"],
        include_prompts=values["include_prompts"],
        state_threshold=values["state_threshold"],
        fetch_drop_last_feature=values["fetch_drop_last_feature"],
        action_min=values.get("action_min"),
        action_max=values.get("action_max"),
        action_bins=values["action_bins"],
        max_queries=values.get("max_queries"),
        max_steps=values.get("max_steps"),
    )
    # Resolve these before any remote request or output mutation.
    from .questions import resolve_question, resolve_task

    resolve_task(config.task_name)
    resolve_question(config.metric, config.question_name)
    if values["dry_run"]:
        if values["backend"] == "fixture":
            SequenceBackend.from_file(values["fixture_responses"])
        print(json.dumps(_safe_effective_config(values, config, episode), indent=2, sort_keys=True))
        return 0

    backend = _backend(values)
    result = evaluate_episode(episode, config, backend)
    output = Path(values["output_dir"]).expanduser().resolve()
    _write_result(output, result, episode, backend, overwrite=values["overwrite"])
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    return 0


def _backend(values: dict[str, Any]):
    if values["backend"] == "fixture":
        return SequenceBackend.from_file(values["fixture_responses"])
    if values["backend"] in {"openai", "openai-compatible"}:
        return OpenAIChatBackend(
            model=values["model"],
            api_key_env=values["api_key_env"],
            endpoint=values.get("endpoint"),
            timeout=values["timeout"],
            retries=values["retries"],
            compatible=values["backend"] == "openai-compatible",
        )
    return OllamaChatBackend(
        model=values["model"],
        endpoint=values["endpoint"],
        timeout=values["timeout"],
        retries=values["retries"],
    )


def _resolved_values(args: argparse.Namespace) -> dict[str, Any]:
    file_values = _load_config(args.config) if args.config else {}
    known = set(vars(args)) - {"command", "config"}
    unknown = sorted(set(file_values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    values = {**DEFAULTS, **file_values}
    for name in known:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    return values


def _load_config(path: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    text = source.read_text()
    if source.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML configuration requires PyYAML") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration must contain a top-level mapping")
    return value


def _write_result(output: Path, result, episode: Episode, backend, *, overwrite: bool) -> None:
    expected = ("run.json", "config.effective.json", "metrics.json", "predictions.jsonl")
    if output.exists() and not overwrite and any((output / name).exists() for name in expected):
        raise FileExistsError(f"Output already exists; pass --overwrite to replace files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    safe_config = {
        **asdict(result.config),
        "data_sha256": episode.sha256,
        "backend": backend.name,
        "model": backend.model,
    }
    run = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "llm_x_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "data_sha256": episode.sha256,
        "backend": backend.name,
        "model": backend.model,
        "raw_responses_stored": False,
        "system_prompt": result.system_prompt or None,
    }
    _atomic_json(output / "run.json", run)
    _atomic_json(output / "config.effective.json", safe_config)
    _atomic_json(output / "metrics.json", result.metrics)
    _atomic_text(
        output / "predictions.jsonl",
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in result.records),
    )


def _safe_effective_config(
    values: dict[str, Any], config: EvaluationConfig, episode: Episode
) -> dict[str, Any]:
    return {
        **asdict(config),
        "data_sha256": episode.sha256,
        "backend": values["backend"],
        "model": "fixture-sequence" if values["backend"] == "fixture" else values.get("model"),
        "output_directory_configured": bool(values.get("output_dir")),
    }


def _episode_schema(episode: Episode) -> dict[str, Any]:
    return {
        "sha256": episode.sha256,
        "length": episode.length,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sorted(episode.arrays.items())
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
