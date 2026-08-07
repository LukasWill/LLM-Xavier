"""Consolidated offline evaluation for the five paper evaluation families."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .backends import ChatBackend
from .data import Episode, EpisodeValidationError
from .metrics import (
    ParsedValue,
    bin_actions,
    element_accuracy,
    parse_direction_predictions,
    parse_discrete_action,
    parse_numeric_predictions,
    parse_vote,
    state_directions,
)
from .prompting import system_prompt, user_prompt
from .questions import render_question, resolve_question, resolve_task


METRICS = ("next-action", "last-action", "next-state", "last-state", "argue-action")


@dataclass(frozen=True)
class EvaluationConfig:
    task_name: str
    metric: str
    question_name: str
    history_size: int
    seed: int = 0
    indexed_history: bool = True
    include_prompts: bool = True
    state_threshold: float = 1e-4
    fetch_drop_last_feature: bool = True
    action_min: float | None = None
    action_max: float | None = None
    action_bins: int = 10
    max_queries: int | None = None
    max_steps: int | None = None
    semantics_version: str = "public-v1"

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.history_size < 0:
            raise ValueError("history_size must be non-negative")
        if self.state_threshold < 0:
            raise ValueError("state_threshold must be non-negative")
        if self.max_queries is not None and self.max_queries < 1:
            raise ValueError("max_queries must be positive")
        if self.max_steps is not None and self.max_steps < 2:
            raise ValueError("max_steps must be at least two")
        if (self.action_min is None) != (self.action_max is None):
            raise ValueError("action_min and action_max must be provided together")


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    system_prompt: str
    metrics: dict[str, Any]
    records: list[dict[str, Any]]


def evaluate_episode(
    episode: Episode,
    config: EvaluationConfig,
    backend: ChatBackend,
) -> EvaluationResult:
    """Evaluate one episode without writing raw model responses to disk."""

    task = resolve_task(config.task_name)
    question = resolve_question(config.metric, config.question_name)
    scene = system_prompt(task)
    drop_last = config.fetch_drop_last_feature and "Fetch" in config.task_name
    usable_length = min(episode.length, config.max_steps or episode.length)
    indices = _query_indices(config.metric, config.history_size, usable_length)
    if config.max_queries is not None:
        indices = indices[: config.max_queries]
    if not indices:
        raise EpisodeValidationError(
            f"No queries are available for length={usable_length}, metric={config.metric}, "
            f"history_size={config.history_size}"
        )

    argue_inputs = _argue_inputs(episode, indices, config.seed) if config.metric == "argue-action" else {}
    records: list[dict[str, Any]] = []
    for index in indices:
        presented_action, presented_is_correct = argue_inputs.get(index, (None, None))
        question_text = render_question(
            config.metric,
            question,
            episode,
            index=index,
            drop_last_feature=drop_last,
            presented_action=presented_action,
        )
        history_start, history_end = _history_range(config.metric, index, config.history_size)
        prompt = user_prompt(
            episode,
            history_start=history_start,
            history_end=history_end,
            indexed_history=config.indexed_history,
            drop_last_state_feature=drop_last,
            question=question_text,
        )
        response = backend.complete(system_prompt=scene, user_prompt=prompt)
        record = _score_response(
            episode,
            config,
            index=index,
            response=response,
            drop_last_feature=drop_last,
            presented_action=presented_action,
            presented_is_correct=presented_is_correct,
        )
        record["history_start"] = history_start
        record["history_end_exclusive"] = history_end
        if config.include_prompts:
            record["prompt"] = prompt
        records.append(record)

    return EvaluationResult(
        config=config,
        system_prompt=scene if config.include_prompts else "",
        metrics=_summarize(records, config.metric),
        records=records,
    )


def _query_indices(metric: str, history_size: int, length: int) -> list[int]:
    # These ranges make the legacy script indexing explicit. Action and argue queries
    # use H+1 historical records. Next-state includes the current state-action tuple.
    if metric in {"next-action", "argue-action"}:
        return list(range(history_size + 1, length))
    if metric in {"last-action", "last-state"}:
        return list(range(history_size + 1, length - 1))
    if metric == "next-state":
        return list(range(history_size, length - 1))
    raise ValueError(f"Unsupported metric: {metric}")


def _history_range(metric: str, index: int, history_size: int) -> tuple[int, int]:
    end = index + 1 if metric == "next-state" else index
    start = end - (history_size + 1)
    if start < 0:
        raise IndexError("Query plan produced a negative history index")
    return start, end


def _score_response(
    episode: Episode,
    config: EvaluationConfig,
    *,
    index: int,
    response: str,
    drop_last_feature: bool,
    presented_action: int | None,
    presented_is_correct: bool | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"index": index}
    metric = config.metric

    if metric in {"next-action", "last-action"}:
        if "continuous" in config.question_name:
            expected_values = episode.action_vector(index).astype(float).tolist()
            start, stop = _action_range(config)
            expected = bin_actions(expected_values, start=start, stop=stop, bins=config.action_bins)
            with_bins = "no_bins" not in config.question_name
            parsed = parse_numeric_predictions(response, len(expected), integers=with_bins)
            if parsed.ok:
                actual = (
                    parsed.value
                    if with_bins
                    else bin_actions(parsed.value, start=start, stop=stop, bins=config.action_bins)
                )
                record["prediction"] = actual
                record["ground_truth"] = expected
                record["element_accuracy"] = element_accuracy(expected, actual)
                record["status"] = "match" if actual == expected else "mismatch"
            else:
                _ignored(record, parsed, expected)
            return record

        expected_action = episode.discrete_action(index)
        parsed = parse_discrete_action(response)
        if parsed.ok:
            record.update(
                prediction=parsed.value,
                ground_truth=expected_action,
                status="match" if parsed.value == expected_action else "mismatch",
            )
        else:
            _ignored(record, parsed, expected_action)
        return record

    if metric in {"next-state", "last-state"}:
        allow_unchanged = "more_options" in config.question_name
        expected = state_directions(
            episode.state_vector(index, drop_last_feature=drop_last_feature),
            episode.state_vector(index + 1, drop_last_feature=drop_last_feature),
            threshold=config.state_threshold,
            allow_unchanged=allow_unchanged,
        )
        parsed = parse_direction_predictions(
            response,
            len(expected),
            allow_unchanged=allow_unchanged,
        )
        if parsed.ok:
            record["prediction"] = parsed.value
            record["ground_truth"] = expected
            record["element_accuracy"] = element_accuracy(expected, parsed.value)
            record["status"] = "match" if parsed.value == expected else "mismatch"
        else:
            _ignored(record, parsed, expected)
        return record

    if metric == "argue-action":
        if presented_action is None or presented_is_correct is None:
            raise ValueError("argue-action scoring requires a deterministic presented action")
        parsed = parse_vote(response)
        record["presented_action"] = presented_action
        record["presented_action_is_correct"] = presented_is_correct
        if parsed.ok:
            record["prediction"] = parsed.value
            record["ground_truth"] = presented_is_correct
            record["status"] = "match" if parsed.value == presented_is_correct else "mismatch"
        else:
            _ignored(record, parsed, presented_is_correct)
        return record

    raise ValueError(f"Unsupported metric: {metric}")


def _ignored(record: dict[str, Any], parsed: ParsedValue, expected: Any) -> None:
    record.update(status="ignored", parse_error=parsed.error, ground_truth=expected)


def _action_range(config: EvaluationConfig) -> tuple[float, float]:
    if config.action_min is not None and config.action_max is not None:
        return config.action_min, config.action_max
    return (-1.0, 1.0) if "Fetch" in config.task_name else (-2.0, 2.0)


def _argue_inputs(
    episode: Episode,
    indices: Sequence[int],
    seed: int,
) -> dict[int, tuple[int, bool]]:
    actions = sorted({episode.discrete_action(index) for index in range(episode.length)})
    if len(actions) < 2:
        raise EpisodeValidationError("argue-action requires at least two distinct recorded actions")
    rng = random.Random(seed)
    correct_schedule = [position % 2 == 0 for position in range(len(indices))]
    rng.shuffle(correct_schedule)
    result: dict[int, tuple[int, bool]] = {}
    for index, use_correct in zip(indices, correct_schedule):
        correct = episode.discrete_action(index)
        if use_correct:
            result[index] = (correct, True)
        else:
            alternatives = [action for action in actions if action != correct]
            result[index] = (rng.choice(alternatives), False)
    return result


def _summarize(records: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    query_count = len(records)
    parsed = [record for record in records if record["status"] != "ignored"]
    matches = [record for record in parsed if record["status"] == "match"]
    summary: dict[str, Any] = {
        "query_count": query_count,
        "parsed_count": len(parsed),
        "ignored_count": query_count - len(parsed),
        "exact_matches": len(matches),
        "parse_rate": len(parsed) / query_count,
        "exact_match_rate_all_queries": len(matches) / query_count,
        "exact_match_rate_parsed_queries": len(matches) / len(parsed) if parsed else None,
    }
    element_records = [record for record in parsed if "element_accuracy" in record]
    summary["mean_element_accuracy_parsed_queries"] = (
        sum(record["element_accuracy"] for record in element_records) / len(element_records)
        if element_records
        else None
    )
    # Legacy scripts used all queries for action matching and parsed queries for state
    # matching and argument voting. Both transparent rates above remain authoritative.
    legacy_denominator = query_count if metric in {"next-action", "last-action"} else len(parsed)
    summary["legacy_compatible_match_rate"] = (
        len(matches) / legacy_denominator if legacy_denominator else None
    )
    if metric == "argue-action":
        yes_votes = sum(record.get("prediction") is True for record in parsed)
        summary["argue_for_rate_parsed_queries"] = yes_votes / len(parsed) if parsed else None
        summary["correct_action_argue_for"] = sum(
            record.get("ground_truth") is True and record.get("prediction") is True
            for record in parsed
        )
        summary["correct_action_argue_against"] = sum(
            record.get("ground_truth") is True and record.get("prediction") is False
            for record in parsed
        )
        summary["wrong_action_argue_for"] = sum(
            record.get("ground_truth") is False and record.get("prediction") is True
            for record in parsed
        )
        summary["wrong_action_argue_against"] = sum(
            record.get("ground_truth") is False and record.get("prediction") is False
            for record in parsed
        )
    return summary


def config_dict(config: EvaluationConfig) -> dict[str, Any]:
    return asdict(config)
