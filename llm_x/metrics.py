"""Response parsers and transparent metric primitives."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


_ACTION_PATTERN = re.compile(r"(?:>>\s*)?Final action choice\s*:\s*\[?(-?\d+)\]?", re.I)
_PREDICTION_PATTERN = re.compile(r"predictions\s*=\s*(\[[^\]\n]*\])", re.I)
_VOTE_PATTERN = re.compile(r"(?:final_vote\s*=\s*|Final vote is\s*)\[?(True|False)\]?", re.I)


@dataclass(frozen=True)
class ParsedValue:
    value: Any | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_discrete_action(text: str) -> ParsedValue:
    match = _ACTION_PATTERN.search(text)
    if not match:
        return ParsedValue(None, "final action marker not found")
    return ParsedValue(int(match.group(1)))


def parse_numeric_predictions(text: str, expected_size: int, *, integers: bool) -> ParsedValue:
    value = _literal_prediction_list(text)
    if not value.ok:
        return value
    if len(value.value) != expected_size:
        return ParsedValue(None, f"expected {expected_size} values, received {len(value.value)}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value.value):
        return ParsedValue(None, "predictions must contain only numbers")
    if integers and any(not float(item).is_integer() for item in value.value):
        return ParsedValue(None, "bin predictions must be integers")
    cast = int if integers else float
    return ParsedValue([cast(item) for item in value.value])


def parse_direction_predictions(
    text: str,
    expected_size: int,
    *,
    allow_unchanged: bool,
) -> ParsedValue:
    value = _literal_prediction_list(text)
    if not value.ok:
        tokens = re.findall(r'"(INC|DEC|UNCH)"', text.upper())
        value = ParsedValue(tokens) if tokens else value
    if not value.ok:
        return value
    if len(value.value) != expected_size:
        return ParsedValue(None, f"expected {expected_size} directions, received {len(value.value)}")
    allowed = {"INC", "DEC", "UNCH"} if allow_unchanged else {"INC", "DEC"}
    tokens = [str(item).upper() for item in value.value]
    if any(token not in allowed for token in tokens):
        return ParsedValue(None, f"directions must be one of {sorted(allowed)}")
    mapping = {"DEC": 0, "INC": 1, "UNCH": 2}
    return ParsedValue([mapping[token] for token in tokens])


def parse_vote(text: str) -> ParsedValue:
    match = _VOTE_PATTERN.search(text)
    if not match:
        return ParsedValue(None, "final vote marker not found")
    return ParsedValue(match.group(1).lower() == "true")


def bin_actions(
    values: Iterable[float],
    *,
    start: float,
    stop: float,
    bins: int = 10,
) -> list[int]:
    if not start < stop:
        raise ValueError("action bin start must be lower than stop")
    if bins < 2:
        raise ValueError("at least two action bins are required")
    array = np.asarray(list(values), dtype=float)
    edges = np.linspace(start, stop, bins + 1)
    indices = np.digitize(array, edges, right=True) - 1
    return np.clip(indices, 0, bins - 1).astype(int).tolist()


def state_directions(
    current: Sequence[float],
    following: Sequence[float],
    *,
    threshold: float,
    allow_unchanged: bool,
    decimals: int = 5,
) -> list[int]:
    first = np.round(np.asarray(current, dtype=float).reshape(-1), decimals=decimals)
    second = np.round(np.asarray(following, dtype=float).reshape(-1), decimals=decimals)
    if first.shape != second.shape:
        raise ValueError(f"state shapes differ: {first.shape} != {second.shape}")
    difference = second - first
    if allow_unchanged:
        return np.where(np.abs(difference) < threshold, 2, np.where(difference > 0, 1, 0)).tolist()
    return np.where(difference > 0, 1, 0).tolist()


def element_accuracy(expected: Sequence[Any], actual: Sequence[Any]) -> float:
    if len(expected) != len(actual):
        raise ValueError("expected and actual values must have the same length")
    if not expected:
        raise ValueError("cannot score an empty prediction")
    return sum(left == right for left, right in zip(expected, actual)) / len(expected)


def _literal_prediction_list(text: str) -> ParsedValue:
    match = _PREDICTION_PATTERN.search(text)
    if not match:
        return ParsedValue(None, "predictions list not found")
    try:
        value = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as exc:
        return ParsedValue(None, f"invalid predictions list: {exc}")
    if not isinstance(value, list):
        return ParsedValue(None, "predictions must be a list")
    return ParsedValue(value)
