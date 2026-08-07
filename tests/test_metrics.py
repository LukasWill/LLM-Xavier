from __future__ import annotations

import pytest

from llm_x.metrics import (
    bin_actions,
    element_accuracy,
    parse_direction_predictions,
    parse_discrete_action,
    parse_numeric_predictions,
    parse_vote,
    state_directions,
)


def test_response_parsers() -> None:
    assert parse_discrete_action("reason\n>>Final action choice: [2]").value == 2
    assert parse_numeric_predictions("predictions = [-0.2, 1]", 2, integers=False).value == [-0.2, 1.0]
    assert parse_numeric_predictions("predictions = [2, 9]", 2, integers=True).value == [2, 9]
    assert parse_direction_predictions(
        'predictions = ["INC", "DEC", "UNCH"]', 3, allow_unchanged=True
    ).value == [1, 0, 2]
    assert parse_vote("Final vote is [False]").value is False
    assert not parse_discrete_action("no formatted answer").ok


def test_action_bin_edges_match_legacy_right_convention() -> None:
    assert bin_actions([-2.0, -1.6, 0.0, 2.0], start=-2, stop=2) == [0, 0, 4, 9]


def test_state_direction_and_element_accuracy() -> None:
    expected = state_directions(
        [0.0, 1.0, 2.0],
        [0.1, 0.9, 2.00001],
        threshold=1e-4,
        allow_unchanged=True,
    )
    assert expected == [1, 0, 2]
    assert element_accuracy(expected, [1, 1, 2]) == pytest.approx(2 / 3)
