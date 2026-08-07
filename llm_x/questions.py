"""Validated access to the task and question variants used by legacy evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import feedback, task
from .data import Episode


METRIC_POOLS = {
    "next-action": "NextActionPrediction_CLS",
    "last-action": "LastActionPrediction_CLS",
    "next-state": "NextStatePrediction_CLS",
    "last-state": "LastStatePrediction_CLS",
    "argue-action": "ArgueAction_CLS",
}


def available_questions() -> dict[str, list[str]]:
    return {
        metric: [cls.question_name for cls in feedback.QUEST_CLS_POOL[pool]]
        for metric, pool in METRIC_POOLS.items()
    }


def resolve_task(task_name: str) -> object:
    task_cls = task.get_task_cls(task_name)
    if task_cls is None:
        known = ", ".join(cls.task_name for cls in task.ALL_CLS)
        raise ValueError(f"Unknown task {task_name!r}. Available tasks: {known}")
    return task_cls()


def resolve_question(metric: str, question_name: str) -> object:
    try:
        pool = METRIC_POOLS[metric]
    except KeyError as exc:
        raise ValueError(f"Unknown metric family: {metric}") from exc
    question_cls = feedback.get_quest_cls(pool, question_name)
    if question_cls is None:
        known = ", ".join(available_questions()[metric])
        raise ValueError(f"Question {question_name!r} is not valid for {metric}. Available: {known}")
    return question_cls()


def render_question(
    metric: str,
    question: object,
    episode: Episode,
    *,
    index: int,
    drop_last_feature: bool,
    presented_action: int | None = None,
) -> str:
    state = _text(episode.state(index, drop_last_feature=drop_last_feature))
    action_dim = int(episode.action_vector(index).size)

    if metric == "next-action":
        if "continuous" in question.question_name:
            return question.render(i=index, action_dim=action_dim, state=state)
        return question.render(i=index, state=state)

    if metric == "last-action":
        next_state = _text(episode.state(index + 1, drop_last_feature=drop_last_feature))
        if "continuous" in question.question_name:
            return question.render(
                i=index,
                k=index + 1,
                action_dim=action_dim,
                state=state,
                next_state=next_state,
            )
        return question.render(i=index, k=index + 1, state=state, next_state=next_state)

    if metric == "next-state":
        return question.render(
            j=index,
            i=index + 1,
            state=state,
            action=_text(episode.arrays["actions"][index]),
            reward=episode.reward(index),
        )

    if metric == "last-state":
        return question.render(
            j=index - 1,
            i=index,
            k=index + 1,
            action=_text(episode.arrays["actions"][index]),
            next_state=_text(episode.state(index + 1, drop_last_feature=drop_last_feature)),
        )

    if metric == "argue-action":
        if presented_action is None:
            raise ValueError("argue-action requires a presented action")
        return question.render(i=index, state=state, action=presented_action)

    raise ValueError(f"Unsupported metric family: {metric}")


def _text(value: Any) -> str:
    return np.array2string(np.asarray(value), precision=4, separator=", ")
