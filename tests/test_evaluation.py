from __future__ import annotations

import json

import numpy as np

from llm_x.backends import SequenceBackend
from llm_x.data import Episode
from llm_x.evaluation import EvaluationConfig, evaluate_episode
from llm_x.questions import available_questions, render_question, resolve_question


def test_discrete_action_evaluation_does_not_retain_raw_responses(episode: Episode) -> None:
    marker = "PRIVATE_RESPONSE_MARKER"
    backend = SequenceBackend(
        [
            f"{marker}\n>>Final action choice: [1]",
            f"{marker}\n>>Final action choice: [0]",
            f"{marker}\nmalformed",
        ]
    )
    result = evaluate_episode(
        episode,
        EvaluationConfig(
            task_name="MountainCar-v0",
            metric="next-action",
            question_name="next_action_prediction",
            history_size=0,
        ),
        backend,
    )
    assert result.metrics["query_count"] == 3
    assert result.metrics["parsed_count"] == 2
    assert result.metrics["exact_matches"] == 1
    assert result.metrics["exact_match_rate_all_queries"] == 1 / 3
    assert marker not in json.dumps(result.records)
    assert "prompt" in result.records[0]


def test_next_state_evaluation(episode: Episode) -> None:
    backend = SequenceBackend(
        [
            'predictions = ["INC", "INC"]',
            'predictions = ["INC", "INC"]',
            'predictions = ["INC", "DEC"]',
        ]
    )
    result = evaluate_episode(
        episode,
        EvaluationConfig(
            task_name="MountainCar-v0",
            metric="next-state",
            question_name="next_state_prediction",
            history_size=0,
            include_prompts=False,
        ),
        backend,
    )
    assert result.metrics["exact_matches"] == 3
    assert result.system_prompt == ""
    assert all("prompt" not in record for record in result.records)


def test_last_action_and_last_state_evaluation(episode: Episode) -> None:
    last_action = evaluate_episode(
        episode,
        EvaluationConfig(
            task_name="MountainCar-v0",
            metric="last-action",
            question_name="last_action_prediction",
            history_size=0,
            include_prompts=False,
        ),
        SequenceBackend([">>Final action choice: [1]", ">>Final action choice: [2]"]),
    )
    assert last_action.metrics["query_count"] == 2
    assert last_action.metrics["exact_matches"] == 2

    last_state = evaluate_episode(
        episode,
        EvaluationConfig(
            task_name="MountainCar-v0",
            metric="last-state",
            question_name="last_state_prediction",
            history_size=0,
            include_prompts=False,
        ),
        SequenceBackend(
            ['predictions = ["INC", "INC"]', 'predictions = ["INC", "DEC"]']
        ),
    )
    assert last_state.metrics["query_count"] == 2
    assert last_state.metrics["exact_matches"] == 2


def test_continuous_action_bins(tmp_path) -> None:
    path = tmp_path / "continuous.npz"
    np.savez(
        path,
        states=np.asarray([[[1.0, 0.0, 0.0]]] * 4, dtype=np.float32),
        actions=np.asarray([[[-2.0]], [[-1.6]], [[0.0]], [[2.0]]], dtype=np.float32),
        rewards=np.asarray([[-1.0]] * 4, dtype=np.float32),
        episodic_return=np.asarray([[-4.0]], dtype=np.float32),
    )
    result = evaluate_episode(
        Episode.load(path),
        EvaluationConfig(
            task_name="Pendulum-v1",
            metric="next-action",
            question_name="next_action_prediction_continuous_bins",
            history_size=0,
            include_prompts=False,
        ),
        SequenceBackend(
            ["predictions = [0]", "predictions = [4]", "predictions = [9]"]
        ),
    )
    assert result.metrics["exact_matches"] == 3
    assert result.metrics["mean_element_accuracy_parsed_queries"] == 1.0


def test_all_registered_question_variants_render(episode: Episode) -> None:
    for metric, names in available_questions().items():
        for name in names:
            index = 1
            text = render_question(
                metric,
                resolve_question(metric, name),
                episode,
                index=index,
                drop_last_feature=False,
                presented_action=episode.discrete_action(index) if metric == "argue-action" else None,
            )
            assert isinstance(text, str)
            assert text.strip()


def test_argue_action_sampling_is_seed_deterministic(episode: Episode) -> None:
    responses = ["final_vote = [True]", "final_vote = [False]", "final_vote = [True]"]
    config = EvaluationConfig(
        task_name="MountainCar-v0",
        metric="argue-action",
        question_name="argue_action",
        history_size=0,
        seed=11,
        include_prompts=False,
    )
    first = evaluate_episode(episode, config, SequenceBackend(responses))
    second = evaluate_episode(episode, config, SequenceBackend(responses))
    assert [record["presented_action"] for record in first.records] == [
        record["presented_action"] for record in second.records
    ]
