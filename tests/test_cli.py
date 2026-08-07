from __future__ import annotations

import json
from pathlib import Path

from llm_x.cli import main


def test_fixture_cli_outputs_are_path_and_response_safe(episode_path: Path, tmp_path: Path) -> None:
    marker = "PRIVATE_RESPONSE_MARKER"
    fixture = tmp_path / "responses.json"
    fixture.write_text(
        json.dumps(
            [
                f"{marker} >>Final action choice: [1]",
                f"{marker} >>Final action choice: [2]",
                f"{marker} >>Final action choice: [1]",
            ]
        )
    )
    output = tmp_path / "output"
    exit_code = main(
        [
            "evaluate",
            "--data-path",
            str(episode_path),
            "--task-name",
            "MountainCar-v0",
            "--metric",
            "next-action",
            "--question-name",
            "next_action_prediction",
            "--history-size",
            "0",
            "--backend",
            "fixture",
            "--fixture-responses",
            str(fixture),
            "--output-dir",
            str(output),
        ]
    )
    assert exit_code == 0
    combined = "\n".join(path.read_text() for path in output.iterdir())
    assert marker not in combined
    assert str(tmp_path) not in combined
    assert json.loads((output / "run.json").read_text())["raw_responses_stored"] is False
    assert json.loads((output / "metrics.json").read_text())["exact_matches"] == 3


def test_dry_run_does_not_create_output(episode_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    fixture = tmp_path / "dry-run-responses.json"
    fixture.write_text(json.dumps([">>Final action choice: [1]"]))
    exit_code = main(
        [
            "evaluate",
            "--data-path",
            str(episode_path),
            "--task-name",
            "MountainCar-v0",
            "--metric",
            "next-action",
            "--question-name",
            "next_action_prediction",
            "--history-size",
            "0",
            "--backend",
            "fixture",
            "--fixture-responses",
            str(fixture),
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output.exists()
