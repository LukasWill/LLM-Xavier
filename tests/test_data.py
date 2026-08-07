from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_x.data import Episode, EpisodeValidationError


def test_load_and_format_episode(episode: Episode) -> None:
    assert episode.length == 4
    assert episode.discrete_action(2) == 2
    assert episode.state_vector(0).tolist() == pytest.approx([-0.5, 0.0])
    assert "Step 0:" in episode.history_text(0, 2, indexed=True)
    assert "states:" in episode.history_text(0, 2, indexed=False)
    assert len(episode.sha256) == 64


def test_rejects_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "missing.npz"
    np.savez(path, states=np.zeros((2, 1)), actions=np.zeros((2, 1)), rewards=np.zeros((2, 1)))
    with pytest.raises(EpisodeValidationError, match="episodic_return"):
        Episode.load(path)


def test_rejects_misaligned_steps(tmp_path: Path) -> None:
    path = tmp_path / "misaligned.npz"
    np.savez(
        path,
        states=np.zeros((3, 1)),
        actions=np.zeros((2, 1)),
        rewards=np.zeros((3, 1)),
        episodic_return=np.zeros((1, 1)),
    )
    with pytest.raises(EpisodeValidationError, match="different lengths"):
        Episode.load(path)
