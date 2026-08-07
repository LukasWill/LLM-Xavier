from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from llm_x.data import Episode


@pytest.fixture
def episode_path(tmp_path: Path) -> Path:
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        states=np.asarray(
            [[[-0.50, 0.00]], [[-0.49, 0.01]], [[-0.47, 0.02]], [[-0.46, 0.01]]],
            dtype=np.float32,
        ),
        actions=np.asarray([[0], [1], [2], [1]], dtype=np.int64),
        rewards=np.asarray([[-1.0], [-1.0], [-1.0], [-1.0]], dtype=np.float32),
        episodic_return=np.asarray([[-4.0]], dtype=np.float32),
    )
    return path


@pytest.fixture
def episode(episode_path: Path) -> Episode:
    return Episode.load(episode_path)
