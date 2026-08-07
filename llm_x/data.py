"""Validated loading and formatting of offline RL episodes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np


REQUIRED_STEP_KEYS = ("states", "actions", "rewards")
REQUIRED_KEYS = (*REQUIRED_STEP_KEYS, "episodic_return")
OPTIONAL_STEP_KEYS = ("achieved_goals", "desired_goals", "agent_dirs", "dir_vectors")


class EpisodeValidationError(ValueError):
    """Raised when an episode does not match the documented NPZ contract."""


@dataclass(frozen=True)
class Episode:
    """An immutable view of one validated offline episode."""

    path: Path
    arrays: Mapping[str, np.ndarray]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Episode file does not exist: {source}")
        if source.suffix.lower() != ".npz":
            raise EpisodeValidationError("Episode files must use the .npz format")

        try:
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as exc:
            raise EpisodeValidationError(f"Unable to read episode {source}: {exc}") from exc

        missing = sorted(set(REQUIRED_KEYS) - set(arrays))
        if missing:
            raise EpisodeValidationError(f"Missing required NPZ keys: {', '.join(missing)}")

        for key, value in arrays.items():
            if value.dtype.hasobject:
                raise EpisodeValidationError(f"Object arrays are not allowed: {key}")
            if not np.issubdtype(value.dtype, np.number):
                raise EpisodeValidationError(f"Array must be numeric: {key} ({value.dtype})")

        lengths = {key: arrays[key].shape[0] for key in REQUIRED_STEP_KEYS if arrays[key].ndim}
        if len(lengths) != len(REQUIRED_STEP_KEYS):
            raise EpisodeValidationError("states, actions, and rewards must have a step dimension")
        if len(set(lengths.values())) != 1:
            detail = ", ".join(f"{key}={value}" for key, value in lengths.items())
            raise EpisodeValidationError(f"Step arrays have different lengths: {detail}")
        length = next(iter(lengths.values()))
        if length < 2:
            raise EpisodeValidationError("An episode must contain at least two steps")
        for key in OPTIONAL_STEP_KEYS:
            if key in arrays and (arrays[key].ndim == 0 or arrays[key].shape[0] != length):
                raise EpisodeValidationError(
                    f"Optional step array {key} must have length {length}, received {arrays[key].shape}"
                )

        digest = sha256(source.read_bytes()).hexdigest()
        return cls(path=source, arrays=arrays, sha256=digest)

    @property
    def length(self) -> int:
        return int(self.arrays["states"].shape[0])

    def state(self, index: int, *, drop_last_feature: bool = False) -> np.ndarray:
        value = np.asarray(self.arrays["states"][index])
        if drop_last_feature:
            if value.shape[-1] < 2:
                raise EpisodeValidationError("Cannot drop the last feature from a scalar state")
            value = value[..., :-1]
        return value

    def state_vector(self, index: int, *, drop_last_feature: bool = False) -> np.ndarray:
        return self.state(index, drop_last_feature=drop_last_feature).reshape(-1)

    def action_vector(self, index: int) -> np.ndarray:
        return np.asarray(self.arrays["actions"][index]).reshape(-1)

    def discrete_action(self, index: int) -> int:
        values = self.action_vector(index)
        if values.size != 1:
            raise EpisodeValidationError("The selected question requires a scalar discrete action")
        value = float(values[0])
        if not value.is_integer():
            raise EpisodeValidationError(f"Expected an integer action, received {value}")
        return int(value)

    def reward(self, index: int) -> float | list[float]:
        values = np.asarray(self.arrays["rewards"][index]).reshape(-1)
        if values.size == 1:
            return float(values[0])
        return [float(value) for value in values]

    def history_text(
        self,
        start: int,
        end: int,
        *,
        indexed: bool,
        drop_last_state_feature: bool = False,
    ) -> str:
        """Format half-open step range ``[start, end)`` without changing its values."""

        if not 0 <= start < end <= self.length:
            raise IndexError(f"Invalid history range [{start}, {end}) for length {self.length}")
        if indexed:
            rows = []
            for index in range(start, end):
                rows.append(
                    "\n".join(
                        (
                            f"Step {index}:",
                            f"  state: {_array_text(self.state(index, drop_last_feature=drop_last_state_feature))}",
                            f"  action: {_array_text(self.arrays['actions'][index])}",
                            f"  reward: {_array_text(self.arrays['rewards'][index])}",
                        )
                    )
                )
            return "\n\n".join(rows)

        return "\n".join(
            (
                f"states:\n{_array_text(_history_states(self, start, end, drop_last_state_feature))}",
                f"actions:\n{_array_text(self.arrays['actions'][start:end])}",
                f"rewards:\n{_array_text(self.arrays['rewards'][start:end])}",
            )
        )


def _array_text(value: np.ndarray) -> str:
    return np.array2string(np.asarray(value), precision=4, separator=", ")


def _history_states(episode: Episode, start: int, end: int, drop_last_feature: bool) -> np.ndarray:
    values = np.asarray(episode.arrays["states"][start:end])
    return values[..., :-1] if drop_last_feature else values
