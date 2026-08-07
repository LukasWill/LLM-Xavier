"""Create the tiny deterministic episode used by the README smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/llmx-smoke.npz")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        states=np.asarray(
            [[[-0.50, 0.00]], [[-0.49, 0.01]], [[-0.47, 0.02]], [[-0.46, 0.01]]],
            dtype=np.float32,
        ),
        actions=np.asarray([[0], [1], [2], [1]], dtype=np.int64),
        rewards=np.asarray([[-1.0], [-1.0], [-1.0], [-1.0]], dtype=np.float32),
        episodic_return=np.asarray([[-4.0]], dtype=np.float32),
    )
    print(output)


if __name__ == "__main__":
    main()
