"""Fail when the prospective release contains private or generated artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", ".mypy_cache", "__pycache__"}
FORBIDDEN_DIRS = {
    "RQ1",
    "RQ2",
    "RQ3",
    "offline_results",
    "offline_results_paper",
    "test_offline_runs",
    "text_game_data",
}
FORBIDDEN_NAMES = {".DS_Store", "responses.jsonl"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".npz", ".cleanrl_model", ".pt", ".pth", ".ckpt"}
TEXT_PATTERNS = {
    "absolute macOS home path": re.compile(r"/Users/[^/\s]+/"),
    "absolute Linux home path": re.compile(r"/home/[^/\s]+/"),
    "private 10/8 address": re.compile(r"(?<![\d.])10(?:\.\d{1,3}){3}(?![\d.])"),
    "private 192.168/16 address": re.compile(r"(?<![\d.])192\.168(?:\.\d{1,3}){2}(?![\d.])"),
    "private 172.16/12 address": re.compile(
        r"(?<![\d.])172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}(?![\d.])"
    ),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "GitHub-style secret": re.compile(r"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def main() -> int:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            failures.append(f"forbidden artifact directory: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.name.startswith("events.out.tfevents"):
            failures.append(f"generated/private artifact: {relative}")
        if path.suffix in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary artifact: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"file exceeds 5 MiB release limit: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # The checker necessarily contains the signatures it searches for.
        if path.resolve() == Path(__file__).resolve():
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")

    if failures:
        print("Release check failed:", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Release check passed: no forbidden paths, generated binaries, private paths, or known secret formats.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
