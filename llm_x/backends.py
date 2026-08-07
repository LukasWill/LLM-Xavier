"""Explicit, timeout-bounded chat backends."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class BackendError(RuntimeError):
    """Raised when a backend cannot return a usable response."""


class ChatBackend(Protocol):
    name: str
    model: str

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class SequenceBackend:
    """Deterministic offline backend used by tests and smoke evaluations."""

    responses: Sequence[str]
    model: str = "fixture-sequence"
    name: str = "fixture"
    _index: int = 0

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        if self._index >= len(self.responses):
            raise BackendError(
                f"Fixture backend exhausted after {len(self.responses)} responses"
            )
        response = self.responses[self._index]
        self._index += 1
        return response

    @classmethod
    def from_file(cls, path: str | Path) -> "SequenceBackend":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Fixture response file does not exist: {source}")
        if source.suffix.lower() == ".jsonl":
            responses = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        else:
            responses = json.loads(source.read_text())
        if not isinstance(responses, list) or not all(isinstance(item, str) for item in responses):
            raise ValueError("Fixture responses must be a JSON array or JSONL stream of strings")
        return cls(responses=responses)


class OpenAIChatBackend:
    """OpenAI or explicitly configured OpenAI-compatible chat backend."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        endpoint: str | None,
        timeout: float,
        retries: int,
        compatible: bool,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BackendError(
                "The OpenAI backend requires the optional dependency: pip install 'llm-x[openai]'"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise BackendError(f"Required credential environment variable is unset: {api_key_env}")
        if compatible and not endpoint:
            raise BackendError("An explicit --endpoint is required for openai-compatible backends")
        _reject_endpoint_credentials(endpoint)

        client_args: dict[str, object] = {"api_key": api_key, "timeout": timeout}
        if endpoint:
            client_args["base_url"] = endpoint
        self._client = OpenAI(**client_args)
        self.model = model
        self.retries = retries
        self.name = "openai-compatible" if compatible else "openai"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content
                if not content:
                    raise BackendError("Backend returned an empty response")
                return content
            except Exception as exc:  # SDK exception types vary by installed version.
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise BackendError(f"Backend failed after {self.retries + 1} attempts: {last_error}")


class OllamaChatBackend:
    """Ollama's documented local HTTP chat endpoint."""

    name = "ollama"

    def __init__(self, *, model: str, endpoint: str, timeout: float, retries: int) -> None:
        if not endpoint:
            raise BackendError("An explicit --endpoint is required for Ollama")
        _reject_endpoint_credentials(endpoint)
        self.model = model
        self.endpoint = endpoint.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.retries = retries

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body.get("message", {}).get("content")
                if not isinstance(content, str) or not content:
                    raise BackendError("Ollama returned an empty or malformed response")
                return content
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise BackendError(f"Ollama failed after {self.retries + 1} attempts: {last_error}")


def _reject_endpoint_credentials(endpoint: str | None) -> None:
    if not endpoint:
        return
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.username is not None or parsed.password is not None:
        raise BackendError("Credentials must not be embedded in an endpoint URL")
