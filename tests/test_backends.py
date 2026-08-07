from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from llm_x.backends import BackendError, OpenAIChatBackend, _reject_endpoint_credentials


def test_openai_backend_uses_system_and_user_messages(monkeypatch) -> None:
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            calls.append(("create", kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="formatted result"))]
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-value-not-published")
    backend = OpenAIChatBackend(
        model="explicit-model",
        api_key_env="TEST_OPENAI_KEY",
        endpoint=None,
        timeout=12,
        retries=0,
        compatible=False,
    )
    assert backend.complete(system_prompt="system", user_prompt="user") == "formatted result"
    create = calls[1][1]
    assert create["model"] == "explicit-model"
    assert create["temperature"] == 0
    assert create["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_endpoint_cannot_embed_credentials() -> None:
    with pytest.raises(BackendError, match="must not be embedded"):
        _reject_endpoint_credentials("https://user:password@example.invalid/v1")
