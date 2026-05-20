"""Tests for the ragdx.llm registry — provider selection without imports."""

from __future__ import annotations

import pytest

from ragdx.config import LLMSettings
from ragdx.errors import LLMConfigError
from ragdx.llm import (
    build_provider,
    get_llm_callable,
    list_providers,
    register_provider,
)
from ragdx.llm.base import LLMProvider


class _DummyProvider(LLMProvider):
    name = "dummy"
    default_model = "dummy-v0"

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if k in {"model", "temperature", "max_tokens", "timeout"}})
        self._kwargs = kwargs
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"echo: {prompt}"


def _factory(settings: LLMSettings) -> LLMProvider:
    return _DummyProvider(model=settings.model or _DummyProvider.default_model)


def test_register_and_list_provider():
    register_provider("dummy", _factory)
    assert "dummy" in list_providers()


def test_build_provider_returns_registered():
    register_provider("dummy", _factory)
    settings = LLMSettings(provider="dummy", model="dummy-v0")
    provider = build_provider(settings)
    assert isinstance(provider, _DummyProvider)
    # Callable interface inherited from LLMProvider
    assert provider("hello") == "echo: hello"


def test_unknown_provider_raises():
    settings = LLMSettings(provider="not-a-real-provider", model="foo")
    with pytest.raises(LLMConfigError):
        build_provider(settings)


def test_get_llm_callable_delegates(monkeypatch):
    register_provider("dummy", _factory)
    monkeypatch.setenv("RAGDX_LLM_PROVIDER", "dummy")
    call = get_llm_callable()
    assert callable(call)
    assert call("ping") == "echo: ping"
