"""Smoke tests for ragdx.config and ragdx.errors."""

from __future__ import annotations

import pytest

from ragdx.config import (
    ExecutionSettings,
    LLMSettings,
    StorageSettings,
    get_settings,
)
from ragdx.errors import (
    ConfigError,
    LLMConfigError,
    LLMError,
    RagdxError,
)


def _isolate_env(monkeypatch: pytest.MonkeyPatch, *keys: str) -> None:
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_storage_settings_uses_env_root(monkeypatch, tmp_path):
    target = tmp_path / "custom-root"
    monkeypatch.setenv("RAGDX_ROOT", str(target))
    s = StorageSettings.from_env()
    assert str(s.root) == str(target)


def test_storage_settings_default(monkeypatch):
    _isolate_env(monkeypatch, "RAGDX_ROOT")
    s = StorageSettings.from_env()
    assert str(s.root) == ".ragdx"


def test_llm_settings_openai_picks_openai_api_key(monkeypatch):
    _isolate_env(
        monkeypatch,
        "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "RAGDX_LLM_MODEL", "OPENAI_BASE_URL",
    )
    monkeypatch.setenv("RAGDX_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMSettings.from_env()
    assert cfg.provider == "openai"
    assert cfg.api_key == "sk-test"
    assert cfg.api_version is None


def test_llm_settings_anthropic_isolated(monkeypatch):
    monkeypatch.setenv("RAGDX_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
    cfg = LLMSettings.from_env()
    assert cfg.provider == "anthropic"
    assert cfg.api_key == "ant-key"


def test_llm_settings_azure_defaults_api_version(monkeypatch):
    monkeypatch.setenv("RAGDX_LLM_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    cfg = LLMSettings.from_env()
    assert cfg.provider == "azure"
    assert cfg.api_version == "2024-06-01"


def test_llm_settings_ollama_defaults_host(monkeypatch):
    _isolate_env(monkeypatch, "OLLAMA_HOST")
    monkeypatch.setenv("RAGDX_LLM_PROVIDER", "ollama")
    cfg = LLMSettings.from_env()
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://127.0.0.1:11434"


def test_execution_settings_strict_default(monkeypatch):
    _isolate_env(monkeypatch, "RAGDX_STRICT_EXECUTE", "RAGDX_RUNNER_TIMEOUT_SEC", "RAGDX_BO_BACKEND")
    s = ExecutionSettings.from_env()
    assert s.strict_execute is True
    assert s.runner_timeout_sec is None
    assert s.bo_backend == "internal"
    assert s.fallback_simulate_on_missing_runner is False


def test_execution_settings_lenient_via_env(monkeypatch):
    monkeypatch.setenv("RAGDX_STRICT_EXECUTE", "0")
    monkeypatch.setenv("RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", "yes")
    s = ExecutionSettings.from_env()
    assert s.strict_execute is False
    assert s.fallback_simulate_on_missing_runner is True


def test_get_settings_is_not_cached(monkeypatch):
    monkeypatch.setenv("RAGDX_ROOT", "first")
    a = get_settings()
    monkeypatch.setenv("RAGDX_ROOT", "second")
    b = get_settings()
    assert str(a.storage.root) == "first"
    assert str(b.storage.root) == "second"


def test_error_hierarchy_subclasses():
    assert issubclass(LLMError, RagdxError)
    assert issubclass(LLMConfigError, LLMError)
    assert issubclass(LLMConfigError, ConfigError)
    err = LLMConfigError("no key")
    assert isinstance(err, RagdxError)
    assert str(err) == "no key"
