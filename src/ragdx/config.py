"""Centralized configuration for ragdx.

All configurable knobs read from environment variables fall through this
module so applications embedding ragdx can override them programmatically
without monkey-patching ``os.environ``.

Environment variables recognised
--------------------------------

Storage
- ``RAGDX_ROOT``: persistence root directory (default ``.ragdx`` in cwd).
  Wins over ``RAGDX_PROJECT`` when both are set (RAGDX_ROOT is the
  fully-qualified path; RAGDX_PROJECT is the per-project shorthand).
- ``RAGDX_PROJECT``: per-project namespace. When set (and ``RAGDX_ROOT``
  is unset), the storage root becomes ``.ragdx/projects/<project>/`` so
  every project's runs / sessions / feedback / causal priors live in
  their own folder. Also surfaced as the ``--project`` root CLI flag.

LLM
- ``RAGDX_LLM_PROVIDER``: ``openai`` | ``anthropic`` | ``azure`` | ``ollama``.
- ``RAGDX_LLM_MODEL``: model name (provider-specific default if unset).
- ``RAGDX_LLM_TIMEOUT``: request timeout in seconds (default 60).
- ``RAGDX_LLM_TEMPERATURE``: sampling temperature (default 0.0).
- ``RAGDX_LLM_MAX_TOKENS``: optional cap on output tokens.
- ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / ``AZURE_OPENAI_API_KEY``: provider
  credentials. Azure additionally uses ``AZURE_OPENAI_ENDPOINT`` and
  ``AZURE_OPENAI_API_VERSION``.
- ``OLLAMA_HOST``: base URL for the local Ollama server (default
  ``http://127.0.0.1:11434``).

Execution
- ``RAGDX_STRICT_EXECUTE``: refuse to fall back to simulator when an external
  runner is missing in ``execute`` mode (default true).
- ``RAGDX_RUNNER_TIMEOUT_SEC``: per-trial subprocess timeout (default unset).
- ``RAGDX_BO_BACKEND``: ``internal`` (default) | ``ax``.
- ``RAGDX_*_RUNNER_CMD``: per-tool subprocess command templates (see
  ``ragdx.optim.executor``).

Logging
- ``RAGDX_LOG_LEVEL`` / ``RAGDX_LOG_FORMAT`` / ``RAGDX_LOG_FILE`` — see
  :mod:`ragdx.utils.logging`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_path(name: str, default: str | Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else Path(default)


@dataclass(frozen=True)
class LLMSettings:
    """Runtime configuration for the default LLM provider."""

    provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> LLMSettings:
        provider = (os.environ.get("RAGDX_LLM_PROVIDER") or "openai").lower().strip()
        model = os.environ.get("RAGDX_LLM_MODEL")
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            api_version = None
        elif provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            base_url = os.environ.get("ANTHROPIC_BASE_URL")
            api_version = None
        elif provider == "azure":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            base_url = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        elif provider == "ollama":
            api_key = None
            base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            api_version = None
        else:
            # Unknown providers are still permitted; callers may wire custom
            # providers via the registry.
            api_key = os.environ.get(f"{provider.upper()}_API_KEY")
            base_url = os.environ.get(f"{provider.upper()}_BASE_URL")
            api_version = None
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            temperature=env_float("RAGDX_LLM_TEMPERATURE", 0.0),
            max_tokens=env_int("RAGDX_LLM_MAX_TOKENS"),
            timeout=env_float("RAGDX_LLM_TIMEOUT", 60.0),
        )


@dataclass(frozen=True)
class ExecutionSettings:
    """Runtime configuration for the optimization executor."""

    strict_execute: bool = True
    runner_timeout_sec: float | None = None
    bo_backend: str = "internal"
    fallback_simulate_on_missing_runner: bool = False

    @classmethod
    def from_env(cls) -> ExecutionSettings:
        timeout_raw = os.environ.get("RAGDX_RUNNER_TIMEOUT_SEC", "").strip()
        timeout: float | None = None
        if timeout_raw:
            try:
                timeout = float(timeout_raw) or None
            except ValueError:
                timeout = None
        return cls(
            strict_execute=env_bool("RAGDX_STRICT_EXECUTE", default=True),
            runner_timeout_sec=timeout,
            bo_backend=(os.environ.get("RAGDX_BO_BACKEND") or "internal").lower(),
            fallback_simulate_on_missing_runner=env_bool(
                "RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", default=False
            ),
        )


@dataclass(frozen=True)
class StorageSettings:
    """Storage paths configuration.

    Two env vars feed this:

    * ``RAGDX_ROOT``: fully-qualified root directory (wins when set).
    * ``RAGDX_PROJECT``: per-project shorthand. Resolved as
      ``.ragdx/projects/<name>/`` so each project keeps its own
      RunStore / OptimizationSessions / feedback / causal priors.

    ``project`` is also exposed as a field so callers can read which
    namespace was in effect (handy for logs and error messages).
    """

    root: Path = Path(".ragdx")
    project: str | None = None

    @classmethod
    def from_env(cls) -> StorageSettings:
        # RAGDX_ROOT is the explicit override: if the user set a full
        # path, respect it verbatim and don't append a project segment.
        explicit_root = os.environ.get("RAGDX_ROOT")
        project = (os.environ.get("RAGDX_PROJECT") or "").strip() or None
        if explicit_root:
            return cls(root=Path(explicit_root), project=project)
        if project:
            return cls(root=Path(".ragdx") / "projects" / project, project=project)
        return cls(root=Path(".ragdx"), project=None)


@dataclass(frozen=True)
class Settings:
    """Aggregate configuration object used by :func:`get_settings`."""

    storage: StorageSettings = field(default_factory=StorageSettings.from_env)
    llm: LLMSettings = field(default_factory=LLMSettings.from_env)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings.from_env)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            storage=StorageSettings.from_env(),
            llm=LLMSettings.from_env(),
            execution=ExecutionSettings.from_env(),
        )


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` snapshot built from the current env.

    The snapshot is not cached: tests and applications can mutate
    ``os.environ`` between calls and expect changes to be observed.
    """

    return Settings.from_env()


__all__ = [
    "ExecutionSettings",
    "LLMSettings",
    "Settings",
    "StorageSettings",
    "env_bool",
    "env_float",
    "env_int",
    "env_path",
    "get_settings",
]
