"""Exception hierarchy for ragdx.

All ragdx-raised exceptions derive from :class:`RagdxError`. Library
consumers can catch the umbrella class to handle any ragdx-specific
failure, or catch the narrower subclasses to react to particular
failure modes.

The exception types are intentionally lightweight (no extra state
beyond message + optional cause) so they remain easy to raise from
adapter code without coupling to a specific framework.
"""

from __future__ import annotations


class RagdxError(Exception):
    """Base class for all ragdx-raised exceptions."""


class ConfigError(RagdxError):
    """Raised when configuration (env vars, settings) is invalid or missing."""


class DependencyError(RagdxError):
    """Raised when an optional dependency required by a code path is missing."""


class LLMError(RagdxError):
    """Raised when an LLM provider call fails or returns an unparsable result."""


class LLMConfigError(LLMError, ConfigError):
    """Raised when an LLM provider cannot be configured (missing key/model)."""


class EvaluationError(RagdxError):
    """Raised when an evaluator cannot compute valid metrics."""


class RunnerError(RagdxError):
    """Raised when an external optimization runner fails or is misconfigured."""


class RunnerMissingError(RunnerError, ConfigError):
    """Raised when ``execute`` mode is requested but no runner is configured."""


class StorageError(RagdxError):
    """Raised on persistence failures (missing/corrupt run, session, etc.)."""


__all__ = [
    "ConfigError",
    "DependencyError",
    "EvaluationError",
    "LLMConfigError",
    "LLMError",
    "RagdxError",
    "RunnerError",
    "RunnerMissingError",
    "StorageError",
]
