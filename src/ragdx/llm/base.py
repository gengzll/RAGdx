"""Base abstractions for ragdx LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


LLMCallable = Callable[[str], str]
"""Minimal LLM call signature accepted across ragdx (prompt → text)."""


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete subclasses implement :meth:`complete`. Provider instances are
    callable (``provider(prompt)``), which matches the ``Callable[[str], str]``
    interface used by :class:`ragdx.engines.llm_diagnosis.LLMDiagnosisExplainer`
    and :class:`ragdx.optim.planner.OptimizationPlanner`.
    """

    name: str = "base"
    default_model: str = ""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or self.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Submit ``prompt`` to the backend and return the raw text."""

    def __call__(self, prompt: str) -> str:  # pragma: no cover - thin wrapper
        return self.complete(prompt)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(model={self.model!r})"


__all__ = ["LLMProvider", "LLMCallable"]
