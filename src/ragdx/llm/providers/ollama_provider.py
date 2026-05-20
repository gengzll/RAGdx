"""Ollama (local) provider implementation.

Talks to a local Ollama server via its HTTP API. No heavy ML dependency is
pulled in by this module — only ``httpx`` is required (already common).
"""

from __future__ import annotations

from typing import Optional

from ragdx.errors import DependencyError, LLMError
from ragdx.llm.base import LLMProvider
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider backed by a local Ollama server.

    Parameters
    ----------
    host:
        Base URL of the Ollama HTTP server. Defaults to
        ``http://127.0.0.1:11434``.
    model:
        Model identifier (e.g. ``llama3.1``). Defaults to ``llama3.1``.
    """

    name = "ollama"
    default_model = "llama3.1"

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        try:
            import httpx  # noqa: F401  type: ignore[import-not-found]
        except ImportError as exc:
            raise DependencyError(
                "OllamaProvider requires the `httpx` package. Install with `pip install ragdx[ollama]`."
            ) from exc
        import os

        self._host = host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        self._host = self._host.rstrip("/")

    def complete(self, prompt: str) -> str:
        import httpx  # type: ignore[import-not-found]

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(f"{self._host}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama HTTP call failed: {exc}") from exc
        text = data.get("response")
        if not isinstance(text, str):
            raise LLMError(f"Ollama response missing `response` field: {data!r}")
        return text


__all__ = ["OllamaProvider"]
