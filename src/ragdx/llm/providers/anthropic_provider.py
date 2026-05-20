"""Anthropic provider implementation."""

from __future__ import annotations

from typing import Optional

from ragdx.errors import DependencyError, LLMConfigError, LLMError
from ragdx.llm.base import LLMProvider
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the official ``anthropic`` Python SDK."""

    name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4096,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        try:
            from anthropic import Anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DependencyError(
                "AnthropicProvider requires the `anthropic` package. "
                "Install with `pip install ragdx[anthropic]`."
            ) from exc
        if not api_key:
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "AnthropicProvider requires an API key. Set ANTHROPIC_API_KEY or pass api_key=..."
            )
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens or 4096,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Anthropic messages.create call failed: {exc}") from exc
        try:
            blocks = response.content
            texts = []
            for block in blocks:
                # Anthropic SDK returns objects with .text or dicts
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    texts.append(text)
            if not texts:
                raise LLMError(f"Anthropic response had no text blocks: {response!r}")
            return "\n".join(texts)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Unexpected Anthropic response shape: {response!r}") from exc


__all__ = ["AnthropicProvider"]
