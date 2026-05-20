"""OpenAI provider implementation."""

from __future__ import annotations

from ragdx.errors import DependencyError, LLMConfigError, LLMError
from ragdx.llm.base import LLMProvider
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the official ``openai`` Python SDK.

    Parameters
    ----------
    api_key:
        OpenAI API key. If omitted, reads ``OPENAI_API_KEY`` via the SDK
        default chain.
    base_url:
        Optional custom base URL (for proxies / compatible endpoints).
    model:
        Model name (default ``gpt-4o-mini``).
    """

    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DependencyError(
                "OpenAIProvider requires the `openai` package. Install with `pip install ragdx[openai]`."
            ) from exc
        if not api_key:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "OpenAIProvider requires an API key. Set OPENAI_API_KEY or pass api_key=..."
            )
        client_kwargs: dict = {"api_key": api_key, "timeout": timeout}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def complete(self, prompt: str) -> str:
        try:
            kwargs: dict = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"OpenAI chat.completions call failed: {exc}") from exc
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {response!r}") from exc
        if content is None:
            raise LLMError("OpenAI response did not include any content.")
        return content


__all__ = ["OpenAIProvider"]
