"""Azure OpenAI provider implementation."""

from __future__ import annotations

from typing import Optional

from ragdx.errors import DependencyError, LLMConfigError, LLMError
from ragdx.llm.base import LLMProvider
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class AzureOpenAIProvider(LLMProvider):
    """LLM provider backed by Azure OpenAI Service.

    ``model`` here must be a deployment name (not the underlying model id).
    """

    name = "azure"
    default_model = "gpt-4o-mini"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-06-01",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        try:
            from openai import AzureOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DependencyError(
                "AzureOpenAIProvider requires the `openai` package. "
                "Install with `pip install ragdx[azure]`."
            ) from exc
        import os

        if not api_key:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not azure_endpoint:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key or not azure_endpoint:
            raise LLMConfigError(
                "AzureOpenAIProvider requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
        )

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
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Azure OpenAI call failed: {exc}") from exc
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise LLMError(f"Unexpected Azure OpenAI response shape: {response!r}") from exc
        if content is None:
            raise LLMError("Azure OpenAI response had no content.")
        return content


__all__ = ["AzureOpenAIProvider"]
