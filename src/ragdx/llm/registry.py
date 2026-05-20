"""Provider registry and high-level factory helpers."""

from __future__ import annotations

from collections.abc import Callable

from ragdx.config import LLMSettings
from ragdx.errors import LLMConfigError
from ragdx.llm.base import LLMCallable, LLMProvider
from ragdx.llm.providers.anthropic_provider import AnthropicProvider
from ragdx.llm.providers.azure_provider import AzureOpenAIProvider
from ragdx.llm.providers.ollama_provider import OllamaProvider
from ragdx.llm.providers.openai_provider import OpenAIProvider

DEFAULT_MODELS: dict[str, str] = {
    "openai": OpenAIProvider.default_model,
    "anthropic": AnthropicProvider.default_model,
    "azure": AzureOpenAIProvider.default_model,
    "ollama": OllamaProvider.default_model,
}


ProviderFactory = Callable[[LLMSettings], LLMProvider]


def _openai_factory(settings: LLMSettings) -> LLMProvider:
    return OpenAIProvider(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )


def _anthropic_factory(settings: LLMSettings) -> LLMProvider:
    return AnthropicProvider(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )


def _azure_factory(settings: LLMSettings) -> LLMProvider:
    return AzureOpenAIProvider(
        api_key=settings.api_key,
        azure_endpoint=settings.base_url,
        api_version=settings.api_version or "2024-06-01",
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )


def _ollama_factory(settings: LLMSettings) -> LLMProvider:
    return OllamaProvider(
        host=settings.base_url,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout,
    )


_REGISTRY: dict[str, ProviderFactory] = {
    "openai": _openai_factory,
    "anthropic": _anthropic_factory,
    "azure": _azure_factory,
    "ollama": _ollama_factory,
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or override) a provider factory under ``name``."""

    _REGISTRY[name.lower()] = factory


def list_providers() -> list[str]:
    """Return the sorted list of provider names currently registered."""

    return sorted(_REGISTRY)


def build_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Instantiate the LLM provider selected by ``settings``.

    If ``settings`` is omitted, the active environment is read via
    :meth:`LLMSettings.from_env`.
    """

    settings = settings or LLMSettings.from_env()
    factory = _REGISTRY.get(settings.provider)
    if factory is None:
        raise LLMConfigError(
            f"Unknown LLM provider {settings.provider!r}. Known: {list_providers()}"
        )
    return factory(settings)


def get_llm_callable(settings: LLMSettings | None = None) -> LLMCallable:
    """Return a ``Callable[[str], str]`` ready to plug into ragdx engines."""

    return build_provider(settings)


__all__ = [
    "DEFAULT_MODELS",
    "ProviderFactory",
    "build_provider",
    "get_llm_callable",
    "list_providers",
    "register_provider",
]
