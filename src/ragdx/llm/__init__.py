"""Pluggable LLM provider layer for ragdx.

ragdx's diagnosis and planning engines accept a plain ``Callable[[str], str]``
so users can wire any backend without touching ragdx code. This subpackage
provides ready-made provider implementations (OpenAI, Anthropic, Azure
OpenAI, Ollama) plus a small registry so applications can resolve a
provider by name or from environment variables.

Typical usage::

    from ragdx.llm import get_llm_callable

    # Reads RAGDX_LLM_PROVIDER / RAGDX_LLM_MODEL / *_API_KEY etc.
    call = get_llm_callable()
    text = call("Diagnose this RAG run...")

Or, fully programmatic::

    from ragdx.llm import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
    text = provider("Diagnose this RAG run...")
"""

from __future__ import annotations

from ragdx.llm.base import LLMCallable, LLMProvider
from ragdx.llm.providers.anthropic_provider import AnthropicProvider
from ragdx.llm.providers.azure_provider import AzureOpenAIProvider
from ragdx.llm.providers.ollama_provider import OllamaProvider
from ragdx.llm.providers.openai_provider import OpenAIProvider
from ragdx.llm.registry import (
    DEFAULT_MODELS,
    build_provider,
    get_llm_callable,
    list_providers,
    register_provider,
)

__all__ = [
    "DEFAULT_MODELS",
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "LLMCallable",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "build_provider",
    "get_llm_callable",
    "list_providers",
    "register_provider",
]
