"""Provider catalog for the LLM endpoint used by ragdx's
generator + judge.

ragdx talks to the model via **LiteLLM** for the generator (DSPy +
BO inner-loop generation) and **langchain-openai** for the ragas
judge. Both libraries are OpenAI-protocol-compatible: any endpoint
that speaks the OpenAI chat-completions API works without code
changes -- you just need three things in your ``RAGConfig.generator``
block (and optionally ``judge``):

* ``model`` -- the LiteLLM-prefixed model identifier.
  ``openai/<name>`` is the universal "OpenAI-protocol" prefix and
  covers OpenAI itself, Zhipu GLM, vLLM / SGLang / Ollama /
  LM Studio, Together AI, Groq, Fireworks, DeepSeek, Moonshot, and
  most OpenAI-compatible vendors. Anthropic / Bedrock / Vertex /
  Azure each have their own LiteLLM prefix because they're not
  OpenAI-protocol -- see the catalog below.
* ``api_base`` -- the chat-completions endpoint base URL. Leave
  empty for providers LiteLLM auto-resolves (OpenAI, Anthropic).
* ``api_key`` -- the credential. Prefer setting it via env var
  rather than putting it in the YAML; ragdx falls back through a
  per-provider env-var chain (see :data:`ENV_VAR_FALLBACK_CHAIN`).

This module is informational -- it does not configure anything by
itself. Use :func:`provider_template` to copy a known-good config
snippet into your YAML, or just consult the catalog when you read
this file.

Example -- switch the ESG demo from ZHIPU to OpenAI::

    # rag_config.yaml
    generator:
      provider: litellm
      model: openai/gpt-4o-mini          # was: openai/glm-4-flash
      api_base: null                     # OpenAI default resolves
      api_key: null                      # falls back to OPENAI_API_KEY env
      temperature: 0.01
      max_tokens: 800
      timeout: 60
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    """One row of the provider catalog.

    Attributes
    ----------
    label:
        Human-readable short name (``"openai"``, ``"zhipu"`` ...).
    api_base:
        Chat-completions base URL. ``None`` means "let LiteLLM /
        the SDK resolve it" -- correct for OpenAI and Anthropic.
    model_prefix:
        LiteLLM model prefix. ``"openai/"`` means "OpenAI-protocol
        endpoint" (covers the bulk of OpenAI-compatible vendors).
        Anthropic, Bedrock, Vertex, Azure each have their own
        because their wire protocol differs.
    sample_model:
        Concrete model id to drop into the YAML's ``model`` field
        as a working starting point (e.g. ``"openai/gpt-4o-mini"``).
        Pick a cheap / fast model for the sample.
    env_vars:
        Env vars LiteLLM / the SDK looks at, in priority order.
        ragdx CLI commands fall back through these when ``api_key``
        is ``null`` in the YAML and ``--api-key`` isn't passed.
    notes:
        Free-form caveats (e.g. "AWS region required", "rate
        limit ~2 RPS").
    """

    label: str
    api_base: str | None
    model_prefix: str
    sample_model: str
    env_vars: tuple[str, ...]
    notes: str = ""


# Public catalog: keep this list in sync with what the runtime
# clamps support (see ``apply_litellm_temperature_clamp`` in
# ``runtime.factories`` -- some providers ignore temperature > 1.0).
CATALOG: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        label="OpenAI",
        api_base=None,  # SDK default https://api.openai.com/v1
        model_prefix="openai/",
        sample_model="openai/gpt-4o-mini",
        env_vars=("OPENAI_API_KEY",),
        notes="Default. Set OPENAI_API_KEY. Both generator and judge work out of the box.",
    ),
    "azure": ProviderSpec(
        label="Azure OpenAI",
        api_base="https://<your-resource>.openai.azure.com",
        model_prefix="azure/",
        sample_model="azure/gpt-4o-mini",
        env_vars=(
            "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY",
        ),
        notes="Also requires AZURE_API_VERSION env var (e.g. 2024-08-01-preview).",
    ),
    "anthropic": ProviderSpec(
        label="Anthropic",
        api_base=None,  # SDK default https://api.anthropic.com
        model_prefix="anthropic/",
        sample_model="anthropic/claude-3-5-haiku-latest",
        env_vars=("ANTHROPIC_API_KEY",),
        notes=(
            "Generator (LiteLLM) works directly. The ragas judge uses "
            "langchain-openai which is OpenAI-protocol only -- set "
            "JudgeSpec.model to a different OpenAI-compatible LM for "
            "the judge, OR install langchain-anthropic and adapt "
            "``build_ragas_judge`` in runtime/factories.py."
        ),
    ),
    "zhipu": ProviderSpec(
        label="Zhipu GLM",
        api_base="https://open.bigmodel.cn/api/paas/v4",
        model_prefix="openai/",
        sample_model="openai/glm-4-flash",
        env_vars=("ZHIPU_API_KEY", "OPENAI_API_KEY"),
        notes=(
            "GLM-4-Flash is rate-limited (~2 RPS); keep "
            "llm_max_concurrent at 2. Temperature is clamped to "
            "[0.01, 1.0] by ragdx -- GLM rejects values outside."
        ),
    ),
    "moonshot": ProviderSpec(
        label="Moonshot Kimi",
        api_base="https://api.moonshot.cn/v1",
        model_prefix="openai/",
        sample_model="openai/moonshot-v1-8k",
        env_vars=("MOONSHOT_API_KEY", "OPENAI_API_KEY"),
    ),
    "deepseek": ProviderSpec(
        label="DeepSeek",
        api_base="https://api.deepseek.com/v1",
        model_prefix="openai/",
        sample_model="openai/deepseek-chat",
        env_vars=("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    ),
    "qwen": ProviderSpec(
        label="DashScope / Qwen",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_prefix="openai/",
        sample_model="openai/qwen-plus",
        env_vars=("DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    ),
    "ollama": ProviderSpec(
        label="Ollama (local)",
        api_base="http://localhost:11434/v1",
        model_prefix="openai/",
        sample_model="openai/llama3.1:8b",
        env_vars=(),
        notes=(
            "No API key needed. Pull the model first: "
            "``ollama pull llama3.1:8b``. The judge will also try "
            "this endpoint -- works for cheap iteration but not "
            "production judging."
        ),
    ),
    "vllm": ProviderSpec(
        label="vLLM / SGLang (local)",
        api_base="http://localhost:8000/v1",
        model_prefix="openai/",
        sample_model="openai/<your-served-model-name>",
        env_vars=(),
        notes=(
            "Run vLLM with --served-model-name to match the YAML's "
            "model field after the openai/ prefix."
        ),
    ),
    "groq": ProviderSpec(
        label="Groq",
        api_base="https://api.groq.com/openai/v1",
        model_prefix="openai/",
        sample_model="openai/llama-3.3-70b-versatile",
        env_vars=("GROQ_API_KEY", "OPENAI_API_KEY"),
    ),
    "together": ProviderSpec(
        label="Together AI",
        api_base="https://api.together.xyz/v1",
        model_prefix="openai/",
        sample_model="openai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        env_vars=("TOGETHER_API_KEY", "OPENAI_API_KEY"),
    ),
}


# ragdx's existing env-var fallback chain (matches the historical
# resolution order used by the CLI commands).
ENV_VAR_FALLBACK_CHAIN: tuple[str, ...] = (
    "ZHIPU_API_KEY", "OPENAI_API_KEY",
)


def provider_template(name: str, *, project_label: str = "my-rag") -> str:
    """Return a paste-ready RAGConfig YAML stub for the named provider.

    Pass the result to :func:`pathlib.Path.write_text` or print it for
    a copy-paste workflow. The corpus/chunker/embedder blocks are
    intentionally left as the simplest sensible defaults so the
    snippet stays focused on the LLM provider switch.

    Raises
    ------
    KeyError: if ``name`` isn't in :data:`CATALOG`.
    """
    spec = CATALOG[name]
    api_base_line = (
        f"  api_base: {spec.api_base}" if spec.api_base
        else "  api_base: null    # let the SDK use the provider default"
    )
    env_hint = (
        f"# env var(s) used as fallback: {', '.join(spec.env_vars)}"
        if spec.env_vars
        else "# no API key needed for this provider"
    )
    return f"""# RAGConfig for {spec.label}.
# {env_hint}
{f'# Notes: {spec.notes}' if spec.notes else ''}
name: {project_label}
runtime: langchain
corpus:
  kind: pdf
  path: your_corpus.pdf
chunker:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 50
embedder:
  kind: huggingface
  model_name: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
retriever:
  vectorstore: faiss
  search_type: similarity
  top_k: 3
  reranker: none
generator:
  provider: litellm
  model: {spec.sample_model}
{api_base_line}
  api_key: null   # set the env var instead
  temperature: 0.01
  max_tokens: 800
  timeout: 60
judge:
  model: null     # falls back to the generator
  api_base: null
  api_key: null
  llm_max_concurrent: 2
  llm_max_retries: 5
"""


__all__ = ["CATALOG", "ENV_VAR_FALLBACK_CHAIN", "ProviderSpec", "provider_template"]
