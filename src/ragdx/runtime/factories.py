"""Build the runtime objects ragdx needs from a :class:`RAGConfig`.

Pre-PR4, ``ragdx.experiments._build_runtime`` did this inline against
an ``ExperimentConfig``. That coupled every ``RAGPipeline``-using
codepath to the experiment workflow's particular config shape. PR4
extracts the actual factories so a ``ragdx evaluate`` / ``ragdx tune``
invocation can build a runtime straight from a user's
``rag_config.yaml`` -- no ``ExperimentConfig`` in sight.

What "runtime" means here
-------------------------

The :class:`RagdxRuntime` dataclass holds the live LLM clients
shared across every stage of an experiment / evaluation:

* ``llm_callable`` -- the generator LLM as a ``(prompt) -> str``
  callable (currently routed through litellm).
* ``dspy_lm`` -- the DSPy ``LM`` wrapping the same generator (used by
  GenerationOptimizer's MIPROv2 path).
* ``ragas_judge`` -- the judge LLM wrapped as a ``LangchainLLMWrapper``
  so ragas can call it.
* ``ragas_embeddings`` -- ragas-side embeddings (cosine similarity
  for context_precision / etc.).
* ``embeddings`` -- the LangChain-compatible embeddings used by FAISS
  for retrieval.
* ``ragas_run_config`` -- the throttle config that controls how many
  judge calls are in flight at once.
* ``system_instruction`` -- the resolved instruction header for the
  generator (config's value or the package default).
* ``llm_max_concurrent`` / ``llm_max_retries`` -- propagated knobs
  the BO + DSPy stages need for their own thread pools.

Same shape as the pre-PR4 ``ragdx.experiments._Runtime``; the latter
is now a re-export so existing internal references keep working.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ragdx.schemas.rag_config import (
    EmbedderSpec,
    GeneratorSpec,
    JudgeSpec,
    RAGConfig,
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the question using only the retrieved context.\n"
    "Be concise. Do not invent facts that are not in the context."
)
"""Mirrors :data:`ragdx.runtime.pipeline.DEFAULT_SYSTEM_INSTRUCTION`."""


@dataclass
class RagdxRuntime:
    """Live LLM clients + ragas throttle, shared across pipeline stages.

    Build via :func:`build_runtime`. All fields are intended to be
    treated as immutable for the lifetime of a run.
    """

    llm_callable: Callable[[str], str]
    dspy_lm: Any
    ragas_judge: Any
    ragas_embeddings: Any
    embeddings: Any  # langchain embeddings for FAISS
    llm_max_concurrent: int = 2
    llm_max_retries: int = 5
    ragas_run_config: Any = None
    # Optional deepeval judge built lazily by ``build_deepeval_judge``.
    # ``None`` when deepeval isn't installed; callers that picked
    # ``--evaluator deepeval`` or ``--dspy-metric geval`` must check.
    deepeval_judge: Any = None
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION


# =====================================================================
# Temperature clamp (idempotent, applied once at build_runtime time)
# =====================================================================
def apply_litellm_temperature_clamp(
    min_temp: float = 0.01, max_temp: float = 1.0,
) -> None:
    """Clamp every outgoing temperature to ``[min_temp, max_temp]``
    across all plausible call paths.

    Two clamps, two reasons:

    * Lower: GLM-4-Flash rejects ``temperature < 0.01``; DSPy / ragas
      internally override the temperature to ~1e-8 for deterministic
      output.
    * Upper: GLM-4-Flash rejects ``temperature > 1.0`` (OpenAI tops
      out at 2.0 but GLM is stricter); DSPy's COPRO ships
      ``init_temperature=1.4`` for the instruction proposer.

    We intercept at three layers so the clamp survives no matter
    which library makes the call:

    1. ``litellm.completion`` / ``batch_completion`` (DSPy + our
       own ``llm_callable``).
    2. ``openai.resources.chat.completions.Completions.create``
       (sync OpenAI client -- ``langchain_openai.ChatOpenAI`` sync path).
    3. ``openai.resources.chat.completions.AsyncCompletions.create``
       (async OpenAI client -- ragas' judge calls).

    Idempotent on each layer (sentinel attribute ``_ragdx_clamped``).
    """
    import litellm

    def _clamp(kw):
        t = kw.get("temperature")
        if t is None:
            return kw
        if 0 < t < min_temp:
            kw["temperature"] = min_temp
        elif t > max_temp:
            kw["temperature"] = max_temp
        return kw

    if not getattr(litellm.completion, "_ragdx_clamped", False):
        orig_completion = litellm.completion
        orig_batch = litellm.batch_completion

        def comp(*a, **kw):
            return orig_completion(*a, **_clamp(kw))

        def batch(*a, **kw):
            return orig_batch(*a, **_clamp(kw))

        comp._ragdx_clamped = True  # type: ignore[attr-defined]
        litellm.completion = comp
        litellm.batch_completion = batch

    try:
        from openai.resources.chat.completions import (
            AsyncCompletions,
            Completions,
        )
    except ImportError:  # pragma: no cover - openai pinning floor
        return

    if not getattr(Completions.create, "_ragdx_clamped", False):
        _orig_sync = Completions.create

        def _sync_create(self, *a, **kw):
            return _orig_sync(self, *a, **_clamp(kw))

        _sync_create._ragdx_clamped = True  # type: ignore[attr-defined]
        Completions.create = _sync_create  # type: ignore[method-assign]

    if not getattr(AsyncCompletions.create, "_ragdx_clamped", False):
        _orig_async = AsyncCompletions.create

        async def _async_create(self, *a, **kw):
            return await _orig_async(self, *a, **_clamp(kw))

        _async_create._ragdx_clamped = True  # type: ignore[attr-defined]
        AsyncCompletions.create = _async_create  # type: ignore[method-assign]


# =====================================================================
# Per-spec builders
# =====================================================================
def build_embedder(spec: EmbedderSpec) -> Any:
    """Return a LangChain-compatible embeddings object for ``spec``.

    Currently only the HuggingFace path is wired (matches what the
    pre-PR4 demo used); openai / custom are queued for follow-up PRs
    when there's a caller that actually exercises them.
    """
    if spec.kind == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=spec.model_name,
            encode_kwargs={"normalize_embeddings": spec.normalize},
        )
    raise NotImplementedError(
        f"EmbedderSpec.kind={spec.kind!r} not implemented yet. "
        "Supported: 'huggingface'."
    )


def build_llm_callable(spec: GeneratorSpec, *, max_retries: int = 5) -> Callable[[str], str]:
    """Wrap the generator described by ``spec`` as ``(prompt) -> str``.

    Routes through litellm so any OpenAI-compatible endpoint
    (Zhipu / vLLM / LM Studio / etc.) works without per-provider code.
    """
    import litellm

    def llm_callable(prompt: str) -> str:
        resp = litellm.completion(
            model=spec.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=spec.api_key,
            api_base=spec.api_base,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            timeout=spec.timeout,
            num_retries=max_retries,
        )
        return resp.choices[0].message.content or ""

    return llm_callable


def build_dspy_lm(spec: GeneratorSpec, *, max_retries: int = 5) -> Any:
    """Return a ``dspy.LM`` configured for the generator."""
    import dspy

    return dspy.LM(
        spec.model,
        api_key=spec.api_key,
        api_base=spec.api_base,
        temperature=spec.temperature,
        max_tokens=max(spec.max_tokens, 400),  # DSPy programs sometimes need more
        cache=False,
        # Transport-layer retries (litellm) — without this, a single
        # connection blip mid-optimization surfaced as an exception and
        # poisoned the answer set with error text.
        num_retries=max_retries,
    )


def build_ragas_judge(judge: JudgeSpec, fallback: GeneratorSpec) -> Any:
    """Build the ragas-side judge LLM (``LangchainLLMWrapper``).

    Falls back to ``fallback`` (the generator's spec) for any
    judge-side field left as ``None``. Recommended in production:
    pin ``judge.model`` to a stronger LLM than the generator.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    model = judge.model or fallback.model
    api_base = judge.api_base or fallback.api_base
    api_key = judge.api_key or fallback.api_key
    # Strip "openai/" prefix that LiteLLM uses for ChatOpenAI's model arg.
    chat_model = model.split("/", 1)[-1] if "/" in model else model
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=chat_model,
            api_key=api_key,
            base_url=api_base,
            temperature=fallback.temperature,
            timeout=fallback.timeout,
            max_retries=judge.llm_max_retries,
        )
    )


def build_ragas_embeddings(embedder: Any) -> Any:
    """Wrap a LangChain embeddings object for ragas."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(embedder)


def build_deepeval_judge(judge: JudgeSpec, fallback: GeneratorSpec) -> Any | None:
    """Build a deepeval-compatible judge LLM.

    DeepEval metrics accept ``model=<DeepEvalBaseLLM | str | None>``.
    For arbitrary OpenAI-compatible endpoints (Zhipu / vLLM / Ollama /
    DeepSeek / ...) we subclass ``DeepEvalBaseLLM`` to route through
    ``langchain_openai.ChatOpenAI``. The same wire as the ragas judge;
    separate function because deepeval's ABC contract
    (``load_model`` / ``generate`` / ``a_generate`` / ``get_model_name``)
    differs from ragas's ``LangchainLLMWrapper``.

    Returns ``None`` when deepeval (or langchain-openai) isn't
    installed -- callers that need a deepeval judge must guard
    accordingly.
    """
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        from langchain_openai import ChatOpenAI
    except Exception:  # pragma: no cover - depends on user env
        return None

    model = judge.model or fallback.model
    api_base = judge.api_base or fallback.api_base
    api_key = judge.api_key or fallback.api_key
    chat_model = model.split("/", 1)[-1] if "/" in model else model
    chat = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=api_base,
        temperature=fallback.temperature,
        timeout=fallback.timeout,
        max_retries=judge.llm_max_retries,
    )

    class _RagdxDeepEvalLM(DeepEvalBaseLLM):
        """Thin DeepEvalBaseLLM wrapper around a langchain ChatOpenAI."""

        def __init__(self, client: Any, name: str) -> None:
            self._client = client
            self._name = name

        def load_model(self) -> Any:
            return self._client

        @staticmethod
        def _to_text(response: Any) -> str:
            content = getattr(response, "content", response)
            if isinstance(content, list):
                # LangChain chat models occasionally return a list of
                # content blocks (text / tool_use dicts). Concatenate
                # the text-only parts.
                parts = []
                for c in content:
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, dict) and c.get("type") == "text":
                        parts.append(str(c.get("text") or ""))
                return "".join(parts)
            return str(content)

        def _parse_schema(self, text: str, schema: Any) -> Any:
            import json
            try:
                return schema.model_validate_json(text)
            except Exception:
                # Best-effort fallback: extract the first ``{...}`` block.
                start = text.find("{")
                end = text.rfind("}")
                if 0 <= start < end:
                    try:
                        return schema.model_validate(
                            json.loads(text[start:end + 1])
                        )
                    except Exception:
                        pass
                # Last resort: hand the raw text back; some deepeval
                # metrics tolerate this, others will retry.
                return text

        def generate(self, prompt: str, schema: Any | None = None) -> Any:
            resp = self._client.invoke(prompt)
            text = self._to_text(resp)
            return text if schema is None else self._parse_schema(text, schema)

        async def a_generate(self, prompt: str, schema: Any | None = None) -> Any:
            resp = await self._client.ainvoke(prompt)
            text = self._to_text(resp)
            return text if schema is None else self._parse_schema(text, schema)

        def get_model_name(self) -> str:
            return self._name

    return _RagdxDeepEvalLM(chat, chat_model)


def build_ragas_run_config(judge: JudgeSpec) -> Any | None:
    """Build the ragas throttle config from ``JudgeSpec``.

    Returns ``None`` when ragas isn't installed -- callers that need
    it must guard accordingly.
    """
    try:
        from ragas.run_config import RunConfig as _RagasRunConfig
    except ImportError:  # pragma: no cover - ragas optional
        return None
    return _RagasRunConfig(
        max_workers=judge.llm_max_concurrent,
        max_retries=8,
        max_wait=30,
        timeout=180,
    )


# =====================================================================
# Top-level: build everything from a RAGConfig
# =====================================================================
def build_runtime(config: RAGConfig) -> RagdxRuntime:
    """Build every live client / wrapper a ragdx run needs.

    Side effects (matches the pre-PR4 ``_build_runtime`` behaviour
    exactly so the experiment workflow stays byte-identical):

    * Sets ``OPENAI_API_KEY`` and ``OPENAI_API_BASE`` in ``os.environ``
      so libraries that read them directly (DSPy, langchain_openai's
      ``ChatOpenAI`` constructor in some paths) find them.
    * Sets ``TOKENIZERS_PARALLELISM=false`` and
      ``HF_HUB_DISABLE_PROGRESS_BARS=1`` -- the sentence-transformers
      defaults are noisy in CI / pipelines.
    * Applies the global temperature clamp once.
    """
    gen = config.generator
    if gen.api_key:
        os.environ["OPENAI_API_KEY"] = gen.api_key
    os.environ["OPENAI_API_BASE"] = gen.api_base
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    apply_litellm_temperature_clamp()

    embedder = build_embedder(config.embedder)
    llm_callable = build_llm_callable(gen, max_retries=config.judge.llm_max_retries)
    dspy_lm = build_dspy_lm(gen, max_retries=config.judge.llm_max_retries)
    ragas_embeddings = build_ragas_embeddings(embedder)
    ragas_judge = build_ragas_judge(config.judge, fallback=gen)
    ragas_run_config = build_ragas_run_config(config.judge)
    # Best-effort: ``build_deepeval_judge`` returns ``None`` if deepeval
    # isn't installed, so this is a no-op when the user didn't opt in
    # to the deepeval extra. Zero cost when unused.
    deepeval_judge = build_deepeval_judge(config.judge, fallback=gen)

    return RagdxRuntime(
        llm_callable=llm_callable,
        dspy_lm=dspy_lm,
        ragas_judge=ragas_judge,
        ragas_embeddings=ragas_embeddings,
        embeddings=embedder,
        llm_max_concurrent=config.judge.llm_max_concurrent,
        llm_max_retries=config.judge.llm_max_retries,
        ragas_run_config=ragas_run_config,
        deepeval_judge=deepeval_judge,
        system_instruction=gen.system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
    )


__all__ = [
    "DEFAULT_SYSTEM_INSTRUCTION",
    "RagdxRuntime",
    "apply_litellm_temperature_clamp",
    "build_deepeval_judge",
    "build_dspy_lm",
    "build_embedder",
    "build_llm_callable",
    "build_ragas_embeddings",
    "build_ragas_judge",
    "build_ragas_run_config",
    "build_runtime",
]
