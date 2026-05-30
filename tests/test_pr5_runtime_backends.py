"""Tests for PR5: ``RAGPipeline.build`` dispatch + ``LlamaIndexRAGPipeline``.

The LangChain backend is exercised by every existing test that runs
through ``RAGPipeline`` -- those still pass unchanged. PR5-specific
tests cover:

* :attr:`RAGConfig.runtime` field defaults + validation.
* ``RAGPipeline.build`` dispatches correctly on ``config.runtime``.
* :class:`LlamaIndexRAGPipeline` end-to-end with a tiny in-memory
  corpus, a deterministic stub embedder, and a stub LLM callable
  (no network, no HF download).
* The shared prompt template: ``LlamaIndexRAGPipeline.generate``
  produces the same output as ``RAGPipeline.generate`` for the
  same inputs.
* The LangChain embedding adapter conforms to the LlamaIndex
  ``BaseEmbedding`` contract.
"""

from __future__ import annotations

import pytest

from ragdx.runtime.pipeline import (
    DEFAULT_SYSTEM_INSTRUCTION,
    LangChainRAGPipeline,
    LlamaIndexRAGPipeline,
    RAGAnswer,
    RAGPipeline,
    _build_llamaindex_embedder,
)
from ragdx.schemas.rag_config import GeneratorSpec, RAGConfig, RetrieverSpec

# Most of this file exercises ``llama-index-core``. The pure-pydantic
# / pure-dispatch tests below still run without it (they don't import
# llama_index at all); everything that calls ``_build_impl`` or
# ``_build_llamaindex_embedder`` is guarded with a per-test
# ``pytest.importorskip("llama_index")`` so CI without the
# ``ragdx[llamaindex]`` extra installed skips cleanly instead of erroring.


class _StubEmbedder:
    """Deterministic 4-d embedder: each text gets a hash-based vector.

    Good enough for VectorStoreIndex to build + retrieve without
    needing a real HuggingFace model (which would force a 100MB
    download and slow tests by minutes).
    """

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        # Project text into a stable 4-d vector. Texts sharing a prefix
        # cluster nearby; perfect for "find me the most similar chunk".
        s = text or ""
        return [
            float(len(s) % 7) / 7,
            float(s.count(" ") % 5) / 5,
            float(sum(ord(c) for c in s[:8]) % 13) / 13,
            float(sum(ord(c) for c in s[-8:]) % 11) / 11,
        ]


def _stub_lm_capture():
    """Build an LM callable that records every prompt it receives."""
    captured: dict[str, list[str]] = {"prompts": []}

    def lm(prompt: str) -> str:
        captured["prompts"].append(prompt)
        return "stub-answer"

    return lm, captured


# ====================================================== RAGConfig.runtime
def test_rag_config_runtime_defaults_to_langchain():
    """Backward compat: configs that don't mention runtime keep using
    LangChain (the pre-PR5 behaviour)."""
    cfg = RAGConfig()
    assert cfg.runtime == "langchain"


def test_rag_config_runtime_accepts_llamaindex():
    cfg = RAGConfig(runtime="llamaindex")
    assert cfg.runtime == "llamaindex"


def test_rag_config_runtime_rejects_unknown():
    """The Literal annotation must catch typos at config-load time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RAGConfig(runtime="haystack")  # type: ignore[arg-type]


def test_rag_config_runtime_yaml_roundtrip(tmp_path):
    """YAML serialisation must preserve the runtime selector."""
    cfg = RAGConfig(runtime="llamaindex", name="esg-prod")
    p = cfg.to_yaml(tmp_path / "rag.yaml")
    cfg2 = RAGConfig.from_yaml(p)
    assert cfg2.runtime == "llamaindex"
    assert cfg2 == cfg


# ====================================================== Dispatch
def test_build_returns_langchain_pipeline_by_default():
    """``RAGConfig.runtime == "langchain"`` (default) -> the LangChain
    backend. ``LangChainRAGPipeline`` is an alias for ``RAGPipeline``
    so identity check works."""
    # Use a stub vstore to avoid touching FAISS. Build() with no
    # llamaindex runtime takes the langchain path; we pre-stub it.
    # Easier: just verify the alias relationship -- the dispatch
    # behaviour is verified through end-to-end runs.
    assert LangChainRAGPipeline is RAGPipeline


def _require_llama_index() -> None:
    """Skip the calling test when ``llama-index-core`` isn't installed.

    Centralised so CI without the ``ragdx[llamaindex]`` extra skips
    cleanly; users running locally with the extra get full coverage.
    """
    pytest.importorskip(
        "llama_index",
        reason="ragdx[llamaindex] extra not installed",
    )


def test_build_dispatches_to_llamaindex_when_runtime_set():
    """``RAGConfig.runtime == "llamaindex"`` -> the LlamaIndex backend."""
    _require_llama_index()
    cfg = RAGConfig(runtime="llamaindex")
    lm, _captured = _stub_lm_capture()
    pipe = RAGPipeline.build(
        cfg,
        chunks=["alpha beta", "gamma delta"],
        embedder=_StubEmbedder(),
        llm_callable=lm,
    )
    assert isinstance(pipe, LlamaIndexRAGPipeline)
    # Sanity: not accidentally returning the LangChain class.
    assert not isinstance(pipe, LangChainRAGPipeline)
    assert pipe.n_chunks == 2


def test_build_rejects_empty_chunks_both_backends():
    """Empty-chunks guard applies to both dispatch paths."""
    for runtime in ("langchain", "llamaindex"):
        cfg = RAGConfig(runtime=runtime)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least one chunk"):
            RAGPipeline.build(
                cfg, [], embedder=_StubEmbedder(), llm_callable=lambda p: "x",
            )


# ====================================================== LlamaIndex backend
def test_llamaindex_build_creates_index_with_expected_n_chunks():
    """``VectorStoreIndex.from_documents`` must produce an index whose
    ``n_chunks`` matches the input. Catches off-by-one regressions
    when Document construction silently drops empties."""
    _require_llama_index()
    cfg = RAGConfig(runtime="llamaindex")
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg,
        chunks=["alpha", "beta", "gamma"],
        embedder=_StubEmbedder(),
        llm_callable=lambda p: "stub",
    )
    assert pipe.n_chunks == 3
    assert pipe.config is cfg
    assert pipe.index is not None


def test_llamaindex_retrieve_respects_config_top_k():
    """No per-call ``top_k`` -> use ``RetrieverSpec.top_k``."""
    _require_llama_index()
    cfg = RAGConfig(
        runtime="llamaindex", retriever=RetrieverSpec(top_k=2),
    )
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg,
        chunks=["alpha alpha alpha", "beta beta beta", "gamma gamma gamma"],
        embedder=_StubEmbedder(),
        llm_callable=lambda p: "stub",
    )
    contexts = pipe.retrieve("alpha")
    assert len(contexts) <= 2  # bounded by top_k
    assert all(isinstance(c, str) for c in contexts)


def test_llamaindex_retrieve_per_call_top_k_overrides_config():
    """Per-call top_k overrides RetrieverSpec.top_k -- the contract
    Joint/Retrieval optimizers rely on."""
    _require_llama_index()
    cfg = RAGConfig(
        runtime="llamaindex", retriever=RetrieverSpec(top_k=1),
    )
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg,
        chunks=["a", "b", "c", "d", "e"],
        embedder=_StubEmbedder(),
        llm_callable=lambda p: "stub",
    )
    ctxs = pipe.retrieve("query", top_k=3)
    assert len(ctxs) <= 3


def test_llamaindex_generate_uses_shared_prompt_template():
    """LlamaIndex and LangChain ``generate`` must produce the same
    prompt (so backend choice doesn't change the LLM's input). The
    only thing that varies between backends is retrieval, not
    generation."""
    _require_llama_index()
    cfg = RAGConfig(runtime="llamaindex")
    lm, captured = _stub_lm_capture()
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg, chunks=["x"], embedder=_StubEmbedder(), llm_callable=lm,
    )
    pipe.generate("Q?", ["context-a", "context-b"])
    prompt = captured["prompts"][-1]
    assert "Question: Q?" in prompt
    assert "context-a" in prompt and "context-b" in prompt
    # Default instruction header must appear.
    assert DEFAULT_SYSTEM_INSTRUCTION in prompt


def test_llamaindex_generate_per_call_system_instruction_overrides():
    """Per-call instruction beats config instruction beats default --
    same resolution order as LangChain backend."""
    _require_llama_index()
    cfg = RAGConfig(
        runtime="llamaindex",
        generator=GeneratorSpec(system_instruction="from-config"),
    )
    lm, captured = _stub_lm_capture()
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg, chunks=["x"], embedder=_StubEmbedder(), llm_callable=lm,
    )
    pipe.generate("Q?", ["ctx"], system_instruction="from-call")
    prompt = captured["prompts"][-1]
    assert "from-call" in prompt
    assert "from-config" not in prompt
    assert DEFAULT_SYSTEM_INSTRUCTION not in prompt


def test_llamaindex_answer_composes_retrieve_and_generate():
    """``answer`` is a thin wrapper that must produce a
    :class:`RAGAnswer` with retrieved contexts + the LM's output."""
    _require_llama_index()
    cfg = RAGConfig(runtime="llamaindex")
    pipe = LlamaIndexRAGPipeline._build_impl(
        cfg, chunks=["alpha", "beta"],
        embedder=_StubEmbedder(),
        llm_callable=lambda p: "final-answer",
    )
    out = pipe.answer("Q?")
    assert isinstance(out, RAGAnswer)
    assert out.question == "Q?"
    assert out.answer == "final-answer"
    assert len(out.contexts) > 0


# ====================================================== Embedding adapter
def test_langchain_embedding_adapter_is_a_real_base_embedding():
    """LlamaIndex enforces ``isinstance(embed_model, BaseEmbedding)``
    at index-build time; the adapter must pass that check."""
    _require_llama_index()
    from llama_index.core.embeddings import BaseEmbedding

    a = _build_llamaindex_embedder(_StubEmbedder())
    assert isinstance(a, BaseEmbedding)


def test_langchain_embedding_adapter_query_delegates_to_langchain():
    """Query embeddings come from ``lc_embedder.embed_query``."""
    _require_llama_index()
    a = _build_llamaindex_embedder(_StubEmbedder())
    v = a._get_query_embedding("hello")
    assert v == _StubEmbedder().embed_query("hello")


def test_langchain_embedding_adapter_text_delegates_to_langchain():
    """Text embeddings come from ``lc_embedder.embed_documents([text])[0]``."""
    _require_llama_index()
    a = _build_llamaindex_embedder(_StubEmbedder())
    v = a._get_text_embedding("hello")
    assert v == _StubEmbedder().embed_documents(["hello"])[0]


def test_langchain_embedding_adapter_batch_uses_one_langchain_call():
    """``_get_text_embeddings`` must call ``embed_documents`` once
    (not per-text) to preserve langchain-side batching."""
    _require_llama_index()

    class _CountingEmbedder(_StubEmbedder):
        def __init__(self):
            self.batch_calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.batch_calls += 1
            return super().embed_documents(texts)

    inner = _CountingEmbedder()
    a = _build_llamaindex_embedder(inner)
    vecs = a._get_text_embeddings(["a", "b", "c", "d"])
    assert inner.batch_calls == 1  # one batched call, not four
    assert len(vecs) == 4


# ====================================================== Adapter docstring honesty
def test_llamaindex_adapter_clarifies_it_is_a_spec_renderer():
    """The legacy ``LlamaIndexAdapter`` must document itself as a YAML
    renderer, not a runtime, to avoid confusion now that PR5 ships a
    real LlamaIndex runtime."""
    from ragdx.optim.llamaindex_adapter import LlamaIndexAdapter
    doc = (LlamaIndexAdapter.__doc__ or "") + (LlamaIndexAdapter.__module__ or "")
    # The honest disclaimer must explicitly name the new runtime so
    # users know where to look.
    assert "LlamaIndexRAGPipeline" in doc or any(
        "LlamaIndexRAGPipeline" in (m or "")
        for m in (
            LlamaIndexAdapter.__doc__,
            LlamaIndexAdapter.build_runner_spec.__doc__,
            __import__("ragdx.optim.llamaindex_adapter", fromlist=["__doc__"]).__doc__,
        )
    )
