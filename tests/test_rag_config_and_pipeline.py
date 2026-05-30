"""Tests for the PR1 contract: :class:`RAGConfig` (schemas) + :class:`RAGPipeline` (runtime).

PR1 introduces these as the single source of truth for "how to run a
RAG" and "how to describe a RAG". These tests lock in:

* default values match what ``run_experiment`` had before the refactor
  (so a bare ``RAGConfig()`` reproduces the demo pipeline);
* YAML round-trip is lossless;
* :meth:`RAGConfig.with_override` swaps one stage cleanly without
  mutating the original;
* :class:`RAGPipeline.retrieve` / ``generate`` / ``answer`` respect the
  per-call overrides BO + DSPy depend on.

Live integrations (FAISS index construction, real HF embeddings) are
exercised by the demo bundles and end-to-end pipeline runs; these unit
tests use stub vstore + stub LLM callables so they stay CI-cheap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ragdx.runtime.pipeline import DEFAULT_SYSTEM_INSTRUCTION, RAGAnswer, RAGPipeline
from ragdx.schemas.rag_config import (
    ChunkerSpec,
    CorpusSpec,
    GeneratorSpec,
    JudgeSpec,
    RAGConfig,
    RetrieverSpec,
)


# ============================================================ RAGConfig
def test_defaults_reproduce_pre_refactor_demo_pipeline():
    """``RAGConfig()`` defaults must match what ``run_experiment`` used
    inline before the PR1 refactor. Any drift here is a backwards-
    incompatible change to the demo behaviour and needs a deliberate
    bump."""
    cfg = RAGConfig()
    assert cfg.chunker.strategy == "recursive"
    assert cfg.chunker.chunk_size == 512
    assert cfg.chunker.chunk_overlap == 50
    assert cfg.embedder.kind == "huggingface"
    assert cfg.embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.embedder.normalize is True
    assert cfg.retriever.vectorstore == "faiss"
    assert cfg.retriever.search_type == "similarity"
    assert cfg.retriever.top_k == 5
    assert cfg.retriever.reranker == "none"
    assert cfg.generator.provider == "litellm"
    assert cfg.generator.model == "openai/glm-4-flash"
    assert cfg.generator.system_instruction is None  # -> falls back to DEFAULT
    assert cfg.judge.llm_max_concurrent == 2
    assert cfg.judge.llm_max_retries == 5


def test_yaml_roundtrip_is_lossless(tmp_path: Path):
    """YAML write+read must produce an equal ``RAGConfig``. Production
    handoff and version control rely on this."""
    cfg = RAGConfig(
        name="esg-analyst",
        corpus=CorpusSpec(kind="pdf", path="docs/report.pdf"),
        chunker=ChunkerSpec(chunk_size=1024, chunk_overlap=100),
        retriever=RetrieverSpec(top_k=8, search_type="mmr"),
        generator=GeneratorSpec(
            model="openai/glm-4-flash",
            system_instruction="You are an ESG analyst.",
            temperature=0.05,
        ),
        judge=JudgeSpec(llm_max_concurrent=6, llm_max_retries=8),
    )
    target = cfg.to_yaml(tmp_path / "rag.yaml")
    assert target.exists()
    cfg2 = RAGConfig.from_yaml(target)
    assert cfg2 == cfg


def test_scrubbed_for_commit_clears_api_keys():
    """A config produced by ``ragdx tune --write-optimized-config`` must
    not contain credentials -- otherwise users would leak API keys by
    committing the YAML. Locks in the security fix for the bug found
    in the e2e test on 2026-05-30."""
    cfg = RAGConfig(
        generator=GeneratorSpec(
            model="openai/glm-4-flash",
            api_base="https://example.com/v1",
            api_key="super-secret-generator-key",
            system_instruction="domain prompt",
        ),
        judge=JudgeSpec(
            model="openai/gpt-4o-mini",
            api_base="https://example.com/judge",
            api_key="super-secret-judge-key",
            llm_max_concurrent=4,
        ),
    )
    safe = cfg.scrubbed_for_commit()

    # Credentials gone.
    assert safe.generator.api_key is None
    assert safe.judge.api_key is None
    # The original is NOT mutated -- the caller may still need to use it.
    assert cfg.generator.api_key == "super-secret-generator-key"
    assert cfg.judge.api_key == "super-secret-judge-key"
    # Everything non-credential is preserved.
    assert safe.generator.model == "openai/glm-4-flash"
    assert safe.generator.api_base == "https://example.com/v1"
    assert safe.generator.system_instruction == "domain prompt"
    assert safe.judge.model == "openai/gpt-4o-mini"
    assert safe.judge.api_base == "https://example.com/judge"
    assert safe.judge.llm_max_concurrent == 4


def test_scrubbed_for_commit_yaml_round_trip_is_safe(tmp_path: Path):
    """End-to-end check: scrubbed config written to YAML produces an
    api_key-free file. Catches future refactors where someone forgets
    to scrub before serialising."""
    cfg = RAGConfig(
        generator=GeneratorSpec(api_key="LEAK-ME-IF-YOU-CAN"),
    )
    target = tmp_path / "post_tune_config.yaml"
    cfg.scrubbed_for_commit().to_yaml(target)
    raw = target.read_text(encoding="utf-8")
    assert "LEAK-ME-IF-YOU-CAN" not in raw
    assert "api_key: null" in raw


def test_with_override_swaps_one_stage_without_mutating_original():
    """The stage-swap helper that the future ``StageOptimizer`` family
    depends on. The override must not touch the source config."""
    base = RAGConfig(generator=GeneratorSpec(system_instruction="base"))
    swapped = base.with_override(retriever=RetrieverSpec(top_k=12))
    assert swapped.retriever.top_k == 12
    assert base.retriever.top_k == 5  # unchanged
    assert swapped.generator.system_instruction == "base"  # carried over


def test_extra_fields_in_yaml_are_rejected(tmp_path: Path):
    """``extra='forbid'`` on every Spec catches typos / drifted field
    names early. Without it a misspelled key would silently fall through."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "chunker:\n  chunk_size: 512\n  chunk_overlpa: 50\n",  # typo: 'overlpa'
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        RAGConfig.from_yaml(bad)


def test_corpus_spec_recursive_items_for_multi_corpus():
    """``CorpusSpec(kind="multi", items=[...])`` represents multi-corpus
    runs (PDF + JSONL pooled together). Recursion must work after
    ``model_rebuild`` runs at module load."""
    cfg = RAGConfig(corpus=CorpusSpec(
        kind="multi",
        items=[
            CorpusSpec(kind="pdf", path="a.pdf"),
            CorpusSpec(kind="jsonl", path="b.jsonl"),
        ],
    ))
    assert len(cfg.corpus.items) == 2
    assert cfg.corpus.items[0].kind == "pdf"
    assert cfg.corpus.items[1].kind == "jsonl"


# ========================================================== RAGPipeline
class _StubVStore:
    """A retriever that returns hardcoded contexts. No embedding work."""

    def __init__(self, contexts: list[str]):
        self.contexts = contexts
        self.last_k = None
        self.last_question = None

    def similarity_search(self, question: str, k: int):
        self.last_question = question
        self.last_k = k

        class _Doc:
            def __init__(self, c: str):
                self.page_content = c

        return [_Doc(c) for c in self.contexts[:k]]


def test_pipeline_retrieve_uses_config_top_k_by_default():
    """No per-call ``top_k`` -> the config's ``RetrieverSpec.top_k``
    decides. This is the production-default path."""
    vs = _StubVStore(["a", "b", "c", "d", "e", "f"])
    pipe = RAGPipeline(
        config=RAGConfig(retriever=RetrieverSpec(top_k=3)),
        vstore=vs, llm_callable=lambda p: "stub", n_chunks=6,
    )
    ctxs = pipe.retrieve("Q?")
    assert ctxs == ["a", "b", "c"]
    assert vs.last_k == 3


def test_pipeline_retrieve_per_call_top_k_overrides_config():
    """The retrieval-stage optimizer's contract: change ``top_k``
    without rebuilding the pipeline."""
    vs = _StubVStore(["a", "b", "c", "d", "e", "f"])
    pipe = RAGPipeline(
        config=RAGConfig(retriever=RetrieverSpec(top_k=3)),
        vstore=vs, llm_callable=lambda p: "stub", n_chunks=6,
    )
    pipe.retrieve("Q?", top_k=5)
    assert vs.last_k == 5  # not 3


def test_pipeline_generate_with_no_contexts_uses_placeholder():
    """Empty context list must not produce a malformed prompt."""
    captured = {}
    pipe = RAGPipeline(
        config=RAGConfig(),
        vstore=object(),
        llm_callable=lambda p: captured.update({"prompt": p}) or "stub",
        n_chunks=0,
    )
    pipe.generate("What is RAG?", [])
    assert "(no context)" in captured["prompt"]


def test_pipeline_generate_includes_default_instruction_when_unset():
    """``GeneratorSpec.system_instruction=None`` -> package default."""
    captured = {}
    pipe = RAGPipeline(
        config=RAGConfig(),  # GeneratorSpec.system_instruction defaults to None
        vstore=object(),
        llm_callable=lambda p: captured.update({"prompt": p}) or "stub",
        n_chunks=0,
    )
    pipe.generate("Q?", ["context-x"])
    assert DEFAULT_SYSTEM_INSTRUCTION in captured["prompt"]
    assert "context-x" in captured["prompt"]


def test_pipeline_answer_combines_retrieve_and_generate():
    """The ``evaluate`` path's main loop will call ``.answer()``;
    verify it composes correctly."""
    vs = _StubVStore(["ctx-1", "ctx-2"])
    pipe = RAGPipeline(
        config=RAGConfig(retriever=RetrieverSpec(top_k=2)),
        vstore=vs, llm_callable=lambda p: "stub-answer", n_chunks=2,
    )
    out = pipe.answer("Q?")
    assert isinstance(out, RAGAnswer)
    assert out.question == "Q?"
    assert out.contexts == ["ctx-1", "ctx-2"]
    assert out.answer == "stub-answer"


def test_pipeline_build_rejects_empty_chunks():
    """Building a pipeline with no chunks is a programming error --
    fail loudly instead of producing a vstore with zero docs."""
    with pytest.raises(ValueError, match="at least one chunk"):
        RAGPipeline.build(
            RAGConfig(), [], embedder=object(), llm_callable=lambda p: "x",
        )


def test_pipeline_build_rejects_unsupported_vectorstore():
    """Future-proofing: a config with a vectorstore kind we don't
    implement yet must error at build time, not at retrieve time."""
    cfg = RAGConfig()
    # bypass pydantic Literal check by replacing the field directly
    cfg.retriever.vectorstore = "qdrant"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported vectorstore"):
        RAGPipeline.build(
            cfg, ["c"], embedder=object(), llm_callable=lambda p: "x",
        )
