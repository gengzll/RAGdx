"""First-class representation of a production RAG configuration.

A :class:`RAGConfig` is the *contract* between a user's production RAG
application and ragdx. Six per-stage ``*Spec`` dataclasses describe
each layer of the pipeline; together they fully reproduce how the
system answers a question.

This module is intentionally light on runtime behaviour -- the actual
"how to build a vstore / retrieve / generate from these specs" lives
in :mod:`ragdx.runtime.pipeline`. Keeping config and behaviour
separate lets us:

* serialize a RAG setup to a single YAML file (production handoff,
  reproducibility, version control)
* override a single stage in-place ("vary only the retriever") -- the
  basis for stage-targeted optimization
* compare configurations diff-style without dragging in heavy
  dependencies (sklearn / langchain / etc.)

YAML round-trip example::

    cfg = RAGConfig(
        corpus=CorpusSpec(kind="pdf", path="docs/report.pdf"),
        generator=GeneratorSpec(model="openai/gpt-4o-mini"),
    )
    cfg.to_yaml(Path("rag_config.yaml"))
    cfg2 = RAGConfig.from_yaml(Path("rag_config.yaml"))
    assert cfg == cfg2

Defaults across every Spec match the behaviour ragdx's
``run_experiment`` had before this module existed -- so passing
``RAGConfig()`` reproduces the demo pipeline 1:1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# Per-stage specs
# =====================================================================
class CorpusSpec(BaseModel):
    """Where the documents come from."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["pdf", "jsonl", "huggingface", "multi"] = "pdf"
    """Loader strategy:
    * ``pdf`` -- single PDF, parsed with :mod:`ragdx.loaders`.
    * ``jsonl`` -- JSONL with ``{text, source?}`` per line.
    * ``huggingface`` -- HuggingFace dataset reference (carries
      questions+contexts inline).
    * ``multi`` -- ``items`` holds an ordered list of corpus specs
      that get chunked separately and pooled.
    """

    path: str | None = None
    """Local path (``pdf`` / ``jsonl``) or HF dataset name (``huggingface``)."""

    items: list[CorpusSpec] = Field(default_factory=list)
    """For ``kind="multi"``: the constituent corpora."""

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> str | None:
        return None if v is None else str(v)


# Late binding for recursive CorpusSpec.items
CorpusSpec.model_rebuild()


class ChunkerSpec(BaseModel):
    """How raw documents are split into chunks."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["recursive", "fixed", "sentence", "passthrough"] = "recursive"
    """Splitter strategy.
    * ``recursive`` -- LangChain RecursiveCharacterTextSplitter style.
    * ``fixed`` -- equal-size character windows.
    * ``sentence`` -- sentence-aware splitter.
    * ``passthrough`` -- treat the input as already-chunked (one
      chunk per record), no further splitting. Used by HuggingFace
      datasets that already carry pre-built passages.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50


class EmbedderSpec(BaseModel):
    """The embedding model used for vector indexing."""

    # ``model_name`` would clash with pydantic's protected ``model_*``
    # namespace -- silence the warning since we own this name semantically.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: Literal["huggingface", "openai", "custom"] = "huggingface"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    """Apply L2 normalization on encoded vectors so cosine similarity =
    dot product. Matches FAISS default behaviour."""


class RetrieverSpec(BaseModel):
    """Vector store + retrieval policy."""

    model_config = ConfigDict(extra="forbid")

    vectorstore: Literal["faiss", "chroma"] = "faiss"
    search_type: Literal["similarity", "mmr"] = "similarity"
    top_k: int = 5
    """Default number of contexts retrieved per query. Optimizers can
    override per-call via ``RAGPipeline.retrieve(question, top_k=N)``."""

    reranker: Literal["none", "cohere", "bge"] = "none"


class GeneratorSpec(BaseModel):
    """The generator LLM + how it's prompted."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai", "anthropic", "litellm", "custom"] = "litellm"
    """Provider routing. ``litellm`` is the default because it
    transparently handles OpenAI-compatible endpoints (Zhipu, vLLM,
    LM Studio, ...) used in ragdx's demos."""

    model: str = "openai/glm-4-flash"
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str | None = None
    """If unset, runtime resolves from ``ZHIPU_API_KEY`` /
    ``OPENAI_API_KEY`` env vars."""

    system_instruction: str | None = None
    """RAG system prompt. ``None`` -> the package default
    (:data:`ragdx.experiments.DEFAULT_SYSTEM_INSTRUCTION`). Override
    for domain-specific guidance; MIPROv2 may evolve this further."""

    temperature: float = 0.01
    max_tokens: int = 350
    timeout: int = 60


class JudgeSpec(BaseModel):
    """The LLM used by ragas / DSPy as the evaluation judge.

    Defaults to the same provider as the generator but can be pinned
    to a stronger model (e.g. ``openai/gpt-4o-mini`` while generator
    runs on a cheaper local model).
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    """``None`` = use the generator's model. Override to use a
    stronger judge (recommended for production tuning)."""

    api_base: str | None = None
    api_key: str | None = None

    llm_max_concurrent: int = 2
    """Max concurrent judge LLM calls per evaluation batch. Default
    ``2`` is tuned for strict rate-limited endpoints. See
    :attr:`ragdx.experiments.ExperimentConfig.llm_max_concurrent`."""

    llm_max_retries: int = 5
    """Transport-layer retry budget per call."""


# =====================================================================
# RAGConfig: composite
# =====================================================================
class RAGConfig(BaseModel):
    """A complete, serializable description of a RAG configuration.

    Six stage specs combine into one object that fully reproduces how
    a RAG system answers a question. Pass to
    :class:`ragdx.runtime.pipeline.RAGPipeline` to actually run the
    pipeline; pass to a ``StageOptimizer`` to vary one stage while
    holding the others fixed.

    Defaults match ragdx's pre-RAGConfig demo behaviour, so
    ``RAGConfig()`` reproduces the ``run_experiment`` baseline exactly.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    """Optional human-readable label (shown in dashboards / reports)."""

    runtime: Literal["langchain", "llamaindex"] = "langchain"
    """Which RAG runtime executes this config.

    * ``"langchain"`` (default) -- FAISS index + LangChain similarity
      search. The path ragdx has used end-to-end since PR1.
    * ``"llamaindex"`` -- LlamaIndex's ``VectorStoreIndex`` +
      ``as_retriever``. Implemented in PR5; requires the
      ``ragdx[llamaindex]`` extra. Same ``EmbedderSpec`` /
      ``GeneratorSpec`` -- the embedding model and LLM endpoint are
      reused, only the index + retriever change.

    Adding new backends (e.g. ``"autorag"``, ``"haystack"``) means
    extending this Literal and registering a new
    :class:`RAGPipeline` subclass."""

    corpus: CorpusSpec = Field(default_factory=CorpusSpec)
    chunker: ChunkerSpec = Field(default_factory=ChunkerSpec)
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    retriever: RetrieverSpec = Field(default_factory=RetrieverSpec)
    generator: GeneratorSpec = Field(default_factory=GeneratorSpec)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)

    # ----- serialization helpers ------------------------------------
    def to_yaml(self, path: str | Path) -> Path:
        """Write the config to ``path`` (YAML). Returns the target path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load a config from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"RAGConfig YAML must be a mapping at the top level; got "
                f"{type(raw).__name__}"
            )
        return cls.model_validate(raw)

    def with_override(self, **stage_overrides: BaseModel) -> RAGConfig:
        """Return a copy of this config with one or more ``*Spec``s
        replaced. Used by stage-targeted optimizers to vary a single
        stage while keeping everything else identical.

        Example::

            base = RAGConfig.from_yaml("prod.yaml")
            trial = base.with_override(retriever=RetrieverSpec(top_k=12))
        """
        return self.model_copy(update=stage_overrides)


__all__ = [
    "ChunkerSpec",
    "CorpusSpec",
    "EmbedderSpec",
    "GeneratorSpec",
    "JudgeSpec",
    "RAGConfig",
    "RetrieverSpec",
]
