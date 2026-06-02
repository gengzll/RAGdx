"""``RAGPipeline`` -- the one place ragdx actually runs a RAG query.

Before this module, the BO loop in ``ragdx.experiments`` had its own
inline ``_build_vstore`` / ``_retrieve`` / ``_generate_answer``, and
``ragdx.optim.executor``'s execute-mode path expected user-supplied
runner scripts to do the equivalent themselves. Two implementations of
"how to RAG" meant any change (a new vstore, a different prompt
header, a tweak to retrieval) had to be made twice.

``RAGPipeline`` is the contract: a single class that, given a
:class:`ragdx.schemas.rag_config.RAGConfig`, a pool of chunks, and an
LLM callable, produces ``(question) -> (contexts, answer)``. Every
ragdx code path that runs RAG goes through this -- BO trials, DSPy
A/B, future stage-targeted optimizers, future ``ragdx evaluate``.

Design choices:

* **Chunking is external.** Each corpus kind (PDF, JSONL, HF dataset,
  multi) has its own loader that knows how to read raw documents and
  produce chunks; we don't want to bake that into the pipeline. The
  caller supplies ``chunks: list[str]``.

* **Per-call overrides.** ``retrieve(question, top_k=N)`` and
  ``generate(question, contexts, system_instruction=...)`` accept
  overrides so a single pipeline can serve a Bayesian search varying
  ``top_k``, or a DSPy A/B varying the prompt, without rebuilding the
  vstore.

* **Heavy imports inside class methods.** Importing
  ``langchain_community.vectorstores.FAISS`` at module load would
  drag in 100+ MB of dependencies for any consumer of this module --
  including the dashboard, the report renderer, the CLI ``--help``
  text. We pay the import cost only when ``RAGPipeline.build`` is
  actually called.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ragdx.schemas.rag_config import RAGConfig

DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the question using only the retrieved context.\n"
    "Be concise. Do not invent facts that are not in the context."
)
"""Fallback system instruction when neither
:attr:`ragdx.schemas.rag_config.GeneratorSpec.system_instruction` nor
a per-call override is supplied. Kept in sync with
:data:`ragdx.experiments.DEFAULT_SYSTEM_INSTRUCTION`; PR2/3 will
collapse these into a single import."""


def _render_few_shot_demos(demos: list) -> str:
    """Format ``GeneratorSpec.few_shot_demos`` for splicing into the
    prompt between the system instruction and the retrieved contexts.

    Returns the empty string when ``demos`` is empty so the
    zero-shot prompt template is byte-identical to the pre-PR7
    behaviour.

    Layout::

        Here are example responses for reference:

        ## Example 1
        Question: <q1>
        Reasoning: <r1>            (omitted when no reasoning)
        Answer: <a1>

        ## Example 2
        ...
        <blank line before context block>

    The ``## Example N`` headers make demos easy to spot in
    long prompts and discourage the LLM from copying their question
    text verbatim into the answer.
    """
    if not demos:
        return ""
    parts = ["Here are example responses for reference:\n"]
    for i, d in enumerate(demos, 1):
        parts.append(f"## Example {i}")
        if getattr(d, "context", None):
            parts.append(f"Context: {d.context}")
        parts.append(f"Question: {d.question}")
        if getattr(d, "reasoning", None):
            parts.append(f"Reasoning: {d.reasoning}")
        parts.append(f"Answer: {d.answer}")
        parts.append("")  # blank line between demos
    return "\n".join(parts) + "\n"


@dataclass
class RAGAnswer:
    """Output of one full :meth:`RAGPipeline.answer` round-trip."""

    question: str
    contexts: list[str]
    answer: str


class RAGPipeline:
    """A configured RAG runtime: vstore + retrieve + generate.

    Build via :meth:`RAGPipeline.build`. Instances are immutable in
    practice -- to swap a stage, build a new pipeline with
    ``config.with_override(...)``.
    """

    def __init__(
        self,
        config: RAGConfig,
        vstore: Any,
        llm_callable: Callable[[str], str],
        n_chunks: int,
    ) -> None:
        self.config = config
        self.vstore = vstore
        self.llm_callable = llm_callable
        self.n_chunks = n_chunks

    # ------------------------------------------------------------------
    # Construction (PR5: dispatches on ``config.runtime``)
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        config: RAGConfig,
        chunks: list[str],
        *,
        embedder: Any,
        llm_callable: Callable[[str], str],
    ) -> RAGPipeline:
        """Construct a pipeline from a config + pre-chunked content.

        Dispatches on :attr:`RAGConfig.runtime`:

        * ``"langchain"`` -- this class's :meth:`_build_langchain`
          (FAISS / Chroma + langchain similarity search).
        * ``"llamaindex"`` -- :class:`LlamaIndexRAGPipeline`
          (VectorStoreIndex + ``as_retriever``).

        Both backends share the same prompt template and
        ``generate`` logic; only the index + retrieval differ.

        Parameters
        ----------
        config:
            Pipeline configuration. ``config.runtime`` decides which
            backend implements the pipeline.
        chunks:
            Already-split text. Chunking strategy is recorded in
            ``config.chunker`` for reproducibility but the actual
            chunking has happened externally.
        embedder:
            A LangChain-compatible embeddings object (e.g.
            ``HuggingFaceEmbeddings``). The LlamaIndex backend wraps
            this transparently via :class:`_LangchainEmbeddingAdapter`
            so the same embedder serves both runtimes.
        llm_callable:
            A ``(prompt: str) -> str`` callable wrapping whatever LLM
            ``config.generator`` describes.
        """
        if not chunks:
            raise ValueError("RAGPipeline.build requires at least one chunk")

        runtime = config.runtime
        if runtime == "llamaindex":
            return LlamaIndexRAGPipeline._build_impl(
                config, chunks, embedder=embedder, llm_callable=llm_callable,
            )
        if runtime != "langchain":  # pragma: no cover - Literal guards this
            raise ValueError(
                f"Unsupported runtime: {runtime!r}. "
                "Supported: 'langchain', 'llamaindex'."
            )
        # Default langchain backend.
        vstore = cls._build_vstore(config, chunks, embedder)
        return cls(
            config=config,
            vstore=vstore,
            llm_callable=llm_callable,
            n_chunks=len(chunks),
        )

    @staticmethod
    def _build_vstore(config: RAGConfig, chunks: list[str], embedder: Any) -> Any:
        kind = config.retriever.vectorstore
        if kind == "faiss":
            from langchain_community.vectorstores import FAISS

            return FAISS.from_texts(chunks, embedder)
        if kind == "chroma":  # pragma: no cover - exercised in PR2 onwards
            from langchain_community.vectorstores import Chroma

            return Chroma.from_texts(chunks, embedder)
        raise ValueError(
            f"Unsupported vectorstore kind: {kind!r}. "
            "Supported: 'faiss', 'chroma'."
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> list[str]:
        """Return up to ``top_k`` contexts for ``question``.

        ``top_k=None`` uses :attr:`RetrieverSpec.top_k`. This is the
        knob a retrieval-stage optimizer varies per trial.
        """
        k = top_k if top_k is not None else self.config.retriever.top_k
        search_type = self.config.retriever.search_type
        if search_type == "mmr":
            docs = self.vstore.max_marginal_relevance_search(question, k=k)
        else:
            docs = self.vstore.similarity_search(question, k=k)
        return [d.page_content for d in docs]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        contexts: list[str],
        *,
        system_instruction: str | None = None,
    ) -> str:
        """Assemble a RAG prompt and call the generator LLM.

        Resolution order for the instruction header:

        1. ``system_instruction`` argument (per-call override -- used
           by DSPy A/B variants).
        2. :attr:`GeneratorSpec.system_instruction` (config override).
        3. :data:`DEFAULT_SYSTEM_INSTRUCTION`.
        """
        instr = (
            system_instruction
            or self.config.generator.system_instruction
            or DEFAULT_SYSTEM_INSTRUCTION
        )
        ctx_str = "\n---\n".join(contexts) if contexts else "(no context)"
        demos_text = _render_few_shot_demos(self.config.generator.few_shot_demos)
        prompt = (
            f"{instr}\n\n"
            f"{demos_text}"
            f"Context:\n{ctx_str}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            return self.llm_callable(prompt)
        except Exception as e:  # pragma: no cover - live LLM
            return f"<generation error: {e}>"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        system_instruction: str | None = None,
    ) -> RAGAnswer:
        """One-shot retrieve + generate. The typical ``evaluate`` path."""
        contexts = self.retrieve(question, top_k=top_k)
        ans = self.generate(question, contexts, system_instruction=system_instruction)
        return RAGAnswer(question=question, contexts=contexts, answer=ans)


# =====================================================================
# Explicit alias for the default (LangChain) backend
# =====================================================================
# Provides a name PR4+ code can use to be explicit about which backend
# it's constructing. ``RAGPipeline`` itself remains the LangChain
# implementation for backward compat (existing tests + stage
# optimizers construct it directly).
LangChainRAGPipeline = RAGPipeline


# =====================================================================
# LlamaIndex backend
# =====================================================================
class LlamaIndexRAGPipeline:
    """RAG runtime backed by LlamaIndex's ``VectorStoreIndex``.

    Selected when ``RAGConfig.runtime == "llamaindex"``. Reuses the
    same embedder + LLM callable the LangChain backend gets -- only
    the index + retrieval differ. The prompt template and
    ``generate`` / ``answer`` semantics match
    :class:`RAGPipeline.generate` byte-for-byte so the experiment
    workflow's BO / DSPy stages behave identically across backends.

    Requires ``llama-index-core`` (declared in the ``ragdx[llamaindex]``
    extra). Heavy import lives inside :meth:`_build_impl` so consumers
    of :mod:`ragdx.runtime.pipeline` who never select the LlamaIndex
    backend don't pay the cost.
    """

    def __init__(
        self,
        config: RAGConfig,
        index: Any,
        llm_callable: Callable[[str], str],
        n_chunks: int,
    ) -> None:
        self.config = config
        self.index = index
        self.llm_callable = llm_callable
        self.n_chunks = n_chunks

    @classmethod
    def _build_impl(
        cls,
        config: RAGConfig,
        chunks: list[str],
        *,
        embedder: Any,
        llm_callable: Callable[[str], str],
    ) -> LlamaIndexRAGPipeline:
        try:
            from llama_index.core import Document, VectorStoreIndex
        except ImportError as exc:  # pragma: no cover - extra check
            raise ImportError(
                "LlamaIndex backend requires the ``ragdx[llamaindex]`` "
                "extra. Install with: pip install 'llama-index-core>=0.12,<1'."
            ) from exc

        embed_model = _build_llamaindex_embedder(embedder)
        docs = [Document(text=c) for c in chunks]
        # Don't pass through Settings (global state); thread the embed
        # model explicitly so multiple pipelines can coexist.
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        return cls(
            config=config,
            index=index,
            llm_callable=llm_callable,
            n_chunks=len(chunks),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
    ) -> list[str]:
        """``index.as_retriever(similarity_top_k=k).retrieve(question)``."""
        k = top_k if top_k is not None else self.config.retriever.top_k
        retriever = self.index.as_retriever(similarity_top_k=k)
        nodes = retriever.retrieve(question)
        return [n.node.get_content() for n in nodes]

    def generate(
        self,
        question: str,
        contexts: list[str],
        *,
        system_instruction: str | None = None,
    ) -> str:
        """Identical to :meth:`RAGPipeline.generate` -- shared prompt
        template + same ``llm_callable``. The two backends produce
        comparable answers given the same retrieved contexts."""
        instr = (
            system_instruction
            or self.config.generator.system_instruction
            or DEFAULT_SYSTEM_INSTRUCTION
        )
        ctx_str = "\n---\n".join(contexts) if contexts else "(no context)"
        demos_text = _render_few_shot_demos(self.config.generator.few_shot_demos)
        prompt = (
            f"{instr}\n\n"
            f"{demos_text}"
            f"Context:\n{ctx_str}\n\nQuestion: {question}\n\nAnswer:"
        )
        try:
            return self.llm_callable(prompt)
        except Exception as e:  # pragma: no cover - live LLM
            return f"<generation error: {e}>"

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        system_instruction: str | None = None,
    ) -> RAGAnswer:
        contexts = self.retrieve(question, top_k=top_k)
        ans = self.generate(question, contexts, system_instruction=system_instruction)
        return RAGAnswer(question=question, contexts=contexts, answer=ans)


def _build_llamaindex_embedder(lc_embedder: Any) -> Any:
    """Wrap a LangChain embeddings object as a LlamaIndex
    :class:`BaseEmbedding`.

    LlamaIndex enforces ``isinstance(embed_model, BaseEmbedding)`` --
    duck typing is rejected at index-build time. The adapter class is
    defined inside this function (rather than at module level) so
    :mod:`ragdx.runtime.pipeline` stays importable without
    ``llama-index-core`` installed: callers who never set
    ``RAGConfig.runtime = "llamaindex"`` pay zero cost.

    Field mapping:

    * LangChain ``embed_query(text)`` -> LlamaIndex ``_get_query_embedding``.
    * LangChain ``embed_documents([text])[0]`` -> LlamaIndex ``_get_text_embedding``.
    * Async variants delegate to sync (LangChain's HF embedder is sync
      under the hood anyway).
    """
    from llama_index.core.embeddings import BaseEmbedding
    from pydantic import PrivateAttr

    class _Adapter(BaseEmbedding):
        # Private attr keeps the langchain object out of pydantic's
        # validated field set (it doesn't have a usable schema).
        _lc: Any = PrivateAttr()

        def __init__(self, lc: Any, **kw: Any) -> None:
            super().__init__(**kw)
            self._lc = lc

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._lc.embed_query(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self._lc.embed_documents([text])[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            # One batched langchain call instead of N -- preserves
            # whatever batching ``HuggingFaceEmbeddings`` does internally.
            return self._lc.embed_documents(texts)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

        async def aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

    return _Adapter(lc_embedder)


__all__ = [
    "DEFAULT_SYSTEM_INSTRUCTION",
    "LangChainRAGPipeline",
    "LlamaIndexRAGPipeline",
    "RAGAnswer",
    "RAGPipeline",
]
