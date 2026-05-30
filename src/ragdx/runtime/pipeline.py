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
    # Construction
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

        Parameters
        ----------
        config:
            Stage specs determining the vstore kind and retrieval /
            generation behaviour.
        chunks:
            Already-split text. Chunking strategy is recorded in
            ``config.chunker`` for reproducibility but the actual
            chunking has happened externally.
        embedder:
            A LangChain-compatible embeddings object (e.g.
            ``HuggingFaceEmbeddings``). PR2 will derive this from
            ``config.embedder`` automatically.
        llm_callable:
            A ``(prompt: str) -> str`` callable wrapping whatever LLM
            ``config.generator`` describes. PR2 will derive this from
            ``config.generator`` automatically.
        """
        if not chunks:
            raise ValueError("RAGPipeline.build requires at least one chunk")

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
        prompt = (
            f"{instr}\n\n"
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


__all__ = ["DEFAULT_SYSTEM_INSTRUCTION", "RAGAnswer", "RAGPipeline"]
