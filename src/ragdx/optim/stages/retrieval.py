""":class:`RetrievalOptimizer` -- vary only the retriever.

Holds the chunker fixed (so the vstore is built **once** and reused
across every trial -- much cheaper than Joint or Chunking) and sweeps
``top_k``. Search-type / reranker can be added once those become
varied in practice.

This is the cheapest stage to optimize in isolation: no re-chunking,
no re-indexing, just different similarity_search calls against the
same vstore. Good first move after a chunking sweep settles.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ragdx.optim.stages.base import StageContext, StageOptimizer
from ragdx.schemas.rag_config import RAGConfig, RetrieverSpec


class RetrievalOptimizer(StageOptimizer):
    """Optimize only the retriever stage."""

    name: ClassVar[str] = "retrieval"

    def search_space(self, ctx: StageContext) -> dict[str, list]:
        return {"top_k": list(ctx.top_ks)}

    def build_trial_config(self, base: RAGConfig, params: dict[str, Any]) -> RAGConfig:
        return base.with_override(
            retriever=RetrieverSpec(
                vectorstore=base.retriever.vectorstore,
                search_type=base.retriever.search_type,
                top_k=params["top_k"],
                reranker=base.retriever.reranker,
            ),
        )

    # No cache invalidation on top_k change: cache_key() default (empty
    # tuple) means a single pipeline serves every trial, and we forward
    # top_k as a per-call override to ``pipeline.retrieve(top_k=...)``.

    def per_call_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"top_k": params["top_k"]}


__all__ = ["RetrievalOptimizer"]
