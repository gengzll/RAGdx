""":class:`JointOptimizer` -- vary chunker + retriever together.

Preserves the behaviour of ``ragdx.experiments._run_bayes_search``
exactly: same three-axis search space (``chunk_size`` x
``chunk_overlap`` x ``top_k``), same vstore cache keyed by
``(chunk_size, chunk_overlap)``, same trial schema. The experiment
workflow uses this for the ``bayes_search`` bundle section so post-
PR3 numbers match pre-PR3 byte-for-byte under the same seed.

This is also the right default for users who don't know which stage
to target -- chunking, retrieval, and pipeline-config-search
interact, so jointly tuning them is usually the safest opening move.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ragdx.optim.stages.base import StageContext, StageOptimizer
from ragdx.schemas.rag_config import ChunkerSpec, RAGConfig, RetrieverSpec


class JointOptimizer(StageOptimizer):
    """Bayesian search over (chunker, retriever) jointly.

    Re-chunking the corpus per (``chunk_size``, ``chunk_overlap``)
    key is expensive on PDF sources, so we cache pipelines by that
    key; ``top_k`` changes don't invalidate the vstore.
    """

    name: ClassVar[str] = "joint"

    def search_space(self, ctx: StageContext) -> dict[str, list]:
        return {
            "chunk_size": list(ctx.chunk_sizes),
            "chunk_overlap": list(ctx.chunk_overlaps),
            "top_k": list(ctx.top_ks),
        }

    def build_trial_config(self, base: RAGConfig, params: dict[str, Any]) -> RAGConfig:
        return base.with_override(
            chunker=ChunkerSpec(
                strategy=base.chunker.strategy,
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
            ),
            retriever=RetrieverSpec(
                vectorstore=base.retriever.vectorstore,
                search_type=base.retriever.search_type,
                top_k=params["top_k"],
                reranker=base.retriever.reranker,
            ),
        )

    def cache_key(self, params: dict[str, Any]) -> tuple:
        # Chunker change invalidates the vstore; top_k change does not.
        return (params["chunk_size"], params["chunk_overlap"])

    def needs_re_chunk(self, params: dict[str, Any]) -> bool:
        return True

    def per_call_overrides(self, params: dict[str, Any]) -> dict[str, Any]:
        # top_k goes through pipeline.retrieve(top_k=...) per call so a
        # single (chunk_size, overlap) pipeline serves multiple top_k
        # values without rebuilding the vstore. Matches pre-PR3 BO behaviour.
        return {"top_k": params["top_k"]}


__all__ = ["JointOptimizer"]
