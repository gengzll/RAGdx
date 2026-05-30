""":class:`ChunkingOptimizer` -- vary only the chunker.

Holds the retriever and generator fixed at the base config's values
and sweeps ``chunk_size`` x ``chunk_overlap``. Useful when a user
suspects their production chunking is wrong but their retriever +
prompt are already tuned.

PDF corpora benefit most: smaller chunks improve granularity but
hurt recall at fixed top_k. The base config's ``retriever.top_k`` is
respected (no per-call override), so the optimizer is honest about
what it's tuning.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ragdx.optim.stages.base import StageContext, StageOptimizer
from ragdx.schemas.rag_config import ChunkerSpec, RAGConfig


class ChunkingOptimizer(StageOptimizer):
    """Optimize only the chunker stage."""

    name: ClassVar[str] = "chunking"

    def search_space(self, ctx: StageContext) -> dict[str, list]:
        return {
            "chunk_size": list(ctx.chunk_sizes),
            "chunk_overlap": list(ctx.chunk_overlaps),
        }

    def build_trial_config(self, base: RAGConfig, params: dict[str, Any]) -> RAGConfig:
        return base.with_override(
            chunker=ChunkerSpec(
                strategy=base.chunker.strategy,
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
            ),
        )

    def cache_key(self, params: dict[str, Any]) -> tuple:
        return (params["chunk_size"], params["chunk_overlap"])

    def needs_re_chunk(self, params: dict[str, Any]) -> bool:
        return True


__all__ = ["ChunkingOptimizer"]
