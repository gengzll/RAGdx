"""Runtime layer: the single source of truth for *how* ragdx runs a RAG.

Everything that actually executes a query lives here. Higher-level
orchestrators (``ragdx.workflows.experiment``, the stage-targeted
optimizers in ``ragdx.optim.stages``) compose these primitives but
never reimplement them.
"""

from ragdx.runtime.pipeline import RAGAnswer, RAGPipeline

__all__ = ["RAGAnswer", "RAGPipeline"]
