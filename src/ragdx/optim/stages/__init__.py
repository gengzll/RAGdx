"""Stage-targeted RAG optimizers.

Each ``StageOptimizer`` varies one slice of a :class:`RAGConfig`
(chunker / retriever / generator / joint) while holding the others
fixed at the base config's values. This is the primitive ``ragdx
optimize --stage <name>`` (PR4) dispatches to: a user describes their
production RAG as a ``RAGConfig`` and asks ragdx to improve *only the
chunker*, or *only the retriever*, without touching anything else.

The default loop (in :mod:`ragdx.optim.stages.base`) drives ragdx's
own Bayesian search; stages whose search space doesn't fit BO
(generation = prompt optimization via DSPy MIPROv2) override
``optimize`` directly.

The experiment workflow now composes
:class:`JointOptimizer` (the legacy "vary chunk + top_k together"
behaviour, byte-identical to the pre-PR3 BO numbers) followed by
:class:`GenerationOptimizer`, so the existing ``bayes_search`` and
``dspy_a_b`` bundle sections come straight from stage outputs.
"""

from ragdx.optim.stages.base import (
    StageContext,
    StageOptimizer,
    StageResult,
    StageTrial,
)
from ragdx.optim.stages.chunking import ChunkingOptimizer
from ragdx.optim.stages.generation import GenerationOptimizer
from ragdx.optim.stages.joint import JointOptimizer
from ragdx.optim.stages.retrieval import RetrievalOptimizer

__all__ = [
    "ChunkingOptimizer",
    "GenerationOptimizer",
    "JointOptimizer",
    "RetrievalOptimizer",
    "StageContext",
    "StageOptimizer",
    "StageResult",
    "StageTrial",
]
