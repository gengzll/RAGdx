"""High-level workflows that compose the lower-level building blocks.

Each workflow is a thin orchestrator -- it takes pre-loaded inputs
(``RAGConfig`` + pre-chunked corpus + records, etc.) and threads them
through ``runtime/`` and ``optim/`` to produce a canonical output.
The CLI commands in :mod:`ragdx.cli` are thin wrappers around these.

Currently exposed:

* :func:`ragdx.workflows.evaluate.evaluate` -- run a RAGConfig over an
  eval suite, score with ragas, return an :class:`EvaluationResult`.
  The bridge that lets the README's "control plane" workflow
  (normalize → diagnose → plan → optimize) consume a config users
  write by hand.
"""

from ragdx.workflows.evaluate import evaluate

__all__ = ["evaluate"]
