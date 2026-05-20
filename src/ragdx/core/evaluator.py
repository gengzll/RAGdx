"""Unified RAG evaluation entry point.

:class:`UnifiedEvaluator` composes several evaluation adapters into a
single :class:`EvaluationResult`. Three modes are supported:

* External score normalization — pass ``ragas_scores`` / ``ragchecker_scores``
  to feed pre-computed numbers through the corresponding normalization map.
* In-process invocation — ``use_ragas=True, run_ragas=True`` will actually
  call ``ragas.evaluate`` if the optional dependency is installed.
* Local proxy — ``use_embedding=True`` runs the dependency-free
  :class:`ragdx.engines.embedding_eval.EmbeddingEvaluator` to produce
  TF-IDF cosine-similarity proxies for the standard metrics. This is the
  recommended path for CI smoke tests and offline triage.

All branches return a unified :class:`EvaluationResult`; multiple branches
are merged metric-by-metric (later branches overwrite earlier values when
they emit the same key).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ragdx.engines.embedding_eval import EmbeddingEvaluator
from ragdx.engines.ragas_adapter import RagasAdapter
from ragdx.engines.ragchecker_adapter import RAGCheckerAdapter
from ragdx.schemas.models import DatasetRecord, EvaluationResult


class UnifiedEvaluator:
    def __init__(
        self,
        ragas_adapter: RagasAdapter | None = None,
        ragchecker_adapter: RAGCheckerAdapter | None = None,
        embedding_evaluator: EmbeddingEvaluator | None = None,
    ):
        self.ragas_adapter = ragas_adapter or RagasAdapter()
        self.ragchecker_adapter = ragchecker_adapter or RAGCheckerAdapter()
        self.embedding_evaluator = embedding_evaluator or EmbeddingEvaluator()

    def merge(self, *results: EvaluationResult) -> EvaluationResult:
        merged = EvaluationResult()
        for result in results:
            merged.retrieval.update(result.retrieval)
            merged.generation.update(result.generation)
            merged.e2e.update(result.e2e)
            merged.metadata.update(result.metadata)
            merged.raw_tool_outputs.update(result.raw_tool_outputs)
        return merged

    def evaluate(
        self,
        records: Iterable[DatasetRecord],
        ragas_scores: Mapping[str, float] | None = None,
        ragchecker_scores: Mapping[str, float] | None = None,
        use_ragas: bool = True,
        use_ragchecker: bool = True,
        use_embedding: bool = False,
        *,
        run_ragas: bool = False,
        ragas_kwargs: Mapping[str, Any] | None = None,
        run_ragchecker: bool = False,
        ragchecker_kwargs: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Run the configured evaluators and return a merged result.

        Parameters
        ----------
        records:
            Iterable of :class:`DatasetRecord`.
        ragas_scores / ragchecker_scores:
            If provided, the corresponding adapter normalizes these external
            scores rather than running its evaluation logic.
        use_ragas / use_ragchecker:
            Toggle each adapter on or off.
        use_embedding:
            If True, also run the local TF-IDF embedding proxy evaluator
            (no external dependencies, suitable for CI).
        run_ragas:
            Forwarded to :meth:`RagasAdapter.evaluate` — if True (and
            ``ragas_scores`` is None) the adapter calls ``ragas.evaluate``
            in-process.
        ragas_kwargs:
            Extra kwargs forwarded to ``ragas.evaluate`` (e.g. ``llm=``,
            ``embeddings=``, ``metrics=``).
        run_ragchecker:
            Forwarded to :meth:`RAGCheckerAdapter.evaluate` — if True (and
            ``ragchecker_scores`` is None) the adapter calls
            ``RAGChecker.evaluate`` in-process.
        ragchecker_kwargs:
            Extra kwargs forwarded to the RAGChecker adapter. Use
            ``evaluator=<RAGChecker>`` to inject a configured checker, or
            ``metric_names=[...]`` to pick a subset of metrics. Other kwargs
            are forwarded to ``RAGChecker(**kwargs)`` if no evaluator is given.
        """

        outputs = []
        records = list(records)
        if use_ragas:
            outputs.append(
                self.ragas_adapter.evaluate(
                    records,
                    raw_scores=ragas_scores,
                    run_ragas=run_ragas,
                    **(dict(ragas_kwargs) if ragas_kwargs else {}),
                )
            )
        if use_ragchecker:
            outputs.append(
                self.ragchecker_adapter.evaluate(
                    records,
                    raw_scores=ragchecker_scores,
                    run_ragchecker=run_ragchecker,
                    **(dict(ragchecker_kwargs) if ragchecker_kwargs else {}),
                )
            )
        if use_embedding:
            outputs.append(self.embedding_evaluator.evaluate(records))
        return self.merge(*outputs)


__all__ = ["UnifiedEvaluator"]
