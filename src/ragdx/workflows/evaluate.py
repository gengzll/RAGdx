"""``ragdx.workflows.evaluate`` -- run a RAGConfig against an eval suite.

This is the missing bridge between a user's production RAG and the
``diagnose / plan / optimize`` half of ragdx. Pre-PR4 the only way to
produce a normalized :class:`EvaluationResult` was to either:

1. Run your production stack externally, run ragas/RAGChecker against
   it externally, and feed the JSON through ``ragdx normalize-tools``.
2. Run ``ragdx experiment`` which does everything end-to-end but
   couples evaluation to the BO + DSPy stages.

Now: describe your RAG in a ``rag_config.yaml``, point ragdx at it,
get an ``EvaluationResult`` JSON directly. The output is shape-
compatible with everything downstream (``diagnose``, ``plan``,
``compare``, ``save``, the RunStore).

::

    from ragdx.schemas.rag_config import RAGConfig
    from ragdx.workflows import evaluate
    cfg = RAGConfig.from_yaml("rag_config.yaml")
    result = evaluate(cfg, chunks=my_chunks, records=my_records)
    result.model_dump_json(indent=2)

The workflow stays pure orchestration -- corpus loading happens
upstream (``ragdx evaluate`` CLI does it via the existing helpers in
``ragdx.experiments``; library callers can pass arbitrary chunks).
"""

from __future__ import annotations

import logging
from typing import Any

from ragdx.runtime.factories import RagdxRuntime, build_runtime
from ragdx.runtime.pipeline import RAGPipeline
from ragdx.schemas.models import DatasetRecord, EvaluationResult
from ragdx.schemas.rag_config import RAGConfig

logger = logging.getLogger(__name__)


def _select_metrics(records: list[DatasetRecord]) -> list:
    """Return ragas metric instances appropriate for the records' GT mode.

    Matches ``ragdx.experiments._build_ragas_metrics_for_mode`` --
    reference-based metrics (``context_recall``, ``answer_correctness``)
    only when GT is populated.
    """
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    has_gt = any((r.ground_truth or "").strip() for r in records)
    if has_gt:
        return [faithfulness, answer_relevancy, context_precision, context_recall]
    return [faithfulness, answer_relevancy, context_precision]


def _scores_to_evaluation_result(
    scores: dict[str, float],
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Partition a flat scores dict into the EvaluationResult's
    retrieval / generation / e2e sections.

    Routing uses the canonical ``LAYER_OF`` map from
    :mod:`ragdx.core.metrics` (the same map the three-layer overview
    and the layer aggregates use), so every computed metric — including
    the deepeval supplements (``bias`` / ``toxicity`` / ``g_eval``) —
    lands in its proper bucket instead of falling into
    ``metadata["unmapped_scores"]``. Genuinely unknown names still go
    to ``unmapped_scores`` for debuggability.
    """
    from ragdx.core.metrics import LAYER_OF

    buckets: dict[str, dict[str, float]] = {
        "retrieval": {}, "generation": {}, "e2e": {},
    }
    other: dict[str, float] = {}
    for k, v in scores.items():
        if not isinstance(v, (int, float)):
            continue
        layer = LAYER_OF.get(k)
        if layer in buckets:
            buckets[layer][k] = float(v)
        else:
            other[k] = float(v)
    md = dict(metadata or {})
    if other:
        md.setdefault("unmapped_scores", other)
    return EvaluationResult(
        retrieval=buckets["retrieval"],
        generation=buckets["generation"],
        e2e=buckets["e2e"],
        metadata=md,
    )


def evaluate(
    config: RAGConfig,
    chunks: list[str],
    records: list[DatasetRecord],
    *,
    runtime: RagdxRuntime | None = None,
    metadata: dict[str, Any] | None = None,
    evaluator: str = "ragas",
) -> EvaluationResult:
    """Run ``config`` over ``records`` and return a normalized
    :class:`EvaluationResult`.

    Parameters
    ----------
    config:
        Production RAG description. Determines vstore kind, retrieval
        policy, generator LLM, judge LLM, etc.
    chunks:
        Already-split content. The workflow doesn't load corpora --
        callers (notably ``ragdx evaluate`` CLI) do that via the
        existing helpers.
    records:
        The eval suite (``{question, ground_truth?, contexts?}``).
        GT mode is inferred from these.
    runtime:
        Optional pre-built runtime. If ``None``, calls
        :func:`build_runtime(config)` -- pay the LLM client construction
        cost once per ``evaluate()`` call.
    metadata:
        Extra fields stuffed into ``result.metadata`` (e.g.
        ``{"config_path": "rag_config.yaml"}``). Useful for downstream
        tools that want to know which config produced the eval.

    Returns
    -------
    EvaluationResult with retrieval / generation / e2e sections
    populated from ragas scores. Any metric ragas reports that doesn't
    map to a known section ends up under ``metadata['unmapped_scores']``.
    """
    if runtime is None:
        runtime = build_runtime(config)

    # 1. Build the pipeline + answer every record.
    pipeline = RAGPipeline.build(
        config,
        chunks,
        embedder=runtime.embeddings,
        llm_callable=runtime.llm_callable,
    )
    logger.info("evaluate: %d chunks, %d records, model=%s",
                pipeline.n_chunks, len(records), config.generator.model)

    answered: list[DatasetRecord] = []
    for r in records:
        out = pipeline.answer(r.question)
        answered.append(
            DatasetRecord(
                question=r.question,
                ground_truth=r.ground_truth,
                contexts=out.contexts,
                answer=out.answer,
            )
        )

    # 2. Score with the chosen evaluator. ragas vs deepeval differ in
    # judge-LM contract and metric naming; we hide both behind the
    # same ``scores`` dict + ``EvaluationResult`` so downstream code
    # (objectives, diagnosis, dashboards) is evaluator-agnostic.
    evaluator_kind = (evaluator or "ragas").lower()
    if evaluator_kind not in {"ragas", "deepeval"}:
        raise ValueError(
            f"Unknown evaluator {evaluator!r}. "
            f"Choose one of: 'ragas', 'deepeval'."
        )

    if evaluator_kind == "deepeval":
        from ragdx.engines.deepeval_adapter import DeepEvalAdapter

        if runtime.deepeval_judge is None:
            raise RuntimeError(
                "evaluator='deepeval' requested but deepeval (or "
                "langchain-openai) isn't installed. Install with "
                "`pip install ragdx[deepeval]`."
            )
        try:
            er = DeepEvalAdapter().evaluate(
                answered,
                run_deepeval=True,
                model=runtime.deepeval_judge,
            )
            scores = {**er.retrieval, **er.generation, **er.e2e}
            ev_meta_extra = {
                "evaluator": "deepeval",
                "deepeval_metrics": er.metadata.get("deepeval_metrics", []),
            }
        except Exception as exc:  # pragma: no cover - depends on judge
            logger.warning("evaluate: deepeval reported error -- %s", exc)
            scores = {}
            ev_meta_extra = {"evaluator": "deepeval", "evaluator_error": str(exc)}
    else:
        # Re-use the experiment workflow's helper so the ragas throttle /
        # error-handling behaviour is identical to the experiment path.
        from ragdx.experiments import _evaluate_with_ragas

        metrics = _select_metrics(answered)
        ev = _evaluate_with_ragas(
            answered,
            runtime.ragas_judge,
            runtime.ragas_embeddings,
            metrics,
            run_config=runtime.ragas_run_config,
        )
        scores = ev.get("scores", {})
        if "error" in ev:
            logger.warning("evaluate: ragas reported error -- %s", ev["error"])
        ev_meta_extra = {"evaluator": "ragas"}
        if ev.get("skipped"):
            ev_meta_extra["ragas_skipped_metrics"] = ev["skipped"]

        # Supplement with deepeval-only metrics (hallucination / bias /
        # toxicity / g_eval) when a deepeval judge is available, so the
        # three-layer view shows real numbers instead of "requires
        # deepeval". Best-effort; no-op when deepeval isn't installed.
        if getattr(runtime, "deepeval_judge", None) is not None:
            try:
                from ragdx.experiments import _supplement_deepeval_metrics
                supp = _supplement_deepeval_metrics(
                    [
                        {
                            "question": r.question,
                            "contexts": list(r.contexts or []),
                            "optimized_answer": r.answer or "",
                        }
                        for r in answered
                    ],
                    runtime,
                )
                if supp:
                    scores = {**scores, **supp}
                    ev_meta_extra["deepeval_supplement"] = supp
            except Exception as exc:  # pragma: no cover - judge-dependent
                logger.warning("evaluate: deepeval supplement skipped -- %s", exc)

    # 3. Normalize into EvaluationResult sections.
    md: dict[str, Any] = {
        "n_records": len(records),
        "n_chunks": pipeline.n_chunks,
        "model": config.generator.model,
        "config_name": config.name,
        **ev_meta_extra,
    }
    if metadata:
        md.update(metadata)
    # ``ragas_skipped_metrics`` (when present) is already in ev_meta_extra
    # for the ragas path; the deepeval path doesn't have an equivalent
    # skip-list concept.
    return _scores_to_evaluation_result(scores, metadata=md)


__all__ = ["evaluate"]
