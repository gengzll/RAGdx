"""Ragas-backed metric for DSPy's MIPROv2 inner loop.

DSPy's built-in no-GT metric is a single ``FaithfulnessJudge`` LLM call
(see :meth:`DSPyAdapter.build_metric_function`). It's cheap, but for
permissive judge LMs like GLM-4-Flash it tends to saturate at 1.0
("fully grounded") for every candidate -- which strips MIPROv2 of the
discriminative signal it needs to pick a winner. The result: all trials
tie, and the BO degenerates to "return the seed program".

This module builds a DSPy-compatible metric that calls ragas's actual
single-turn-scoring API for every ``(example, prediction)`` pair, then
aggregates the per-metric scores into a single scalar via the same
weights ``default_objective("no_gt")`` uses. Ragas's three no-GT
metrics together (``context_precision`` + ``faithfulness`` +
``answer_relevancy``) carry enough variance to push DSPy past the
tie-on-seed problem.

Cost: each metric call now makes ~3 LLM calls (one per ragas metric)
instead of 1. With MIPROv2 ``auto="medium"``, total LLM calls roughly
triple. Trade-off: real discriminative search.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


# Default per-metric weights mirror ``default_objective("no_gt")`` in
# :mod:`ragdx.optim.objectives` so the DSPy inner loop and the ragdx
# ``ragas`` outer-loop objective optimise on the same scale.
_DEFAULT_NO_GT_WEIGHTS = {
    "faithfulness": 1.5,
    "answer_relevancy": 1.0,
    "context_precision": 0.3,
}


def make_ragas_metric(
    judge: Any,
    embeddings: Any,
    *,
    metric_names: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> Callable[..., float]:
    """Build a DSPy metric callable that scores via ragas.

    Parameters
    ----------
    judge:
        A ``ragas.llms.LangchainLLMWrapper``-compatible LM, used as the
        judge for every ragas metric call. Typically
        ``runtime.ragas_judge`` from the ragdx runtime.
    embeddings:
        A ragas-compatible embeddings object (typically
        ``runtime.ragas_embeddings``). Required by
        ``answer_relevancy``; ignored by the other two no-GT metrics.
    metric_names:
        Subset of ``{"context_precision", "faithfulness",
        "answer_relevancy"}`` to evaluate. Defaults to all three.
    weights:
        Per-metric weights for the weighted-sum aggregator. Defaults to
        ``_DEFAULT_NO_GT_WEIGHTS`` (matches the ragdx ``no_gt`` objective).
        Unknown metric names fall back to weight ``1.0``.

    Returns
    -------
    A callable ``metric(example, pred, trace=None) -> float`` suitable
    for DSPy teleprompters (MIPROv2 / COPRO / BootstrapFewShot).

    The returned metric:

    * Extracts ``question``, ``contexts``, and ``answer`` from the DSPy
      ``Example`` / ``Prediction``.
    * Builds a ``ragas.SingleTurnSample``.
    * For each requested metric, sets ``metric.llm`` / ``metric.embeddings``
      to the supplied judge / embeddings and calls
      ``single_turn_ascore(sample)``.
    * Returns the weighted sum. Exceptions per-metric are logged and
      treated as score ``0.0`` (so a flaky judge call doesn't crash
      the optimization).

    Failure modes -- returns ``0.0``:

    * Empty answer or empty contexts.
    * All metric calls raise.
    * ``ragas`` import fails (the caller should fall back to the
      DSPy built-in metric in that case).
    """
    # Lazy import so callers without ragas installed can still import
    # this module (e.g. ``cli/tune.py`` does ``--help`` discovery).
    try:
        from ragas import SingleTurnSample
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "Building the ragas DSPy metric requires the `ragas` package. "
            "Install with `pip install ragdx[ragas]`."
        ) from exc

    if metric_names is None:
        metric_names = ["context_precision", "faithfulness", "answer_relevancy"]
    if weights is None:
        weights = dict(_DEFAULT_NO_GT_WEIGHTS)

    _metric_objs: dict[str, Any] = {
        "context_precision": context_precision,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    for name in metric_names:
        if name not in _metric_objs:
            raise ValueError(
                f"Unknown ragas metric name {name!r}; pick from "
                f"{sorted(_metric_objs)}."
            )

    # Inject judge / embeddings up-front so MIPROv2's threadpool calls
    # all reuse the same configured metric objects. Ragas metrics are
    # documented as thread-safe once .llm / .embeddings are set.
    for name in metric_names:
        m = _metric_objs[name]
        m.llm = judge
        if embeddings is not None and hasattr(m, "embeddings"):
            m.embeddings = embeddings

    # Per-call timeout. A hung HTTP connection on the judge LM (e.g.
    # the GLM endpoint after a network switch) would otherwise sit
    # indefinitely waiting on a dead socket -- discovered on
    # 2026-06-02 when a `--mipro-auto medium` run stalled on Trial 14
    # for 15+ minutes. 90s is generous for a single ragas single-turn
    # call; if it expires we record 0.0 and let MIPROv2 keep going.
    _PER_CALL_TIMEOUT_S = 90.0

    def _safe_async_score(metric: Any, sample: Any) -> float:
        """Call ``metric.single_turn_ascore(sample)`` from sync code,
        with a hard timeout so a hung HTTP connection can't stall
        the entire MIPROv2 run.

        ``asyncio.run`` requires no running event loop. MIPROv2's inner
        loop is sync (threaded), so this is safe. If a loop is already
        running (e.g. someone wrapping this in async code), fall back to
        ``loop.run_until_complete``.
        """
        async def _scored_with_timeout() -> float:
            return float(
                await asyncio.wait_for(
                    metric.single_turn_ascore(sample),
                    timeout=_PER_CALL_TIMEOUT_S,
                )
            )

        try:
            return asyncio.run(_scored_with_timeout())
        except RuntimeError:
            # Already inside an event loop (rare).
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_scored_with_timeout())

    def metric(example: Any, pred: Any, trace: Any | None = None) -> float:
        question = str(getattr(example, "question", "") or "")
        ctx = getattr(example, "context", None) or getattr(example, "contexts", "")
        if isinstance(ctx, list):
            contexts = [str(c) for c in ctx if c]
        else:
            contexts = [str(ctx)] if ctx else []
        answer = str(getattr(pred, "answer", "") or "")

        if not answer or not contexts:
            return 0.0

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        total = 0.0
        any_succeeded = False
        for name in metric_names:
            m = _metric_objs[name]
            w = weights.get(name, 1.0)
            try:
                s = _safe_async_score(m, sample)
                if isinstance(s, (int, float)):
                    total += float(s) * float(w)
                    any_succeeded = True
            except asyncio.TimeoutError:
                logger.warning(
                    "ragas %s scoring hit %.0fs timeout (treated as 0.0). "
                    "Network hiccup or judge LM hang -- MIPROv2 will keep "
                    "exploring.", name, _PER_CALL_TIMEOUT_S,
                )
            except Exception as exc:  # pragma: no cover - depends on judge LM
                logger.debug(
                    "ragas %s scoring failed (treated as 0.0): %s", name, exc,
                )

        # If *every* metric raised, return 0 rather than the
        # default-initialised 0.0 (semantically equivalent here, but
        # makes the failure visible to MIPROv2's tie-breaking).
        return total if any_succeeded else 0.0

    metric.__ragdx_kind__ = "ragas_composite"  # type: ignore[attr-defined]
    metric.__ragdx_metric_names__ = list(metric_names)  # type: ignore[attr-defined]
    metric.__ragdx_weights__ = dict(weights)  # type: ignore[attr-defined]
    return metric


__all__ = ["make_ragas_metric"]
