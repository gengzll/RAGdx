"""DeepEval-backed DSPy inner-loop metric.

Parallel to :mod:`ragdx.optim._ragas_dspy_metric` -- this one drives
DSPy optimizers (MIPROv2 / COPRO / BootstrapFewShot / GEPA) using
deepeval's metrics instead of ragas's. The headline reason: deepeval's
**G-Eval** is a chain-of-thought LLM-as-judge that's much less prone
to saturation than ragas's two-step claim-extraction ``faithfulness``
on permissive judges (GLM-4-Flash, Kimi, etc.).

The metric callable shape is identical to the ragas / embed-rubric
ones: ``metric(example, pred, trace=None) -> float`` returning a
weighted-sum scalar. DSPy's teleprompters expect that signature
verbatim.

Cost: 1 LLM call for G-Eval per ``metric()`` invocation. The
``AnswerRelevancy`` + ``ContextualRelevancy`` companions add 1-2 more
each (deepeval embeds the answer into multiple statements internally).
Roughly comparable to ragas's 3-call default; the saturation tradeoff
is what makes it worth the cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


# Default weights. ``g_eval`` dominates because it's the
# non-saturating signal (the very reason for picking deepeval over
# ragas); the other two are sanity rails.
_DEFAULT_DEEPEVAL_WEIGHTS: dict[str, float] = {
    "g_eval": 1.5,
    "answer_relevancy": 0.5,
    "faithfulness": 0.5,
}


def make_deepeval_metric(
    judge_lm: Any,
    *,
    weights: dict[str, float] | None = None,
    geval_criteria: str | None = None,
    threshold: float = 0.5,
) -> Callable[..., float]:
    """Build a DSPy-compatible metric callable that scores via deepeval.

    Parameters
    ----------
    judge_lm:
        A ``DeepEvalBaseLLM`` instance (typically
        ``runtime.deepeval_judge`` from ``build_runtime``). Must
        accept ``generate(prompt: str, schema=None)`` -- the contract
        DeepEval's metric machinery uses internally.
    weights:
        Per-metric weights for the weighted-sum aggregator. Defaults
        to :data:`_DEFAULT_DEEPEVAL_WEIGHTS`. Unknown metric names
        fall back to weight ``1.0``.
    geval_criteria:
        Free-text scoring rubric handed to G-Eval. Defaults to a
        production-realistic "groundedness + completeness" criterion;
        override per-domain (e.g. medical / legal RAG).
    threshold:
        DeepEval metric ``threshold`` -- doesn't change the numeric
        score we return, only deepeval's internal "passed" flag.

    Returns
    -------
    A callable ``metric(example, pred, trace=None) -> float``.

    Failure modes -- return ``0.0``:
    * Empty answer or empty contexts.
    * ``deepeval`` import fails (caller should fall back to ragas /
      embed_rubric).
    * The judge LM is ``None`` (deepeval isn't set up).

    Per-component failures are logged at DEBUG and contribute ``0.0``;
    a single flaky judge call doesn't crash the trial.
    """
    if judge_lm is None:
        raise ValueError(
            "deepeval metric requires a judge LM. Pass "
            "``judge_lm=runtime.deepeval_judge``."
        )
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as exc:
        raise ImportError(
            "Building the deepeval DSPy metric requires the `deepeval` "
            "package. Install with `pip install ragdx[deepeval]`."
        ) from exc

    if weights is None:
        weights = dict(_DEFAULT_DEEPEVAL_WEIGHTS)

    # Reasonable default rubric. Two axes (groundedness + answer
    # relevance) so we get a multi-dimensional signal in a single call.
    if geval_criteria is None:
        geval_criteria = (
            "Score the answer against the retrieved context on TWO axes "
            "combined into one [0, 1] score:\n"
            "  (a) Groundedness: every claim in the answer is supported "
            "by the context. Fully grounded answers score higher; "
            "answers that invent facts score lower.\n"
            "  (b) Helpfulness: the answer actually responds to the "
            "question with substance, not just 'I don't know'.\n"
            "Multiply the two intuitions together and return the "
            "result as a scalar in [0, 1]."
        )

    # Build the metrics once -- deepeval metrics are reusable across
    # test cases. Wrapping each test case in fresh ``measure`` calls
    # below; the metric's internal state (.score / .reason) is reset
    # per call automatically.
    geval_metric = GEval(
        name="grounded_helpfulness",
        criteria=geval_criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge_lm,
        threshold=threshold,
    )
    answer_rel_metric = AnswerRelevancyMetric(
        model=judge_lm, threshold=threshold,
    )
    faith_metric = FaithfulnessMetric(
        model=judge_lm, threshold=threshold,
    )

    _component_metrics = {
        "g_eval": geval_metric,
        "answer_relevancy": answer_rel_metric,
        "faithfulness": faith_metric,
    }

    def _safe_measure(metric: Any, test_case: Any) -> float:
        try:
            metric.measure(test_case)
            s = float(getattr(metric, "score", 0.0) or 0.0)
            return max(0.0, min(1.0, s))
        except Exception as exc:  # pragma: no cover - depends on judge
            logger.debug(
                "deepeval %s scoring failed (treated as 0.0): %s",
                getattr(metric, "__name__", type(metric).__name__), exc,
            )
            return 0.0

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

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )

        total = 0.0
        any_succeeded = False
        for name, comp in _component_metrics.items():
            w = weights.get(name, 1.0)
            if w == 0:
                continue
            s = _safe_measure(comp, test_case)
            if s > 0:
                any_succeeded = True
            total += s * float(w)

        return total if any_succeeded else 0.0

    metric.__ragdx_kind__ = "deepeval_composite"  # type: ignore[attr-defined]
    metric.__ragdx_weights__ = dict(weights)  # type: ignore[attr-defined]
    metric.__ragdx_geval_criteria__ = geval_criteria  # type: ignore[attr-defined]
    return metric


__all__ = ["make_deepeval_metric"]
