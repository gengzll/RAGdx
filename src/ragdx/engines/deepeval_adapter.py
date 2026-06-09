"""DeepEval evaluation adapter — parallel to :mod:`ragdx.engines.ragas_adapter`.

DeepEval (https://github.com/confident-ai/deepeval) is a peer of ragas
with two notable differences that motivated adding it as an alternative
evaluator:

* **G-Eval** -- a configurable chain-of-thought LLM-as-judge metric
  that's much less prone to the saturation we see with ragas's
  ``faithfulness`` on permissive judge LMs (GLM-4-Flash, etc.).
* **Test-case style API** -- per-record ``LLMTestCase`` + ``metric.measure``
  rather than ragas's batch ``evaluate(dataset, metrics)`` model. We
  hide that difference behind the same :class:`EvaluationResult`
  contract so the rest of ragdx (objectives, diagnosis, dashboards)
  is agnostic.

Three execution paths, mirroring the ragas adapter:

1. ``raw_scores`` provided  → normalize external precomputed scores.
2. ``deepeval`` installed   → call ``deepeval.evaluate()`` against the
   prepared test cases and aggregate per-metric means.
3. ``deepeval`` not installed → return a "prepared_only" payload.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragdx.core.normalization import DEEPEVAL_MAP
from ragdx.optim._gt_helpers import (
    gt_mode,
    has_answers,
    has_contexts,
    has_ground_truth,
)
from ragdx.schemas.models import DatasetRecord, EvaluationResult
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class DeepEvalAdapter:
    """Adapter around the optional ``deepeval`` evaluation library.

    Public surface intentionally mirrors :class:`RagasAdapter` so the
    rest of ragdx can swap evaluators behind a single
    ``--evaluator {ragas, deepeval}`` flag.
    """

    # ------------------------------------------------------------------
    # Score normalization (path 1: precomputed scores)
    # ------------------------------------------------------------------
    def normalize_scores(self, raw_scores: Mapping[str, float]) -> EvaluationResult:
        """Bucket raw deepeval scores into a :class:`EvaluationResult`.

        Unknown metric names are kept in ``raw_tool_outputs`` so caller
        can still see them; only mapped ones land in the typed
        retrieval / generation / e2e buckets.
        """
        result = EvaluationResult(
            metadata={"tool": "deepeval"},
            raw_tool_outputs={"deepeval": dict(raw_scores)},
        )
        for metric, value in raw_scores.items():
            mapped = DEEPEVAL_MAP.get(metric)
            if not mapped:
                logger.debug(
                    "Unknown deepeval metric %s — kept in raw_tool_outputs only",
                    metric,
                )
                continue
            bucket, target = mapped
            getattr(result, bucket)[target] = float(value)
        return result

    # ------------------------------------------------------------------
    # Test-case construction
    # ------------------------------------------------------------------
    def _to_test_cases(
        self,
        records: Iterable[DatasetRecord],
    ) -> list[Any]:
        """Convert ragdx records into ``deepeval.test_case.LLMTestCase``.

        Field mapping:
        * question        → ``input``
        * answer          → ``actual_output``
        * contexts        → ``retrieval_context``
        * ground_truth    → ``expected_output``
        """
        from deepeval.test_case import LLMTestCase

        cases: list[Any] = []
        for r in records:
            kwargs: dict[str, Any] = {
                "input": r.question,
                "actual_output": r.answer or "",
                "retrieval_context": list(r.contexts),
            }
            if r.ground_truth:
                kwargs["expected_output"] = r.ground_truth
            cases.append(LLMTestCase(**kwargs))
        return cases

    # ------------------------------------------------------------------
    # Default metric set (no-GT vs with-GT auto-pick)
    # ------------------------------------------------------------------
    def _default_metrics(
        self,
        mode: str,
        *,
        model: Any | None = None,
        threshold: float = 0.5,
    ) -> list[Any] | None:
        """Standard deepeval metric set for the given GT mode.

        * with_gt: precision + recall + relevancy + faithfulness +
          answer_relevancy
        * no_gt:   relevancy + faithfulness + answer_relevancy
          (skip the ones that need ``expected_output``)

        Returns ``None`` if deepeval is not installed.
        """
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                ContextualRelevancyMetric,
                FaithfulnessMetric,
            )
        except Exception as exc:  # pragma: no cover - depends on user env
            logger.warning("deepeval metrics import failed: %s", exc)
            return None

        common_kwargs: dict[str, Any] = {"threshold": threshold}
        # DeepEval metrics accept ``model=<DeepEvalBaseLLM | str | None>``.
        # ``None`` falls back to deepeval's default (gpt-4o-mini).
        if model is not None:
            common_kwargs["model"] = model

        metrics: list[Any] = [
            AnswerRelevancyMetric(**common_kwargs),
            FaithfulnessMetric(**common_kwargs),
            ContextualRelevancyMetric(**common_kwargs),
        ]
        if mode == "with_gt":
            metrics += [
                ContextualPrecisionMetric(**common_kwargs),
                ContextualRecallMetric(**common_kwargs),
            ]
        return metrics

    # ------------------------------------------------------------------
    # Real evaluation (path 2)
    # ------------------------------------------------------------------
    def _evaluate_in_process(
        self,
        records: Sequence[DatasetRecord],
        *,
        metrics: Sequence[Any] | None,
        model: Any | None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> EvaluationResult:
        try:
            from deepeval import evaluate as deepeval_evaluate
        except Exception as exc:
            raise ImportError(
                "Real deepeval evaluation requires `deepeval` installed. "
                "Install with `pip install ragdx[deepeval]` (or `pip install deepeval`)."
            ) from exc

        mode = gt_mode(records)
        if metrics is None:
            picked = self._default_metrics(mode, model=model, threshold=threshold)
            if not picked:
                raise RuntimeError(
                    "No deepeval metrics could be imported. Install a compatible "
                    "deepeval release or pass `metrics=[...]` explicitly."
                )
            used_metrics = list(picked)
        else:
            used_metrics = list(metrics)

        test_cases = self._to_test_cases(records)
        logger.info(
            "Running deepeval.evaluate on %d records with %d metrics",
            len(records), len(used_metrics),
        )

        # deepeval.evaluate returns ``EvaluationResult`` (NOT ours, it's
        # deepeval's own class). It has ``.test_results`` -- a list of
        # ``TestResult`` objects, each carrying per-metric ``MetricData``
        # entries with the actual score. Aggregate per-metric means.
        try:
            de_result = deepeval_evaluate(
                test_cases=test_cases, metrics=used_metrics, **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"deepeval.evaluate failed: {type(exc).__name__}: {exc}"
            ) from exc

        raw_scores = self._aggregate_deepeval_result(de_result)

        out = self.normalize_scores(raw_scores)
        out.metadata.update(
            {
                "record_count": len(records),
                "mode": "evaluated",
                "deepeval_metrics": [self._metric_name(m) for m in used_metrics],
                "data_diagnostics": {
                    "has_ground_truth": has_ground_truth(records),
                    "has_answers": has_answers(records),
                    "has_contexts": has_contexts(records),
                },
            }
        )
        return out

    @staticmethod
    def _aggregate_deepeval_result(de_result: Any) -> dict[str, float]:
        """Pull per-metric mean scores out of deepeval's ``EvaluationResult``.

        deepeval gives one ``MetricData`` per (test_case x metric) pair;
        we want the mean per metric across test cases (matches ragas's
        DataFrame-mean semantics).
        """
        results = getattr(de_result, "test_results", None) or []
        acc: dict[str, list[float]] = {}
        for tr in results:
            metric_data = getattr(tr, "metrics_data", None) or getattr(
                tr, "metrics_metadata", None) or []
            for md in metric_data:
                name = getattr(md, "name", None)
                score = getattr(md, "score", None)
                if name is None or score is None:
                    continue
                try:
                    s = float(score)
                except (TypeError, ValueError):
                    continue
                acc.setdefault(str(name), []).append(s)
        return {k: sum(v) / len(v) for k, v in acc.items() if v}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _metric_name(metric: Any) -> str:
        return (
            getattr(metric, "__name__", None)
            or getattr(getattr(metric, "__class__", type(metric)), "__name__", "")
            or str(metric)
        )

    # ------------------------------------------------------------------
    # Public API (mirrors RagasAdapter.evaluate)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        records: Iterable[DatasetRecord],
        raw_scores: Mapping[str, float] | None = None,
        *,
        run_deepeval: bool = False,
        metrics: Sequence[Any] | None = None,
        model: Any | None = None,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate ``records`` with DeepEval.

        Parameters
        ----------
        records: iterable of :class:`DatasetRecord`.
        raw_scores: if provided, skip deepeval and normalize external scores.
        run_deepeval: when True (and ``raw_scores`` is None), actually
            invoke ``deepeval.evaluate`` in-process.
        metrics: optional explicit list of deepeval metric instances.
            ``None`` auto-picks the canonical no-GT / with-GT set.
        model: a ``DeepEvalBaseLLM`` instance to use as judge.
            ``None`` lets deepeval fall back to its default
            (gpt-4o-mini; requires ``OPENAI_API_KEY``).
        threshold: per-metric pass/fail threshold (default 0.5). Doesn't
            change the numeric score; only affects deepeval's notion of
            "passed" vs "failed". We still report the raw scalar.
        kwargs: passed verbatim to ``deepeval.evaluate``.
        """
        records = list(records)

        if raw_scores is not None:
            out = self.normalize_scores(raw_scores)
            out.metadata.update(
                {"record_count": len(records), "mode": "precomputed"}
            )
            return out

        if run_deepeval:
            return self._evaluate_in_process(
                records, metrics=metrics, model=model, threshold=threshold,
                **kwargs,
            )

        # Pre-flight: do we even have deepeval available?
        try:
            import deepeval  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "deepeval is not installed. Install with "
                "`pip install ragdx[deepeval]`, or pass `raw_scores={...}` "
                "for normalization only."
            ) from exc

        return EvaluationResult(
            metadata={
                "tool": "deepeval",
                "record_count": len(records),
                "mode": "prepared_only",
                "note": (
                    "deepeval is installed but evaluation was not invoked. "
                    "Re-call with run_deepeval=True to compute scores "
                    "in-process, or pass raw_scores=... to normalize "
                    "externally-computed scores."
                ),
            }
        )


__all__ = ["DeepEvalAdapter"]
