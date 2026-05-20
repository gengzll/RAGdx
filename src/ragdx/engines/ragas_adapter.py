"""RAGAS evaluation adapter — now with real in-process evaluation.

Three execution paths are supported:

1. ``raw_scores`` provided  → normalize external precomputed scores (cheapest).
2. ``ragas`` installed       → call ``ragas.evaluate()`` against the prepared
   dataset and aggregate per-metric means into a :class:`EvaluationResult`.
3. ``ragas`` not installed   → return a "prepared_only" result containing the
   ragas-compatible records, so the caller can run RAGAS externally.

Path #2 is the production path. It requires an ``llm`` and ``embeddings``
object compatible with the installed ragas version. We let the caller inject
these via ``llm=``/``embeddings=`` kwargs, falling back to ragas defaults if
omitted (which in turn rely on ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from ragdx.core.normalization import RAGAS_MAP
from ragdx.schemas.models import DatasetRecord, EvaluationResult
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class RagasAdapter:
    """Adapter around the optional ``ragas`` evaluation library."""

    def normalize_scores(self, raw_scores: Mapping[str, float]) -> EvaluationResult:
        result = EvaluationResult(
            metadata={"tool": "ragas"},
            raw_tool_outputs={"ragas": dict(raw_scores)},
        )
        for metric, value in raw_scores.items():
            mapped = RAGAS_MAP.get(metric)
            if not mapped:
                logger.debug("Unknown ragas metric %s — kept in raw_tool_outputs only", metric)
                continue
            bucket, target = mapped
            getattr(result, bucket)[target] = float(value)
        return result

    def _to_ragas_records(self, records: Iterable[DatasetRecord]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in records:
            rows.append(
                {
                    "user_input": r.question,
                    "response": r.answer or "",
                    "retrieved_contexts": list(r.contexts),
                    "reference": r.ground_truth or "",
                    "reference_contexts": list(r.reference_contexts),
                }
            )
        return rows

    # ------------------------------------------------------------------ #
    # Real ragas evaluation                                               #
    # ------------------------------------------------------------------ #
    def _default_metric_modules(self) -> List[Any] | None:
        """Best-effort import of the most common ragas metrics.

        Returns ``None`` if ragas is not installed or none of the standard
        metrics can be imported (e.g. a major-version mismatch).
        """
        try:
            from ragas import metrics as _m  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on user env
            logger.warning("ragas metrics import failed: %s", exc)
            return None

        chosen: List[Any] = []
        for attr in (
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
        ):
            obj = getattr(_m, attr, None)
            if obj is not None:
                chosen.append(obj)
        return chosen or None

    def _evaluate_in_process(
        self,
        records: Sequence[DatasetRecord],
        *,
        metrics: Sequence[Any] | None,
        llm: Any | None,
        embeddings: Any | None,
        **kwargs: Any,
    ) -> EvaluationResult:
        try:
            from datasets import Dataset  # type: ignore[import-not-found]
            from ragas import evaluate as ragas_evaluate  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                "Real ragas evaluation requires both `ragas` and `datasets` installed. "
                "Install with `pip install ragdx[ragas]` (or `pip install ragas datasets`)."
            ) from exc

        used_metrics = list(metrics) if metrics else (self._default_metric_modules() or [])
        if not used_metrics:
            raise RuntimeError(
                "No ragas metrics could be imported. Install a compatible ragas release or "
                "pass `metrics=[...]` explicitly."
            )
        dataset = Dataset.from_list(self._to_ragas_records(records))
        logger.info(
            "Running ragas.evaluate on %d records with %d metrics",
            len(records), len(used_metrics),
        )
        eval_kwargs: dict[str, Any] = {"dataset": dataset, "metrics": used_metrics}
        if llm is not None:
            eval_kwargs["llm"] = llm
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings
        eval_kwargs.update(kwargs)
        result_obj = ragas_evaluate(**eval_kwargs)

        # ragas returns a Result object: try the public to_pandas first, then dict scores.
        raw_scores: dict[str, float] = {}
        try:
            df = result_obj.to_pandas()
            numeric = df.select_dtypes(include="number")
            raw_scores = {col: float(numeric[col].mean()) for col in numeric.columns}
        except Exception:
            # Older releases expose ``scores`` as a dict or list[dict]
            scores = getattr(result_obj, "scores", None)
            if isinstance(scores, dict):
                raw_scores = {k: float(v) for k, v in scores.items() if _is_number(v)}
            elif isinstance(scores, list) and scores and isinstance(scores[0], dict):
                # Average per metric across rows.
                acc: dict[str, list[float]] = {}
                for row in scores:
                    for k, v in row.items():
                        if _is_number(v):
                            acc.setdefault(k, []).append(float(v))
                raw_scores = {k: sum(vs) / len(vs) for k, vs in acc.items() if vs}
            else:
                raise RuntimeError(
                    "Unable to extract scores from ragas result; install pandas or upgrade ragas."
                )

        out = self.normalize_scores(raw_scores)
        out.metadata.update(
            {"record_count": len(records), "mode": "evaluated", "ragas_metrics": [m.__class__.__name__ if hasattr(m, "__class__") else str(m) for m in used_metrics]}
        )
        return out

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        records: Iterable[DatasetRecord],
        raw_scores: Mapping[str, float] | None = None,
        *,
        run_ragas: bool = False,
        metrics: Sequence[Any] | None = None,
        llm: Any | None = None,
        embeddings: Any | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate ``records`` with RAGAS.

        Parameters
        ----------
        records: iterable of :class:`DatasetRecord` (question / answer / contexts).
        raw_scores: if provided, skip ragas and normalize these external scores.
        run_ragas: when True (and ``raw_scores`` is None), actually invoke
            ``ragas.evaluate`` in-process. Defaults to False so the adapter
            stays cheap & predictable for users who want only normalization.
        metrics: optional explicit list of ragas metric instances/classes.
        llm / embeddings: ragas-compatible LLM and embeddings (required for
            metrics like ``faithfulness`` that call out to a model).
        kwargs: passed verbatim to ``ragas.evaluate``.
        """
        records = list(records)

        if raw_scores is not None:
            out = self.normalize_scores(raw_scores)
            out.metadata.update({"record_count": len(records), "mode": "precomputed"})
            return out

        if run_ragas:
            return self._evaluate_in_process(
                records, metrics=metrics, llm=llm, embeddings=embeddings, **kwargs,
            )

        # Pre-flight: do we even have ragas available?
        try:
            import ragas  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "ragas is not installed. Install with `pip install ragdx[ragas]`, "
                "or pass `raw_scores={...}` for normalization only."
            ) from exc

        return EvaluationResult(
            metadata={
                "tool": "ragas",
                "record_count": len(records),
                "mode": "prepared_only",
                "prepared_records": self._to_ragas_records(records),
                "note": (
                    "ragas is installed but evaluation was not invoked. Re-call with "
                    "run_ragas=True to compute scores in-process, or pass raw_scores=... "
                    "to normalize externally-computed scores."
                ),
            }
        )


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)
