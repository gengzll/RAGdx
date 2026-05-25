"""RAGChecker adapter with optional in-process execution.

Mirrors :class:`RagasAdapter`. If ``run_ragchecker=True`` and the
``ragchecker`` package is installed, calls ``RAGResults.run_checking`` directly
and aggregates per-metric averages.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ragdx.core.normalization import RAGCHECKER_MAP
from ragdx.optim._gt_helpers import (
    filter_metrics_by_data,
    gt_mode,
    has_answers,
    has_contexts,
    has_ground_truth,
    validate_metrics_for_mode,
)
from ragdx.schemas.models import DatasetRecord, EvaluationResult
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class RAGCheckerAdapter:
    def normalize_scores(self, raw_scores: Mapping[str, float]) -> EvaluationResult:
        result = EvaluationResult(
            metadata={"tool": "ragchecker"},
            raw_tool_outputs={"ragchecker": dict(raw_scores)},
        )
        for metric, value in raw_scores.items():
            mapped = RAGCHECKER_MAP.get(metric)
            if not mapped:
                logger.debug("Unknown ragchecker metric %s — kept in raw_tool_outputs only", metric)
                continue
            bucket, target = mapped
            getattr(result, bucket)[target] = float(value)
        return result

    def _to_ragchecker_records(self, records: Iterable[DatasetRecord]) -> list[dict[str, Any]]:
        # ragchecker's RAGResult dataclass requires a non-default ``query_id``
        # and expects ``retrieved_context`` as a list of {doc_id, text} dicts
        # (RetrievedDoc). Passing a bare list[str] makes ``RAGResults.from_dict``
        # raise KeyError('query_id') / fail to deserialize the contexts.
        rows = []
        for i, r in enumerate(records):
            rows.append(
                {
                    "query_id": f"q{i}",
                    "query": r.question,
                    "gt_answer": r.ground_truth or "",
                    "response": r.answer or "",
                    "retrieved_context": [
                        {"doc_id": f"q{i}-d{j}", "text": ctx}
                        for j, ctx in enumerate(r.contexts)
                    ],
                }
            )
        return rows

    def _evaluate_in_process(
        self,
        records: Sequence[DatasetRecord],
        *,
        evaluator: Any | None = None,
        metric_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        try:
            from ragchecker import RAGChecker, RAGResults  # type: ignore[import-not-found]
        except Exception as exc:
            raise ImportError(
                "Real ragchecker evaluation requires `ragchecker` installed. "
                "Install with `pip install ragdx[ragchecker]`."
            ) from exc

        # Pre-flight data check: RAGChecker's metrics all rely on ``response``
        # (the system answer); claim_recall additionally needs gt_answer; every
        # metric needs retrieved_context. Without these we'd be averaging
        # silently-wrong 0s, so we filter or fail loudly instead.
        if not has_answers(records):
            raise RuntimeError(
                "RAGChecker requires `answer` on every record (it decomposes the "
                "answer into atomic claims). The supplied records have no answers; "
                "run your RAG pipeline first to populate them."
            )
        if not has_contexts(records):
            raise RuntimeError(
                "RAGChecker requires `contexts` (retrieved chunks) to verify claims. "
                "The supplied records have no contexts."
            )
        mode = gt_mode(records)
        requested = list(metric_names) if metric_names else list(RAGCHECKER_MAP.keys())

        # Declarative pre-flight: requested metrics must be in the canonical set
        # for this gt_mode (raise loudly rather than silently dropping).
        mode_errors = validate_metrics_for_mode(requested, mode)
        if mode_errors:
            raise RuntimeError(
                f"Requested RAGChecker metrics incompatible with gt_mode={mode!r}: "
                f"{mode_errors}. Use ragdx.optim._gt_helpers.select_metrics({mode!r}) "
                "to see the canonical metric set for this mode."
            )

        # Defensive data-driven check (records may carry the mode but have
        # broken/empty fields). Raise rather than silently drop.
        kept, skipped = filter_metrics_by_data(requested, records)
        if skipped:
            raise RuntimeError(
                f"Records do not actually carry the data needed by these metrics: "
                f"{skipped}. Check that answers/contexts were populated by the "
                "RAG pipeline."
            )

        prepared = self._to_ragchecker_records(records)
        results = RAGResults.from_dict({"results": prepared})
        checker = evaluator or RAGChecker(**kwargs)
        metric_args: dict[str, Any] = {"metrics": kept}
        logger.info("Running ragchecker on %d records with metrics=%s", len(prepared), kept)
        checker.evaluate(results, **metric_args)

        # Extract per-record metrics and average them
        acc: dict[str, list[float]] = {}
        rows: list[Any] = getattr(results, "results", []) or []
        for row in rows:
            metrics_dict = getattr(row, "metrics", None) or (row.get("metrics") if isinstance(row, dict) else None)
            if not metrics_dict:
                continue
            for k, v in metrics_dict.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    acc.setdefault(k, []).append(float(v))
        if not acc:
            raise RuntimeError(
                "ragchecker did not return any numeric per-record metrics; check evaluator config."
            )
        averages = {k: sum(vs) / len(vs) for k, vs in acc.items()}
        out = self.normalize_scores(averages)
        out.metadata.update(
            {
                "record_count": len(records),
                "mode": "evaluated",
                "requested_metrics": requested,
                "skipped_metrics": skipped,
                "data_diagnostics": {
                    "has_ground_truth": has_ground_truth(records),
                    "has_answers": has_answers(records),
                    "has_contexts": has_contexts(records),
                },
            }
        )
        return out

    def evaluate(
        self,
        records: Iterable[DatasetRecord],
        raw_scores: Mapping[str, float] | None = None,
        *,
        run_ragchecker: bool = False,
        evaluator: Any | None = None,
        metric_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        records = list(records)

        if raw_scores is not None:
            out = self.normalize_scores(raw_scores)
            out.metadata.update({"record_count": len(records), "mode": "precomputed"})
            return out

        if run_ragchecker:
            return self._evaluate_in_process(
                records, evaluator=evaluator, metric_names=metric_names, **kwargs,
            )

        try:
            import ragchecker  # noqa: F401
        except Exception as exc:
            raise ImportError(
                "ragchecker is not installed. Install with `pip install ragdx[ragchecker]`, "
                "or pass `raw_scores={...}` for normalization only."
            ) from exc

        return EvaluationResult(
            metadata={
                "tool": "ragchecker",
                "record_count": len(records),
                "mode": "prepared_only",
                "prepared_records": self._to_ragchecker_records(records),
                "note": (
                    "ragchecker is installed but evaluation was not invoked. Re-call with "
                    "run_ragchecker=True to compute scores in-process, or pass raw_scores=... "
                    "to normalize externally-computed scores."
                ),
            }
        )
