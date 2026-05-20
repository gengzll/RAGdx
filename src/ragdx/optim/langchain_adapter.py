"""LangChain optimization adapter.

Two modes are supported:

1. ``build_runner_spec`` / ``run`` — emit a YAML/dict config to be picked up by
   an external ``RAGDX_LANGCHAIN_RUNNER_CMD`` subprocess. This is the legacy
   contract preserved for back-compat.

2. ``make_in_process_runner`` — return a callable suitable for
   :class:`OptimizationExecutor`'s ``in_process_runners``. The callable resolves
   the user's pipeline factory (``examples.langchain_pipeline:create_pipeline``
   by default), runs each dataset record through it, and scores the answers
   against ground truth using a pluggable scorer.

Scoring is intentionally simple by default (token recall + faithfulness proxy)
because real evaluator integration is the caller's choice — ragas / ragchecker
can be plugged in via ``scorer=`` if available.
"""

from __future__ import annotations

import importlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from ragdx.schemas.models import (
    DatasetRecord,
    EvaluationResult,
    OptimizationExperiment,
    OptimizationTrial,
    ToolRunResult,
)
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)

ScorerFn = Callable[[List[Dict[str, Any]]], Dict[str, float]]


class LangChainAdapter:
    def build_runner_spec(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> Dict[str, Any]:
        llm_provider = parameters.get("llm_provider", "openai")
        return {
            "framework": "langchain",
            "entrypoint": "examples/run_langchain_trial.py",
            "objective_metric": experiment.parameters.get("objective_metric", "answer_correctness"),
            "objectives": experiment.objectives,
            "constraints": experiment.constraints,
            "program_contract": {
                "dataset_path": experiment.parameters.get("dataset_path", "examples/demo_dataset.jsonl"),
                "pipeline_module": experiment.parameters.get(
                    "pipeline_module", "examples.langchain_pipeline:create_pipeline"
                ),
                "evaluator_mode": experiment.parameters.get("evaluator_mode", "offline"),
            },
            "runtime": {
                "provider": llm_provider,
                "vectorstore": parameters.get("vectorstore", "faiss"),
                "retriever_k": parameters.get("top_k", 6),
                "search_type": parameters.get("search_type", "similarity"),
                "reranker": parameters.get("reranker", "none"),
                "temperature": parameters.get("temperature", 0.0),
            },
            "search_parameters": parameters,
        }

    def run(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> ToolRunResult:
        return ToolRunResult(
            tool="langchain",
            success=True,
            payload=self.build_runner_spec(experiment, parameters),
            note="Config rendered for a LangChain retrieval chain runner. "
                 "Set RAGDX_LANGCHAIN_RUNNER_CMD to execute it as a subprocess, "
                 "or use LangChainAdapter().make_in_process_runner() for in-process execution.",
        )

    # ------------------------------------------------------------------ #
    # In-process runner                                                    #
    # ------------------------------------------------------------------ #
    def make_in_process_runner(
        self,
        *,
        dataset_path: str | Path | None = None,
        records: Sequence[DatasetRecord] | None = None,
        scorer: ScorerFn | None = None,
        timeout_per_query: float | None = None,
    ) -> Callable[..., Dict[str, float]]:
        """Return an ``in_process_runners["langchain"]`` callable.

        Resolution order for the dataset:
        ``records`` (explicit) → ``dataset_path`` → ``experiment.parameters["dataset_path"]``
        (resolved at call-time) → ``examples/demo_dataset.jsonl``.

        The pipeline factory is loaded from ``experiment.parameters["pipeline_module"]``
        in ``module:attr`` form, called with the trial parameters as kwargs, and is
        expected to return a callable that accepts a ``DatasetRecord`` (or just a
        query string) and returns either a string answer or a dict containing at
        least ``answer`` and ``contexts``.
        """
        cached_records = list(records) if records is not None else None
        default_dataset = Path(dataset_path) if dataset_path else None
        score_fn = scorer or _default_token_scorer

        def runner(
            *,
            experiment: OptimizationExperiment,
            trial: OptimizationTrial,
            baseline: EvaluationResult,
        ) -> Dict[str, float]:
            ds_path = (
                default_dataset
                or Path(experiment.parameters.get("dataset_path", "examples/demo_dataset.jsonl"))
            )
            recs = cached_records if cached_records is not None else _load_jsonl_dataset(ds_path)
            if not recs:
                raise RuntimeError(f"Dataset {ds_path} produced zero records")

            factory_spec = experiment.parameters.get(
                "pipeline_module", "examples.langchain_pipeline:create_pipeline"
            )
            factory = _import_attr(factory_spec)
            logger.info(
                "Building LangChain pipeline factory=%s trial=%s",
                factory_spec, trial.trial_id,
                extra={"trial_id": trial.trial_id, "tool": "langchain"},
            )
            pipeline = factory(**trial.parameters)

            outputs: List[Dict[str, Any]] = []
            latencies: List[float] = []
            for rec in recs:
                start = time.perf_counter()
                try:
                    if timeout_per_query is not None:
                        result = _call_with_optional_timeout(
                            pipeline, rec, timeout=timeout_per_query
                        )
                    else:
                        result = _invoke_pipeline(pipeline, rec)
                except Exception as exc:  # one bad record shouldn't kill the trial
                    logger.warning(
                        "Pipeline error on record %s: %s",
                        getattr(rec, "question", "?")[:80], exc,
                    )
                    result = {"answer": "", "contexts": [], "error": str(exc)}
                latencies.append((time.perf_counter() - start) * 1000.0)
                outputs.append(
                    {
                        "question": rec.question,
                        "ground_truth": rec.ground_truth or "",
                        "answer": result.get("answer", "") if isinstance(result, dict) else str(result),
                        "contexts": result.get("contexts", []) if isinstance(result, dict) else [],
                    }
                )

            scores = dict(score_fn(outputs))
            if "latency_ms" in experiment.objectives or "latency_ms_max" in experiment.constraints:
                scores.setdefault("latency_ms", statistics.mean(latencies) if latencies else 0.0)
            logger.info(
                "LangChain trial scored trial=%s scores=%s",
                trial.trial_id, {k: round(v, 4) for k, v in scores.items()},
                extra={"trial_id": trial.trial_id, "tool": "langchain"},
            )
            return scores

        return runner


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _import_attr(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(
            f"pipeline_module must be in 'module:attr' form, got {spec!r}"
        )
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def _load_jsonl_dataset(path: Path) -> List[DatasetRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records: List[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
            try:
                records.append(DatasetRecord(**payload))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no} invalid DatasetRecord: {exc}") from exc
    return records


def _invoke_pipeline(pipeline: Any, record: DatasetRecord) -> Any:
    """Call ``pipeline`` with the most ergonomic supported signature."""
    if callable(pipeline):
        try:
            return pipeline(record)
        except TypeError:
            return pipeline(record.question)
    # LangChain Runnable
    if hasattr(pipeline, "invoke"):
        return pipeline.invoke({"question": record.question})
    raise TypeError(f"Don't know how to invoke pipeline of type {type(pipeline).__name__}")


def _call_with_optional_timeout(pipeline: Any, record: DatasetRecord, timeout: float) -> Any:
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_invoke_pipeline, pipeline, record)
        return future.result(timeout=timeout)


_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _default_token_scorer(outputs: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Token-overlap heuristic. Not a real evaluator — only a sane default."""
    outs = list(outputs)
    if not outs:
        return {}
    answer_correctness: List[float] = []
    citation_accuracy: List[float] = []
    context_recall: List[float] = []
    for row in outs:
        gt_tokens = set(_tokens(row.get("ground_truth", "")))
        ans_tokens = set(_tokens(row.get("answer", "")))
        ctx_tokens = set(_tokens(" ".join(row.get("contexts") or [])))
        if gt_tokens:
            answer_correctness.append(len(gt_tokens & ans_tokens) / max(1, len(gt_tokens)))
            context_recall.append(len(gt_tokens & ctx_tokens) / max(1, len(gt_tokens)))
        else:
            answer_correctness.append(0.0)
            context_recall.append(0.0)
        if ans_tokens:
            citation_accuracy.append(len(ans_tokens & ctx_tokens) / max(1, len(ans_tokens)))
        else:
            citation_accuracy.append(0.0)

    return {
        "answer_correctness": round(statistics.mean(answer_correctness), 4),
        "citation_accuracy": round(statistics.mean(citation_accuracy), 4),
        "context_recall": round(statistics.mean(context_recall), 4),
        "faithfulness": round(statistics.mean(citation_accuracy), 4),
    }
