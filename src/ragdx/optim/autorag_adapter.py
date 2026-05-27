"""AutoRAG **spec renderer** — builds an AutoRAG-style YAML payload.

.. important::
   This adapter does *not* invoke AutoRAG. It produces a YAML / dict
   spec that mirrors AutoRAG's configuration shape and returns it
   inside a :class:`ToolRunResult`. To actually execute the spec, the
   project relies on the executor's subprocess runner contract — set
   the ``RAGDX_AUTORAG_RUNNER_CMD`` environment variable to a command
   that consumes ``{config}`` (the rendered YAML path) and writes a
   ``{output}`` JSON with metric scores. See
   :func:`ragdx.cli.show_runner_templates` for examples.

The adapter is GT-aware: when the trainset has populated ground-truths it
includes reference-based metrics (``answer_correctness``, ``context_recall``)
in the spec; when GT is missing it falls back to reference-free signals
(``faithfulness``, ``answer_relevancy``, ``context_precision``, …).

This lets the same project drive AutoRAG on either a freshly-collected PDF
corpus (no labels yet) or a curated test set with GT.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ragdx.optim._gt_helpers import GTMode, gt_mode, select_metrics
from ragdx.schemas.models import DatasetRecord, OptimizationExperiment, ToolRunResult


class AutoRAGAdapter:
    def _resolve_mode(
        self,
        experiment: OptimizationExperiment,
        records: Iterable[DatasetRecord] | None,
    ) -> GTMode:
        """Pick GT mode from explicit experiment flag, then trainset, then default."""
        explicit = experiment.parameters.get("gt_mode")
        if explicit in ("with_gt", "no_gt"):
            return explicit
        if records is not None:
            return gt_mode(records)
        return "no_gt"

    def build_search_spec(
        self,
        experiment: OptimizationExperiment,
        parameters: dict[str, Any],
        *,
        records: Iterable[DatasetRecord] | None = None,
    ) -> dict[str, Any]:
        retriever = parameters.get("retriever", "hybrid")
        reranker = parameters.get("reranker", "none")
        mode = self._resolve_mode(experiment, records)

        # Metric selection — adapter caller can override via experiment.parameters
        # or via the trainset's GT signal.
        extra = experiment.parameters.get("extra_metrics") or []
        drop = experiment.parameters.get("drop_metrics") or []
        metrics = select_metrics(mode, extra=extra, drop=drop)

        # objective_metric defaults differ by mode (faithfulness is universal,
        # answer_correctness only makes sense when we actually have GT).
        default_objective = "answer_correctness" if mode == "with_gt" else "faithfulness"
        objective_metric = experiment.parameters.get("objective_metric", default_objective)

        return {
            "framework": "autorag",
            "gt_mode": mode,
            "objective_metric": objective_metric,
            "objectives": experiment.objectives,
            "metrics": metrics,
            "yaml_template": {
                "version": 1,
                "optimization": {
                    "strategy": experiment.search_strategy,
                    "trials": experiment.max_trials,
                },
                "node_lines": [
                    {
                        "name": "retrieval_line",
                        "nodes": [
                            {
                                "kind": "retrieval",
                                "name": retriever,
                                "params": {"top_k": parameters.get("top_k", 6)},
                            },
                            {
                                "kind": "reranker",
                                "name": reranker,
                                "params": {"enabled": reranker != "none"},
                            },
                            {
                                "kind": "chunking",
                                "name": "semantic_chunker",
                                "params": {
                                    "chunk_size": parameters.get("chunk_size", 512),
                                    "chunk_overlap": parameters.get("chunk_overlap", 64),
                                },
                            },
                        ],
                    }
                ],
                "postprocess": {
                    "context_ordering": parameters.get("context_ordering", "retrieval_score"),
                },
                "evaluation": {
                    "metrics": metrics,
                    "primary": objective_metric,
                    "requires_ground_truth": mode == "with_gt",
                },
            },
            "search_parameters": parameters,
        }

    # The env var consumed by ``OptimizationExecutor`` to launch the
    # external AutoRAG run. Surfaced here so callers can document /
    # validate the integration without reaching into the executor.
    RUNNER_ENV_VAR = "RAGDX_AUTORAG_RUNNER_CMD"

    def run(
        self,
        experiment: OptimizationExperiment,
        parameters: dict[str, Any],
        *,
        records: Iterable[DatasetRecord] | None = None,
    ) -> ToolRunResult:
        """Render an AutoRAG spec.

        .. warning::
           This does **not** execute AutoRAG. The returned
           :class:`ToolRunResult` carries the spec under ``.payload``;
           ``OptimizationExecutor`` writes it to a YAML config and
           invokes ``$RAGDX_AUTORAG_RUNNER_CMD`` to run the actual
           AutoRAG search. With no runner configured, execution falls
           back to simulated scoring (loud warning in `strict_execute`
           mode).
        """
        spec = self.build_search_spec(experiment, parameters, records=records)
        mode = spec["gt_mode"]
        runner_hint = (
            f" Set {self.RUNNER_ENV_VAR}='<cmd> --config {{config}} "
            "--output {output}' to execute this spec via your installed "
            "AutoRAG release."
        )
        if mode == "with_gt":
            note = (
                "Spec rendered for AutoRAG (with-GT mode). Reference-based metrics "
                "like answer_correctness / context_recall are included. The YAML "
                "node names mirror AutoRAG's shape — adjust to match your installed "
                "AutoRAG release if needed." + runner_hint
            )
        else:
            note = (
                "Spec rendered for AutoRAG (no-GT mode). Only reference-free metrics "
                "(faithfulness, answer_relevancy, context_precision, hallucination) are "
                "included. Provide an LLM-as-judge in your AutoRAG runtime so these "
                "metrics can be computed." + runner_hint
            )
        return ToolRunResult(tool="autorag", success=True, payload=spec, note=note)
