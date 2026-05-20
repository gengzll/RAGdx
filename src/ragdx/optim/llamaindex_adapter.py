from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import OptimizationExperiment, ToolRunResult


class LlamaIndexAdapter:
    def build_runner_spec(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "framework": "llamaindex",
            "entrypoint": "examples/run_llamaindex_trial.py",
            "objective_metric": experiment.parameters.get("objective_metric", "answer_correctness"),
            "objectives": experiment.objectives,
            "constraints": experiment.constraints,
            "program_contract": {
                "dataset_path": experiment.parameters.get("dataset_path", "examples/demo_dataset.jsonl"),
                "pipeline_module": experiment.parameters.get("pipeline_module", "examples.llamaindex_pipeline:create_query_engine"),
                "evaluator_mode": experiment.parameters.get("evaluator_mode", "offline"),
            },
            "runtime": {
                "index_kind": parameters.get("index_kind", "vector"),
                "retriever_top_k": parameters.get("top_k", 6),
                "response_mode": parameters.get("response_mode", "compact"),
                "node_parser": parameters.get("node_parser", "sentence_window"),
                "reranker": parameters.get("reranker", "none"),
                "temperature": parameters.get("temperature", 0.0),
            },
            "search_parameters": parameters,
        }

    def run(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> ToolRunResult:
        return ToolRunResult(
            tool="llamaindex",
            success=True,
            payload=self.build_runner_spec(experiment, parameters),
            note="Config rendered for a LlamaIndex query-engine runner. Set RAGDX_LLAMAINDEX_RUNNER_CMD to execute it.",
        )
