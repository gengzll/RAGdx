from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import OptimizationExperiment, ToolRunResult


class DSPyAdapter:
    def build_optimizer_spec(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> Dict[str, Any]:
        optimizer = parameters.get("optimizer", "MIPROv2")
        return {
            "framework": "dspy",
            "optimizer": optimizer,
            "objective_metric": experiment.parameters.get("objective_metric", "answer_correctness"),
            "objectives": experiment.objectives,
            "program_contract": {
                "input_fields": ["question", "contexts"],
                "output_fields": ["answer", "citations"],
                "expected_tunable_parts": ["instructions", "fewshot_demos", "citation_formatting", "decomposition"],
            },
            "search_parameters": parameters,
            "compile_hints": {
                "bayesian_optimizer": optimizer == "MIPROv2",
                "fewshot_enabled": parameters.get("fewshot_count", 0) > 0,
                "decomposition": parameters.get("decomposition", False),
            },
        }

    def run(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> ToolRunResult:
        return ToolRunResult(
            tool="dspy",
            success=True,
            payload=self.build_optimizer_spec(experiment, parameters),
            note="Config rendered. Attach your DSPy program, trainset, and metric function in the runtime environment.",
        )
