from __future__ import annotations

from typing import Any, Dict

from ragdx.schemas.models import OptimizationExperiment, ToolRunResult


class AutoRAGAdapter:
    def build_search_spec(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> Dict[str, Any]:
        retriever = parameters.get("retriever", "hybrid")
        reranker = parameters.get("reranker", "none")
        return {
            "framework": "autorag",
            "objective_metric": experiment.parameters.get("objective_metric", "answer_correctness"),
            "objectives": experiment.objectives,
            "yaml_template": {
                "version": 1,
                "optimization": {"strategy": experiment.search_strategy, "trials": experiment.max_trials},
                "node_lines": [
                    {
                        "name": "retrieval_line",
                        "nodes": [
                            {"kind": "retrieval", "name": retriever, "params": {"top_k": parameters.get("top_k", 6)}},
                            {"kind": "reranker", "name": reranker, "params": {"enabled": reranker != "none"}},
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
                "postprocess": {"context_ordering": parameters.get("context_ordering", "retrieval_score")},
            },
            "search_parameters": parameters,
        }

    def run(self, experiment: OptimizationExperiment, parameters: Dict[str, Any]) -> ToolRunResult:
        return ToolRunResult(
            tool="autorag",
            success=True,
            payload=self.build_search_spec(experiment, parameters),
            note="Config rendered. Save the YAML payload and adjust node names to match your installed AutoRAG release and data connectors.",
        )
