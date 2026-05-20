"""
Optimization Planning Engine

Main Idea:
This module provides intelligent planning for RAG pipeline optimization. It generates structured optimization plans with experiments, parameter ranges, and evaluation strategies based on diagnosis reports.

Functionalities:
- Plan generation: Creates optimization plans with prioritized experiments
- LLM-enhanced planning: Uses LLMs to refine and improve optimization strategies
- Multi-strategy support: Bayesian optimization, evolutionary algorithms, and custom strategies
- Metric-aware planning: Considers metric types (maximize/minimize) and component relationships
- Budget allocation: Distributes trial budgets across experiments based on expected impact

Supported optimization strategies:
- Bayesian optimization for continuous parameters
- Evolutionary algorithms for discrete/complex spaces
- LLM-guided planning for novel approaches

Usage:
Basic planning:

    from ragdx.optim.planner import OptimizationPlanner

    planner = OptimizationPlanner()
    plan = planner.build_plan(diagnosis_report, result=evaluation_result)

With LLM enhancement:

    llm_planner = OptimizationPlanner(llm_callable=my_llm_function)
    plan = llm_planner.build_plan(diagnosis_report, result=evaluation_result, strategy="bayesian", budget=20)

The planner generates OptimizationPlan objects with experiments, parameter spaces, and evaluation instructions.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

from ragdx.schemas.models import DiagnosisReport, EvaluationResult, OptimizationExperiment, OptimizationPlan, SearchStrategy


MAXIMIZE_METRICS = {
    "context_recall",
    "context_precision",
    "context_entities_recall",
    "hit_rate_at_k",
    "faithfulness",
    "response_relevancy",
    "context_utilization",
    "answer_correctness",
    "citation_accuracy",
    "user_success_rate",
}
MINIMIZE_METRICS = {"hallucination", "noise_sensitivity", "latency_ms", "cost_usd"}
COMPONENT_METRICS = {
    "retrieval": ["context_recall", "context_precision", "citation_accuracy", "latency_ms", "cost_usd"],
    "generation": ["faithfulness", "response_relevancy", "citation_accuracy", "hallucination", "cost_usd"],
    "pipeline": ["answer_correctness", "citation_accuracy", "faithfulness", "context_recall", "latency_ms", "cost_usd"],
}


class OptimizationPlanner:
    def __init__(self, llm_callable: Callable[[str], str | Dict[str, Any]] | None = None):
        self.llm_callable = llm_callable

    def _coerce_json(self, payload: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        text = (payload or "").strip()
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        return json.loads(text)

    def _component_metrics(self, component: str, objective_metric: str) -> List[str]:
        metrics = list(COMPONENT_METRICS.get(component, COMPONENT_METRICS["pipeline"]))
        if objective_metric not in metrics:
            metrics.insert(0, objective_metric)
        seen = []
        for metric in metrics:
            if metric not in seen:
                seen.append(metric)
        return seen

    def _target_for_metric(self, metric: str, current: float | None, expected: float | None) -> float:
        if metric in MAXIMIZE_METRICS:
            base = 0.75 if current is None and expected is None else max(x for x in [current, expected, 0.0] if x is not None)
            if expected is not None and current is not None and current < expected:
                return round(min(0.99, max(expected, current + max(0.02, 0.35 * (expected - current)))), 4)
            if current is not None:
                return round(min(0.99, current + max(0.01, (1.0 - current) * 0.12)), 4)
            return round(min(0.99, base), 4)
        current = 0.12 if current is None and expected is None else min(x for x in [current, expected, 1.0] if x is not None)
        if expected is not None and current is not None and current > expected:
            return round(max(0.0, expected), 4)
        if metric == "latency_ms":
            return round(max(50.0, current * 0.92 if current is not None else 4000.0), 4)
        if metric == "cost_usd":
            return round(max(0.0001, current * 0.93 if current is not None else 0.02), 4)
        return round(max(0.0, current * 0.9 if current is not None else 0.1), 4)

    def _weights_for_component(
        self,
        component: str,
        objective_metric: str,
        result: EvaluationResult | None = None,
        report: DiagnosisReport | None = None,
    ) -> Dict[str, float]:
        metrics = self._component_metrics(component, objective_metric)
        expected = report.expected_thresholds if report else {}
        metric_gaps = report.metric_gaps if report else {}
        scores: Dict[str, float] = {}
        for metric in metrics:
            current = result.score(metric) if result else None
            target = self._target_for_metric(metric, current, expected.get(metric))
            if metric in MAXIMIZE_METRICS:
                gap = max(metric_gaps.get(metric, 0.0), max(0.0, (target - (current or 0.0))))
                urgency = gap + (0.08 if current is None else 0.0)
            else:
                current_v = 0.0 if current is None else current
                allowed = target
                gap = max(0.0, current_v - allowed)
                urgency = gap + (0.05 if current is None else 0.0)
            base_importance = 0.8 if metric == objective_metric else 0.45
            if metric in {"citation_accuracy", "faithfulness"}:
                base_importance += 0.12
            if metric in {"latency_ms", "cost_usd"}:
                base_importance += 0.08
            if metric in {"hallucination", "noise_sensitivity"}:
                base_importance += 0.18
            scores[metric] = base_importance + urgency
        total = sum(scores.values()) or 1.0
        return {metric: round(value / total, 4) for metric, value in scores.items()}

    def _constraints(self, result: EvaluationResult | None = None, report: DiagnosisReport | None = None) -> Dict[str, float]:
        baseline = {
            "hallucination": result.score("hallucination") if result else None,
            "noise_sensitivity": result.score("noise_sensitivity") if result else None,
            "latency_ms": result.score("latency_ms") if result else None,
            "cost_usd": result.score("cost_usd") if result else None,
        }
        expected = report.expected_thresholds if report else {}
        constraints: Dict[str, float] = {}
        for metric in ["hallucination", "noise_sensitivity", "latency_ms", "cost_usd"]:
            current = baseline.get(metric)
            expected_target = expected.get(metric)
            target = self._target_for_metric(metric, current, expected_target)
            key = f"{metric}_max"
            if current is None:
                constraints[key] = target
                continue
            if metric in {"latency_ms", "cost_usd"}:
                slack = 250.0 if metric == "latency_ms" else 0.003
                constraints[key] = round(min(target + slack, current * 1.08 + slack), 4)
            else:
                constraints[key] = round(min(target + 0.015, current + 0.01), 4)
        return constraints



    def _component_for_stage(self, stage: str, target_component: str) -> str:
        return {"corpus": "retrieval", "retrieval": "retrieval", "generation": "generation", "orchestration": "pipeline", "joint": "pipeline"}.get(stage, target_component)

    def _metric_direction(self, metric: str) -> str:
        if metric in MAXIMIZE_METRICS:
            return "maximize"
        if metric in MINIMIZE_METRICS:
            return "minimize"
        return "monitor"

    def _baseline_metrics_for_component(self, component: str, objective_metric: str, result: EvaluationResult | None) -> Dict[str, float | None]:
        return {metric: (result.score(metric) if result else None) for metric in self._component_metrics(component, objective_metric)}

    def _target_semantics(self, metric: str, current: float | None, target: float) -> Dict[str, Any]:
        direction = self._metric_direction(metric)
        if direction == "maximize":
            if current is None:
                mode = "target"
                regression_cap = None
            elif target > current + 1e-9:
                mode = "improve"
                regression_cap = round(max(0.0, current - 0.01), 4)
            else:
                mode = "maintain"
                regression_cap = round(max(0.0, current - 0.01), 4)
            out = {"direction": direction, "mode": mode, "target_value": target}
            if current is not None:
                out["baseline_value"] = round(current, 4)
                out["delta_from_baseline"] = round(target - current, 4)
            if regression_cap is not None:
                out["min_acceptable"] = regression_cap
            return out
        # minimize
        if current is None:
            mode = "target"
            cap = None
        elif target < current - 1e-9:
            mode = "reduce"
            cap = round(current * 1.05, 4) if metric in {"latency_ms", "cost_usd"} else round(current + 0.01, 4)
        else:
            mode = "maintain"
            cap = round(current * 1.05, 4) if metric in {"latency_ms", "cost_usd"} else round(current + 0.01, 4)
        out = {"direction": direction, "mode": mode, "target_value": target}
        if current is not None:
            out["baseline_value"] = round(current, 4)
            out["delta_from_baseline"] = round(target - current, 4)
        if cap is not None:
            out["max_acceptable"] = cap
        return out

    def _build_metric_plan(self, component: str, objective_metric: str, result: EvaluationResult | None, report: DiagnosisReport | None) -> Tuple[Dict[str, Any], Dict[str, float | None], Dict[str, str], Dict[str, float], Dict[str, float]]:
        baseline_metrics = self._baseline_metrics_for_component(component, objective_metric, result)
        directions = {metric: self._metric_direction(metric) for metric in baseline_metrics}
        target_thresholds: Dict[str, float] = {}
        target_specs: Dict[str, Any] = {}
        expected = report.expected_thresholds if report else {}
        for metric, current in baseline_metrics.items():
            target = self._target_for_metric(metric, current, expected.get(metric))
            if metric in MAXIMIZE_METRICS and current is not None:
                target = max(round(current, 4), target)
            if metric in MINIMIZE_METRICS and current is not None:
                target = min(round(current, 4), target)
            target_thresholds[metric] = round(target, 4)
            target_specs[metric] = self._target_semantics(metric, current, round(target, 4))
        objective_weights = self._weights_for_component(component, objective_metric, result=result, report=report)
        return target_specs, baseline_metrics, directions, target_thresholds, objective_weights

    def _corpus_space(self) -> Dict[str, List[object]]:
        return {
            "parser": ["pymupdf_layout", "unstructured", "html_semantic"],
            "document_structure": ["preserve_sections", "flat_chunks"],
            "chunk_size": [384, 512, 768, 1024],
            "chunk_overlap": [32, 64, 96, 128],
            "table_strategy": ["repeat_headers", "table_to_text"],
        }

    def _retrieval_space(self) -> Dict[str, List[object]]:
        return {
            "retriever": ["bm25", "vector", "hybrid"],
            "embedding_model": ["bge-m3", "e5-large-v2", "gte-large"],
            "reranker": ["none", "bge-reranker", "cross-encoder"],
            "top_k": [4, 6, 8, 10],
            "query_rewrite": ["none", "multi_query", "decompose"],
            "context_ordering": ["retrieval_score", "section_then_score", "diverse_then_score"],
        }

    def _generation_space(self) -> Dict[str, List[object]]:
        return {
            "optimizer": ["MIPROv2", "BootstrapFewShot", "GEPA"],
            "fewshot_count": [0, 2, 4, 6],
            "prompt_style": ["grounded_qa", "claim_then_evidence", "citation_first"],
            "temperature": [0.0, 0.2, 0.4],
            "max_citations": [1, 2, 3],
            "verifier": ["none", "claim_checker"],
        }

    def _orchestration_space(self) -> Dict[str, List[object]]:
        return {
            "retry_retrieval": [False, True],
            "followup_question": [False, True],
            "abstention_policy": ["lenient", "balanced", "strict"],
            "context_budget": [3000, 4500, 6000],
            "planner": ["single_step", "two_pass"],
        }

    def _stack_runtime_space(self, framework: str) -> Dict[str, List[object]]:
        if framework == "langchain":
            return {
                "vectorstore": ["faiss", "chroma"],
                "search_type": ["similarity", "mmr"],
                "top_k": [4, 6, 8],
                "reranker": ["none", "cross-encoder"],
                "temperature": [0.0, 0.2],
                "llm_provider": ["openai"],
            }
        return {
            "index_kind": ["vector", "hybrid"],
            "node_parser": ["sentence_window", "hierarchical"],
            "response_mode": ["compact", "tree_summarize"],
            "top_k": [4, 6, 8],
            "reranker": ["none", "cohere"],
            "temperature": [0.0, 0.2],
        }

    def _joint_space(self) -> Dict[str, List[object]]:
        return {
            "retrieval_profile": ["recall_heavy", "balanced", "precision_heavy"],
            "generator_profile": ["grounded_qa", "citation_first"],
            "verifier": ["none", "claim_checker"],
            "context_budget": [3000, 4500, 6000],
        }

    def _llm_reasoning_prompt(
        self,
        report: DiagnosisReport,
        result: EvaluationResult | None,
        objective_metric: str,
        strategy: SearchStrategy,
        budget: int,
        heuristic_plan: OptimizationPlan,
    ) -> str:
        payload = {
            "objective_metric": objective_metric,
            "strategy": strategy,
            "budget": budget,
            "baseline_metrics": {
                "retrieval": result.retrieval if result else {},
                "generation": result.generation if result else {},
                "e2e": result.e2e if result else {},
            },
            "diagnosis": report.model_dump(),
            "heuristic_plan": heuristic_plan.model_dump(),
            "instruction": {
                "task": "Refine the optimization plan so targets, weights, constraints, and budget are baseline-aware and diagnosis-aware.",
                "rules": [
                    "Do not set weak targets or loose constraints when the baseline is already stronger.",
                    "For maximize metrics, target modest improvement above the stronger of current baseline and expected threshold.",
                    "For minimize metrics, preserve or tighten the current baseline unless the diagnosis shows the baseline is already violating a required bound.",
                    "Prioritize the metrics with the largest gaps and the most causal support.",
                    "Return only JSON.",
                ],
                "schema": {
                    "global_rationale": ["string"],
                    "component_guidance": {
                        "retrieval": {
                            "enable": True,
                            "max_trials": 4,
                            "objective_weights": {"context_recall": 0.4},
                            "target_thresholds": {"context_recall": 0.85},
                            "constraint_overrides": {"latency_ms_max": 1200},
                            "search_space_focus": {"top_k": [6, 8], "reranker": ["bge-reranker"]},
                            "notes": "string",
                        }
                    }
                },
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _llm_refine_plan(
        self,
        report: DiagnosisReport,
        result: EvaluationResult | None,
        objective_metric: str,
        strategy: SearchStrategy,
        budget: int,
        heuristic_plan: OptimizationPlan,
    ) -> Dict[str, Any]:
        if self.llm_callable is None:
            return {}
        try:
            raw = self.llm_callable(
                self._llm_reasoning_prompt(
                    report=report,
                    result=result,
                    objective_metric=objective_metric,
                    strategy=strategy,
                    budget=budget,
                    heuristic_plan=heuristic_plan,
                )
            )
            data = self._coerce_json(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _apply_focus(self, search_space: Dict[str, List[Any]], focus: Dict[str, Any]) -> Dict[str, List[Any]]:
        if not focus:
            return search_space
        updated = dict(search_space)
        for key, values in focus.items():
            if key in updated and isinstance(values, list) and values:
                allowed = [v for v in updated[key] if v in values]
                if allowed:
                    updated[key] = allowed
        return updated

    def _apply_llm_guidance(self, experiments: List[OptimizationExperiment], llm_guidance: Dict[str, Any]) -> tuple[List[OptimizationExperiment], List[str]]:
        rationale = list(llm_guidance.get("global_rationale", [])) if isinstance(llm_guidance, dict) else []
        component_guidance = llm_guidance.get("component_guidance", {}) if isinstance(llm_guidance, dict) else {}
        stage_to_component = {
            "corpus": "retrieval",
            "retrieval": "retrieval",
            "generation": "generation",
            "orchestration": "pipeline",
            "joint": "pipeline",
        }
        updated: List[OptimizationExperiment] = []
        for exp in experiments:
            guidance = component_guidance.get(stage_to_component.get(exp.stage, exp.target_component), {})
            if guidance.get("enable") is False:
                continue
            if isinstance(guidance.get("objective_weights"), dict) and guidance["objective_weights"]:
                total = sum(float(v) for v in guidance["objective_weights"].values() if isinstance(v, (int, float))) or 1.0
                exp.objectives = {k: round(float(v) / total, 4) for k, v in guidance["objective_weights"].items() if isinstance(v, (int, float))}
                exp.parameters["objective_weights"] = dict(exp.objectives)
            if isinstance(guidance.get("constraint_overrides"), dict):
                exp.constraints.update(guidance["constraint_overrides"])
            if isinstance(guidance.get("target_thresholds"), dict):
                exp.parameters["target_thresholds"] = guidance["target_thresholds"]
                baseline_metrics = exp.parameters.get("baseline_metrics", {})
                metric_directions = exp.parameters.get("metric_directions", {})
                exp.parameters["target_specs"] = {
                    k: self._target_semantics(k, baseline_metrics.get(k), float(v))
                    for k, v in guidance["target_thresholds"].items() if isinstance(v, (int, float))
                }
            if isinstance(guidance.get("search_space_focus"), dict):
                exp.search_space = self._apply_focus(exp.search_space, guidance["search_space_focus"])
            if isinstance(guidance.get("max_trials"), int) and guidance["max_trials"] > 0:
                exp.max_trials = guidance["max_trials"]
            if guidance.get("notes"):
                exp.notes = (exp.notes + "\nLLM planner notes: " + str(guidance["notes"])).strip()
            updated.append(exp)
        return updated, rationale

    def build_plan(
        self,
        report: DiagnosisReport,
        result: EvaluationResult | None = None,
        objective_metric: str = "answer_correctness",
        strategy: SearchStrategy = "bayesian",
        budget: int = 12,
    ) -> OptimizationPlan:
        experiments: List[OptimizationExperiment] = []
        rationale: List[str] = []
        candidates = set(report.optimization_candidates)
        lead_nodes = [s.node for s in report.causal_signals[:3]]
        constraints = self._constraints(result=result, report=report)

        def add_experiment(exp: OptimizationExperiment, why: str) -> None:
            experiments.append(exp)
            rationale.append(why)

        corpus_needed = "corpus_chunking_search" in candidates or "corpus_chunking_defect" in lead_nodes
        retrieval_needed = "autorag_pipeline_search" in candidates or any(n in lead_nodes for n in ["retrieval_recall_defect", "retrieval_precision_defect"])
        generation_needed = "dspy_prompt_optimization" in candidates or any(n in lead_nodes for n in ["grounding_defect", "citation_binding_defect", "context_packing_defect"])
        orchestration_needed = any(n in lead_nodes for n in ["distribution_shift", "judge_or_metric_instability"]) or bool(result and len(result.feedback_events) > 0)
        runtime_framework = ((result.metadata.get("runtime_framework") if result else None) or (result.metadata.get("framework") if result else None) or "")
        runtime_framework = runtime_framework.lower() if isinstance(runtime_framework, str) else ""

        if corpus_needed:
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("retrieval", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name="corpus-chunking-search",
                    tool="manual",
                    target_component="retrieval",
                    stage="corpus",
                    description="Optimize parsing, structure preservation, and chunking before retrieval tuning.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._corpus_space(),
                    search_strategy="pareto_evolutionary" if strategy == "pareto_evolutionary" else strategy,
                    max_trials=max(3, budget // 4),
                    notes="Trace-aware corpus search is used when chunking or ingestion defects appear likely. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                "A corpus-stage experiment was added because the diagnosis indicates missing evidence may originate in parsing or chunking.",
            )

        if retrieval_needed:
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("retrieval", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name="retrieval-pipeline-search",
                    tool="autorag",
                    target_component="retrieval",
                    stage="retrieval",
                    description="Optimize retrieval, reranking, and context packing for better evidence quality.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._retrieval_space(),
                    search_strategy=strategy,
                    max_trials=max(4, budget // 3),
                    notes="AutoRAG-style search space generated from retrieval diagnosis. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    depends_on=["corpus-chunking-search"] if corpus_needed else [],
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                "A retrieval-stage search was selected because the diagnosis indicates evidence miss or ranking noise.",
            )

        if generation_needed:
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("generation", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name="generator-prompt-optimization",
                    tool="dspy",
                    target_component="generation",
                    stage="generation",
                    description="Optimize grounded answer synthesis, citation behavior, and verification.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._generation_space(),
                    search_strategy=strategy,
                    max_trials=max(4, budget // 3),
                    notes="DSPy optimization surface generated from generation diagnosis. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    depends_on=["retrieval-pipeline-search"] if retrieval_needed else [],
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                "A generation-stage optimization was selected because the diagnosis indicates grounding, citation, or response formation issues.",
            )

        if orchestration_needed:
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("pipeline", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name="orchestration-policy-search",
                    tool="manual",
                    target_component="pipeline",
                    stage="orchestration",
                    description="Tune abstention, retry, and planner policies when drift, evaluation instability, or user feedback requires policy-level adaptation.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._orchestration_space(),
                    search_strategy="pareto_evolutionary",
                    max_trials=max(3, budget // 4),
                    notes="Policy-aware search is added when production feedback or evaluation instability is material. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    depends_on=[e.name for e in experiments if e.stage in {"retrieval", "generation"}],
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                "An orchestration-stage search was added because feedback and trace evidence suggest policy-level adaptation may be needed.",
            )

        if "joint_ablation_eval" in candidates or not experiments:
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("pipeline", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name="joint-pipeline-optimization",
                    tool="manual",
                    target_component="pipeline",
                    stage="joint",
                    description="Run multi-objective search over retrieval/generation operating profiles when component-level root cause is mixed.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._joint_space(),
                    search_strategy="pareto_evolutionary" if strategy == "pareto_evolutionary" else strategy,
                    max_trials=max(4, budget - sum(e.max_trials for e in experiments)),
                    notes="Joint search is used when the pipeline bottleneck is mixed or inconclusive. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    depends_on=[e.name for e in experiments],
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                "A joint search was added to resolve mixed retrieval and generation failures with direct end-to-end tradeoff tracking.",
            )

        if runtime_framework in {"langchain", "llamaindex"}:
            framework_tool = runtime_framework
            target_specs, baseline_metrics, metric_directions, target_thresholds, objective_weights = self._build_metric_plan("pipeline", objective_metric, result, report)
            add_experiment(
                OptimizationExperiment(
                    name=f"{runtime_framework}-stack-validation",
                    tool=framework_tool,
                    target_component="pipeline",
                    stage="joint",
                    description=f"Validate and optimize the end-to-end {runtime_framework} runtime with executable configs.",
                    parameters={
                        "diagnosis_summary": report.summary,
                        "planner": "ragdx",
                        "dataset_path": result.metadata.get("dataset_path", "examples/demo_dataset.jsonl") if result else "examples/demo_dataset.jsonl",
                        "pipeline_module": result.metadata.get("pipeline_module", f"examples.{runtime_framework}_pipeline:create_pipeline") if result else f"examples.{runtime_framework}_pipeline:create_pipeline",
                        "baseline_metrics": baseline_metrics,
                        "metric_directions": metric_directions,
                        "target_thresholds": target_thresholds,
                        "target_specs": target_specs,
                        "constraint_bounds": dict(constraints),
                        "objective_weights": objective_weights,
                    },
                    objectives=objective_weights,
                    search_space=self._stack_runtime_space(runtime_framework),
                    search_strategy=strategy,
                    max_trials=max(3, budget // 4),
                    notes=f"{runtime_framework} runtime validation is added because the evaluation metadata selected that framework. Objective weights are trade-off coefficients, not target metric values.",
                    constraints=dict(constraints),
                    depends_on=[e.name for e in experiments],
                    baseline_score=result.score(objective_metric) if result else None,
                ),
                f"An executable {runtime_framework} validation experiment was added so candidate configurations can be run directly against the chosen runtime framework.",
            )

        heuristic_plan = OptimizationPlan(objective_metric=objective_metric, experiments=experiments, rationale=rationale)
        llm_guidance = self._llm_refine_plan(report, result, objective_metric, strategy, budget, heuristic_plan)
        if llm_guidance:
            experiments, llm_rationale = self._apply_llm_guidance(experiments, llm_guidance)
            rationale.extend(llm_rationale)

        return OptimizationPlan(objective_metric=objective_metric, experiments=experiments, rationale=rationale)
