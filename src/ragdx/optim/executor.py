"""
Optimization Execution Engine

Main Idea:
This module provides the execution engine for RAG pipeline optimization. It runs optimization experiments, manages trial execution, and integrates with various RAG frameworks and optimization algorithms.

Functionalities:
- Plan execution: Runs optimization plans with multiple experiments and trials
- Bayesian optimization: Uses Gaussian processes for efficient parameter space exploration
- Framework adapters: Integrates with DSPy, AutoRAG, LangChain, LlamaIndex
- Simulation mode: Simulates optimization for testing and demonstration
- Trial management: Tracks trial results, constraints, and convergence
- Checkpointing: Saves optimization state for resumability

Supported execution modes:
- simulate: Mock execution for testing
- prepare_only: Generate configuration files without running
- execute: Full execution with actual RAG framework integration

Usage:
Execute an optimization plan:

    from ragdx.optim.executor import OptimizationExecutor

    executor = OptimizationExecutor()
    session = executor.execute_plan(
        optimization_plan,
        baseline=evaluation_result,
        strategy="bayesian",
        mode="simulate"
    )

The executor returns OptimizationSession objects with trial results and optimization history.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ragdx.core.thresholds import LOWER_IS_BETTER
from ragdx.optim.autorag_adapter import AutoRAGAdapter
from ragdx.optim.dspy_adapter import DSPyAdapter
from ragdx.optim.heavy_bo import HeavyBOBackend
from ragdx.optim.langchain_adapter import LangChainAdapter
from ragdx.optim.llamaindex_adapter import LlamaIndexAdapter
from ragdx.schemas.models import (
    EvaluationResult,
    ExecutionMode,
    OptimizationExperiment,
    OptimizationPlan,
    OptimizationSession,
    OptimizationTrial,
    SearchStrategy,
)
from ragdx.storage.run_store import RunStore
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class MissingRunnerError(RuntimeError):
    """Raised when execute mode is requested but no runner is configured."""


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class OptimizationExecutor:
    """Run optimization plans.

    Parameters
    ----------
    root: persistence root. Defaults to ``.ragdx``.
    strict_execute: if True (default), running ``mode='execute'`` without a
        runner template configured fails the trial with a clear error instead
        of silently falling back to simulated scoring. Can also be controlled
        via env var ``RAGDX_STRICT_EXECUTE`` (default true). Tests that want
        the previous lenient behavior can set ``strict_execute=False`` or
        ``RAGDX_STRICT_EXECUTE=0``.
    in_process_runners: optional mapping ``{tool: callable}`` providing
        in-process trial runners. When present, takes precedence over the
        ``RAGDX_*_RUNNER_CMD`` subprocess templates. Each callable receives
        ``(experiment, trial, baseline)`` and must return ``Dict[str, float]``
        of objective/constraint metrics.
    """

    def __init__(
        self,
        root: str | Path = ".ragdx",
        *,
        strict_execute: bool | None = None,
        in_process_runners: Dict[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.dspy = DSPyAdapter()
        self.autorag = AutoRAGAdapter()
        self.langchain = LangChainAdapter()
        self.llamaindex = LlamaIndexAdapter()
        self.heavy_bo = HeavyBOBackend()
        self.store = RunStore(self.root)
        if strict_execute is None:
            strict_execute = _env_bool("RAGDX_STRICT_EXECUTE", default=True)
        self.strict_execute = strict_execute
        self.in_process_runners: Dict[str, Any] = dict(in_process_runners or {})

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _checkpoint(self, session: OptimizationSession) -> None:
        self.store.save_session(session)

    def _enumerate_candidates(self, search_space: Dict[str, List[Any]], limit: int) -> List[Dict[str, Any]]:
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        products = list(itertools.product(*values))
        candidates = [dict(zip(keys, combo)) for combo in products]
        if len(candidates) <= limit:
            return candidates
        rng = random.Random(7)
        rng.shuffle(candidates)
        return candidates[: max(limit * 6, limit)]

    def _hash_score(self, payload: Dict[str, Any]) -> float:
        text = json.dumps(payload, sort_keys=True)
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _simulate_objectives(self, baseline: EvaluationResult, experiment: OptimizationExperiment, params: Dict[str, Any]) -> Dict[str, float]:
        metric_names = set(experiment.objectives) | {k[:-4] for k in experiment.constraints if k.endswith("_max")} | {k[:-4] for k in experiment.constraints if k.endswith("_min")}
        metrics: Dict[str, float] = {}
        noise = self._hash_score({"exp": experiment.name, "params": params})
        for metric in metric_names:
            default_base = 0.2 if metric in LOWER_IS_BETTER else 0.65
            base = baseline.score(metric, default_base) or default_base
            gain = 0.0
            if experiment.tool == "autorag":
                if metric in {"context_recall", "answer_correctness"}:
                    gain += 0.03 if params.get("retriever") == "hybrid" else 0.01
                    gain += 0.02 if params.get("top_k", 0) in {6, 8} else 0.0
                if metric in {"context_precision", "citation_accuracy"}:
                    gain += 0.025 if params.get("reranker") != "none" else -0.005
                if metric == "latency_ms":
                    gain += 0.03 if params.get("top_k", 0) in {4, 6} else -0.01
                if metric == "cost_usd":
                    gain += 0.02 if params.get("top_k", 0) in {4, 6} else -0.005
                if metric == "hallucination":
                    gain += 0.02 if params.get("reranker") != "none" else -0.01
                if metric == "noise_sensitivity":
                    gain += 0.02 if params.get("context_ordering") != "retrieval_score" else -0.005
            elif experiment.tool == "dspy":
                if metric in {"faithfulness", "citation_accuracy", "answer_correctness"}:
                    gain += 0.03 if params.get("optimizer") == "MIPROv2" else 0.015
                    gain += 0.02 if params.get("prompt_style") in {"citation_first", "claim_then_evidence"} else 0.0
                    gain += 0.01 if params.get("verifier") == "claim_checker" else 0.0
                if metric == "response_relevancy":
                    gain += 0.015 if params.get("fewshot_count", 0) in {2, 4} else 0.0
                    gain += 0.01 if params.get("temperature", 0.0) <= 0.2 else -0.005
                if metric == "latency_ms":
                    gain += 0.02 if params.get("temperature", 0.0) <= 0.2 else -0.01
                if metric == "cost_usd":
                    gain += 0.02 if params.get("fewshot_count", 0) <= 2 else -0.01
                if metric in {"hallucination", "noise_sensitivity"}:
                    gain += 0.05 if params.get("verifier") == "claim_checker" else -0.005
                    gain += 0.015 if params.get("prompt_style") == "citation_first" else 0.0
            else:
                if metric == "answer_correctness":
                    gain += 0.025 if params.get("verifier") == "claim_checker" else 0.0
                    gain += 0.01 if params.get("retrieval_profile") == "balanced" else 0.0
                if metric in {"faithfulness", "context_recall", "citation_accuracy"}:
                    gain += 0.015 if params.get("generator_profile") == "citation_first" else 0.005
                if metric == "latency_ms":
                    gain += 0.02 if params.get("planner") == "single_step" else -0.005
                if metric == "cost_usd":
                    gain += 0.02 if params.get("verifier") == "none" else -0.005
                if metric in {"hallucination", "noise_sensitivity"}:
                    gain += 0.04 if params.get("abstention_policy") == "strict" else -0.004

            perturb = (noise - 0.5) * 0.02
            candidate = base + gain + perturb
            if metric in LOWER_IS_BETTER:
                candidate = max(0.0, min(1.0, base - gain - perturb))
            else:
                candidate = max(0.0, min(1.0, candidate))
            metrics[metric] = round(candidate, 4)
        return metrics

    def _evaluate_constraints(self, metrics: Dict[str, float], constraints: Dict[str, float]) -> tuple[bool, Dict[str, float], float]:
        violations: Dict[str, float] = {}
        penalty = 0.0
        for name, threshold in constraints.items():
            if name.endswith("_max"):
                metric = name[:-4]
                value = metrics.get(metric)
                if value is None:
                    continue
                excess = round(max(0.0, value - threshold), 4)
                if excess > 0:
                    violations[name] = excess
                    penalty += excess
            elif name.endswith("_min"):
                metric = name[:-4]
                value = metrics.get(metric)
                if value is None:
                    continue
                deficit = round(max(0.0, threshold - value), 4)
                if deficit > 0:
                    violations[name] = deficit
                    penalty += deficit
        return len(violations) == 0, violations, round(penalty, 4)

    def _utility(self, metrics: Dict[str, float], objectives: Dict[str, float], constraints: Dict[str, float] | None = None) -> tuple[float, bool, Dict[str, float], float]:
        score = 0.0
        total = 0.0
        for metric, weight in objectives.items():
            val = metrics.get(metric, 0.0)
            if metric in LOWER_IS_BETTER:
                val = 1.0 - val
            score += weight * val
            total += weight
        raw = round(score / max(total, 1e-9), 4)
        feasible, violations, penalty = self._evaluate_constraints(metrics, constraints or {})
        adjusted = raw if feasible else round(max(0.0, raw - min(0.8, penalty)), 4)
        return adjusted, feasible, violations, penalty

    def _dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        better_or_equal = True
        strictly_better = False
        for metric in set(a) | set(b):
            av = a.get(metric, 0.0)
            bv = b.get(metric, 0.0)
            if metric in LOWER_IS_BETTER:
                if av > bv:
                    better_or_equal = False
                if av < bv:
                    strictly_better = True
            else:
                if av < bv:
                    better_or_equal = False
                if av > bv:
                    strictly_better = True
        return better_or_equal and strictly_better

    def _pareto_front(self, trials: List[OptimizationTrial], feasible_only: bool = True) -> List[str]:
        front: List[str] = []
        done_trials = [t for t in trials if t.objective_scores and ((t.feasible is True) if feasible_only else True)]
        for trial in done_trials:
            dominated = False
            for other in done_trials:
                if other.trial_id == trial.trial_id:
                    continue
                if self._dominates(other.objective_scores, trial.objective_scores):
                    dominated = True
                    break
            if not dominated:
                front.append(trial.trial_id)
        return front

    def _objective_reference_point(self, experiment: OptimizationExperiment) -> Dict[str, float]:
        ref = {}
        for metric in experiment.objectives:
            ref[metric] = 1.05 if metric in LOWER_IS_BETTER else -0.05
        return ref

    def _to_max_vector(self, metrics: Dict[str, float], objective_names: List[str]) -> List[float]:
        vec = []
        for metric in objective_names:
            v = float(metrics.get(metric, 0.0))
            vec.append(1.0 - v if metric in LOWER_IS_BETTER else v)
        return vec

    def _hypervolume_2d(self, points: List[List[float]], ref: List[float]) -> float:
        if not points:
            return 0.0
        pts = sorted(points, key=lambda p: p[0], reverse=True)
        hv = 0.0
        best_y = ref[1]
        for x, y in pts:
            if y <= best_y:
                continue
            hv += max(0.0, x - ref[0]) * max(0.0, y - best_y)
            best_y = y
        return round(hv, 6)

    def _hypervolume(self, trials: List[OptimizationTrial], experiment: OptimizationExperiment, feasible_only: bool = True) -> float:
        objective_names = list(experiment.objectives.keys())[:2]
        if len(objective_names) < 2:
            return 0.0
        points = [self._to_max_vector(t.objective_scores, objective_names) for t in trials if t.objective_scores and ((t.feasible is True) if feasible_only else True)]
        ref = [0.0, 0.0]
        return self._hypervolume_2d(points, ref)

    def _build_surrogate_models(self, trials: List[OptimizationTrial], experiment: OptimizationExperiment):
        if len(trials) < 4:
            return None
        feature_rows = [t.parameters for t in trials if t.objective_scores]
        if len(feature_rows) < 4:
            return None
        keys = list(feature_rows[0].keys())
        num_keys = [k for k in keys if isinstance(feature_rows[0][k], (int, float, bool))]
        cat_keys = [k for k in keys if k not in num_keys]

        def make_pipeline():
            transformers = []
            if num_keys:
                transformers.append(("num", Pipeline([("scale", StandardScaler())]), num_keys))
            if cat_keys:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_keys))
            pre = ColumnTransformer(transformers=transformers, remainder="drop")
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=7)
            return Pipeline([("pre", pre), ("gp", model)])

        models = {}
        X = feature_rows
        for metric in set(experiment.objectives) | {k[:-4] for k in experiment.constraints if k.endswith("_max")} | {k[:-4] for k in experiment.constraints if k.endswith("_min")}:
            y = [t.objective_scores.get(metric) for t in trials if t.objective_scores and metric in t.objective_scores]
            X_m = [t.parameters for t in trials if t.objective_scores and metric in t.objective_scores]
            if len(y) < 4:
                continue
            pipe = make_pipeline()
            pipe.fit(X_m, y)
            models[metric] = pipe
        return models if models else None

    def _feasibility_probability(self, candidate: Dict[str, Any], models: Dict[str, Any], constraints: Dict[str, float]) -> float:
        if not constraints:
            return 1.0
        p = 1.0
        for name, threshold in constraints.items():
            metric = name[:-4]
            model = models.get(metric)
            if model is None:
                continue
            mean_pred, std_pred = model.predict([candidate], return_std=True)
            mu = float(mean_pred[0])
            sigma = max(float(std_pred[0]), 1e-6)
            if name.endswith("_max"):
                p *= _normal_cdf((threshold - mu) / sigma)
            else:
                p *= 1.0 - _normal_cdf((threshold - mu) / sigma)
        return max(0.0, min(1.0, p))

    def _predicted_objectives(self, candidate: Dict[str, Any], models: Dict[str, Any], objective_names: List[str]) -> tuple[Dict[str, float], Dict[str, float]]:
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for metric in objective_names:
            model = models.get(metric)
            if model is None:
                means[metric] = 0.5 if metric not in LOWER_IS_BETTER else 0.2
                stds[metric] = 0.1
                continue
            mean_pred, std_pred = model.predict([candidate], return_std=True)
            means[metric] = float(mean_pred[0])
            stds[metric] = max(float(std_pred[0]), 1e-6)
        return means, stds

    def _expected_hv_improvement(self, candidate: Dict[str, Any], trials: List[OptimizationTrial], experiment: OptimizationExperiment, models: Dict[str, Any]) -> float:
        objective_names = list(experiment.objectives.keys())[:2]
        if len(objective_names) < 2:
            return 0.0
        current_hv = self._hypervolume(trials, experiment, feasible_only=True)
        means, stds = self._predicted_objectives(candidate, models, objective_names)
        optimistic = {}
        for metric in objective_names:
            if metric in LOWER_IS_BETTER:
                optimistic[metric] = max(0.0, means[metric] - 0.5 * stds[metric])
            else:
                optimistic[metric] = min(1.0, means[metric] + 0.5 * stds[metric])
        pseudo = OptimizationTrial(trial_id="pseudo", experiment_name=experiment.name, tool=experiment.tool, strategy=experiment.search_strategy, objective_scores=optimistic, feasible=True)
        new_hv = self._hypervolume(trials + [pseudo], experiment, feasible_only=True)
        return max(0.0, new_hv - current_hv)

    def _sample_bayesian(self, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial], experiment: OptimizationExperiment) -> Dict[str, Any]:
        observed = {json.dumps(t.parameters, sort_keys=True): t for t in trials}
        remaining = [c for c in candidates if json.dumps(c, sort_keys=True) not in observed]
        if not remaining:
            return candidates[0]

        backend = os.environ.get("RAGDX_BO_BACKEND", "internal").lower()
        if backend != "internal":
            suggestion = self.heavy_bo.suggest(backend=backend, experiment=experiment, candidates=remaining, trials=trials)
            if suggestion is not None:
                key = json.dumps(suggestion.parameters, sort_keys=True)
                if key not in observed:
                    return suggestion.parameters

        if len(trials) < 3:
            return remaining[0]
        models = self._build_surrogate_models(trials, experiment)
        if not models:
            return remaining[0]
        scored = []
        for cand in remaining:
            p_feas = self._feasibility_probability(cand, models, experiment.constraints)
            ehvi = self._expected_hv_improvement(cand, [t for t in trials if t.objective_scores], experiment, models)
            means, stds = self._predicted_objectives(cand, models, list(experiment.objectives.keys()))
            exploration = sum(stds.values()) / max(len(stds), 1)
            scalar = 0.0
            tot = 0.0
            for metric, weight in experiment.objectives.items():
                val = means[metric]
                val = 1.0 - val if metric in LOWER_IS_BETTER else val
                scalar += weight * val
                tot += weight
            scalar /= max(tot, 1e-9)
            score = 0.55 * p_feas * ehvi + 0.30 * p_feas * scalar + 0.15 * exploration
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _sample_pareto(self, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial], experiment: OptimizationExperiment) -> Dict[str, Any]:
        observed = {json.dumps(t.parameters, sort_keys=True): t for t in trials}
        remaining = [c for c in candidates if json.dumps(c, sort_keys=True) not in observed]
        if not remaining:
            return candidates[0]
        if len(trials) < 2:
            return remaining[0]
        front_ids = set(self._pareto_front(trials, feasible_only=True) or self._pareto_front(trials, feasible_only=False))
        front_trials = [t for t in trials if t.trial_id in front_ids]
        if not front_trials:
            return remaining[0]
        parent = random.Random(len(trials)).choice(front_trials)
        child = dict(parent.parameters)
        rng = random.Random(len(trials) * 17)
        mutable_keys = list(child.keys())
        for key in rng.sample(mutable_keys, k=max(1, min(2, len(mutable_keys)))):
            options = list({cand[key] for cand in candidates})
            child[key] = rng.choice(options)
        child_key = json.dumps(child, sort_keys=True)
        if child_key not in observed:
            return child
        return remaining[0]

    def _write_config(self, session_id: str, experiment: OptimizationExperiment, parameters: Dict[str, Any], trial_id: str) -> str:
        out_dir = self.root / "optimization" / session_id / "configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{trial_id}_{experiment.tool}.yaml"
        if experiment.tool == "autorag":
            payload = self.autorag.run(experiment, parameters).payload
        elif experiment.tool == "dspy":
            payload = self.dspy.run(experiment, parameters).payload
        elif experiment.tool == "langchain":
            payload = self.langchain.run(experiment, parameters).payload
        elif experiment.tool == "llamaindex":
            payload = self.llamaindex.run(experiment, parameters).payload
        else:
            payload = {"framework": "ragdx-manual", "objective_metric": "answer_correctness", "objectives": experiment.objectives, "search_parameters": parameters}
        payload["constraints"] = experiment.constraints
        payload["optimization_metadata"] = {"stage": experiment.stage, "search_strategy": experiment.search_strategy, "max_trials": experiment.max_trials}
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return str(path)

    def _runner_template(self, tool: str) -> str | None:
        mapping = {
            "dspy": os.environ.get("RAGDX_DSPY_RUNNER_CMD"),
            "autorag": os.environ.get("RAGDX_AUTORAG_RUNNER_CMD"),
            "langchain": os.environ.get("RAGDX_LANGCHAIN_RUNNER_CMD"),
            "llamaindex": os.environ.get("RAGDX_LLAMAINDEX_RUNNER_CMD"),
            "manual": os.environ.get("RAGDX_MANUAL_RUNNER_CMD"),
        }
        return mapping.get(tool)

    def _parse_output_metrics(self, output_path: Path, experiment: OptimizationExperiment) -> Dict[str, float]:
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output file not found: {output_path}")
        data = json.loads(output_path.read_text(encoding="utf-8"))
        if "objective_scores" in data and isinstance(data["objective_scores"], dict):
            metrics = {k: float(v) for k, v in data["objective_scores"].items()}
        else:
            metrics: Dict[str, float] = {}
            for bucket in ("retrieval", "generation", "e2e"):
                if isinstance(data.get(bucket), dict):
                    metrics.update({k: float(v) for k, v in data[bucket].items()})
            if not metrics and isinstance(data.get("metrics"), dict):
                metrics = {k: float(v) for k, v in data["metrics"].items()}
        needed = set(experiment.objectives) | {k[:-4] for k in experiment.constraints if k.endswith("_max")} | {k[:-4] for k in experiment.constraints if k.endswith("_min")}
        selected = {k: metrics[k] for k in needed if k in metrics}
        if not selected:
            raise ValueError(f"No objective or constraint metrics found in output file {output_path}")
        return selected

    def _apply_trial_metrics(
        self,
        trial: OptimizationTrial,
        experiment: OptimizationExperiment,
        metrics: Dict[str, float],
        note: str,
        *,
        simulated: bool = False,
    ) -> None:
        trial.objective_scores = metrics
        trial.utility, trial.feasible, trial.constraint_violations, trial.feasibility_penalty = self._utility(metrics, experiment.objectives, experiment.constraints)
        trial.status = "done"
        trial.simulated = simulated
        trial.logs.append(note)
        if not trial.feasible:
            trial.logs.append(f"Constraint violations: {trial.constraint_violations}")

    def _execute_external_trial(self, session_id: str, experiment: OptimizationExperiment, trial: OptimizationTrial, baseline: EvaluationResult) -> None:
        # 1. Prefer in-process runner if one is registered for this tool.
        runner = self.in_process_runners.get(experiment.tool)
        if runner is not None:
            logger.info(
                "Running trial in-process tool=%s trial=%s session=%s",
                experiment.tool, trial.trial_id, session_id,
            )
            try:
                metrics = runner(experiment=experiment, trial=trial, baseline=baseline)
            except Exception as exc:
                trial.status = "failed"
                trial.logs.append(f"In-process runner raised: {exc!r}")
                trial.notes = f"In-process runner error: {exc}"
                logger.exception("In-process runner failed", extra={"trial_id": trial.trial_id})
                return
            if not isinstance(metrics, dict):
                trial.status = "failed"
                trial.logs.append(
                    f"In-process runner returned {type(metrics).__name__}; expected Dict[str, float]."
                )
                return
            self._apply_trial_metrics(
                trial, experiment, {k: float(v) for k, v in metrics.items()},
                f"In-process runner scored: {metrics}",
            )
            return

        # 2. Fall back to subprocess runner template.
        template = self._runner_template(experiment.tool)
        if not template:
            # Strict mode (default) refuses to silently fake results.
            legacy_fallback = _env_bool(
                "RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER", default=False
            )
            if self.strict_execute and not legacy_fallback:
                msg = (
                    f"execute mode requested for tool={experiment.tool!r} but no runner is configured. "
                    f"Set environment variable RAGDX_{experiment.tool.upper()}_RUNNER_CMD, or register an "
                    f"in-process runner via OptimizationExecutor(in_process_runners=...), or run with "
                    f"mode='simulate' if you want simulated scores."
                )
                trial.status = "failed"
                trial.logs.append(msg)
                trial.notes = "Missing runner command template (strict execute mode)."
                logger.error(msg, extra={"trial_id": trial.trial_id, "tool": experiment.tool})
                return
            # Lenient legacy path — only entered when the user explicitly opts in.
            logger.warning(
                "Lenient fallback: simulating tool=%s because no runner is configured. "
                "Results are NOT from a real RAG pipeline.",
                experiment.tool,
                extra={"trial_id": trial.trial_id, "tool": experiment.tool},
            )
            metrics = self._simulate_objectives(baseline, experiment, trial.parameters)
            self._apply_trial_metrics(
                trial, experiment, metrics,
                f"No external runner configured for tool={experiment.tool}; "
                "RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER=1 forced simulated scoring.",
                simulated=True,
            )
            trial.logs.append("WARNING: trial scored by simulator, not by a real runner.")
            return

        trial_dir = self.root / "optimization" / session_id / "outputs"
        trial_dir.mkdir(parents=True, exist_ok=True)
        output_path = trial_dir / f"{trial.trial_id}_metrics.json"
        log_path = trial_dir / f"{trial.trial_id}.log"
        # ``shlex.split`` runs in POSIX mode below, which treats backslashes
        # as escape characters and silently corrupts Windows paths
        # (e.g. ``.ragdx\out`` -> ``.ragdxout``). Forward slashes are
        # accepted by every stdlib path API on Windows, so we normalize
        # any executor-controlled path before substitution.
        def _norm(p: str) -> str:
            return str(p).replace("\\", "/")
        try:
            command = template.format(
                config=_norm(trial.config_path),
                output=_norm(output_path),
                workdir=_norm(trial_dir),
                trial_id=trial.trial_id, session_id=session_id, tool=experiment.tool,
            )
        except KeyError as exc:
            trial.status = "failed"
            trial.notes = f"Runner template references unknown placeholder {exc}."
            trial.logs.append(trial.notes)
            logger.error(trial.notes, extra={"trial_id": trial.trial_id})
            return
        trial.runner_command = command
        trial.output_path = str(output_path)
        trial.log_path = str(log_path)
        trial.logs.append(f"Launching external runner: {command}")
        logger.info(
            "Launching runner tool=%s trial=%s cmd=%s",
            experiment.tool, trial.trial_id, command,
            extra={"trial_id": trial.trial_id, "tool": experiment.tool, "session_id": session_id},
        )
        timeout = float(os.environ.get("RAGDX_RUNNER_TIMEOUT_SEC", "0") or 0) or None
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    shlex.split(command), stdout=lf, stderr=subprocess.STDOUT,
                    text=True, timeout=timeout,
                )
        except subprocess.TimeoutExpired as exc:
            trial.status = "failed"
            trial.notes = f"Runner exceeded RAGDX_RUNNER_TIMEOUT_SEC={timeout}s."
            trial.logs.append(trial.notes)
            logger.error("Runner timeout: %s", exc, extra={"trial_id": trial.trial_id})
            return
        except FileNotFoundError as exc:
            trial.status = "failed"
            trial.notes = f"Runner executable not found: {exc}"
            trial.logs.append(trial.notes)
            logger.error(trial.notes, extra={"trial_id": trial.trial_id})
            return
        trial.return_code = proc.returncode
        if proc.returncode != 0:
            trial.status = "failed"
            trial.logs.append(f"External runner failed with return code {proc.returncode}; see {log_path}.")
            logger.error(
                "Runner non-zero exit: tool=%s trial=%s rc=%s",
                experiment.tool, trial.trial_id, proc.returncode,
                extra={"trial_id": trial.trial_id, "tool": experiment.tool},
            )
            return
        try:
            metrics = self._parse_output_metrics(output_path, experiment)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            trial.status = "failed"
            trial.notes = f"Runner output unreadable: {exc}"
            trial.logs.append(trial.notes)
            logger.error(trial.notes, extra={"trial_id": trial.trial_id})
            return
        self._apply_trial_metrics(trial, experiment, metrics, f"Loaded objective scores from {output_path}: {metrics}")

    def _update_front_and_best(self, session: OptimizationSession, experiment: OptimizationExperiment | None = None) -> None:
        front = self._pareto_front(session.trials, feasible_only=False)
        feasible_front = self._pareto_front(session.trials, feasible_only=True)
        session.pareto_front_ids = front
        session.feasible_pareto_front_ids = feasible_front
        for t in session.trials:
            t.pareto_front = t.trial_id in front
        feasible_completed = [t for t in session.trials if t.utility is not None and t.feasible]
        all_completed = [t for t in session.trials if t.utility is not None]
        if feasible_completed:
            session.best_trial_id = max(feasible_completed, key=lambda t: t.utility or 0.0).trial_id
        elif all_completed:
            session.best_trial_id = max(all_completed, key=lambda t: t.utility or 0.0).trial_id
        if experiment is not None:
            session.hypervolume = self._hypervolume(session.trials, experiment, feasible_only=False)
            session.feasible_hypervolume = self._hypervolume(session.trials, experiment, feasible_only=True)

    def execute_plan(self, plan: OptimizationPlan, baseline: EvaluationResult, strategy: SearchStrategy = "bayesian", mode: ExecutionMode = "simulate", run_id: str | None = None) -> OptimizationSession:
        session_id = uuid4().hex[:12]
        total_trials = sum(exp.max_trials for exp in plan.experiments)
        initial_status = "running" if mode in {"simulate", "execute"} else "prepared"
        session = OptimizationSession(session_id=session_id, created_at=self._timestamp(), run_id=run_id, strategy=strategy, mode=mode, status=initial_status, plan=plan, total_trials=total_trials)
        logger.info(
            "Starting optimization session=%s mode=%s strategy=%s total_trials=%s experiments=%s strict_execute=%s",
            session_id, mode, strategy, total_trials, len(plan.experiments), self.strict_execute,
            extra={"session_id": session_id, "run_id": run_id},
        )
        self._checkpoint(session)

        all_trials: List[OptimizationTrial] = []
        completed_experiment_names: set[str] = set()
        for exp in plan.experiments:
            unmet = [dep for dep in exp.depends_on if dep not in completed_experiment_names]
            if unmet:
                raise ValueError(f"Experiment {exp.name} cannot start because dependencies are not completed: {unmet}")
            session.current_experiment = exp.name
            self._checkpoint(session)
            candidates = self._enumerate_candidates(exp.search_space, exp.max_trials)
            exp_trials: List[OptimizationTrial] = []
            for _ in range(exp.max_trials):
                params = self._sample_bayesian(candidates, exp_trials, exp) if exp.search_strategy == "bayesian" else self._sample_pareto(candidates, exp_trials, exp)
                trial_id = uuid4().hex[:10]
                trial = OptimizationTrial(trial_id=trial_id, experiment_name=exp.name, tool=exp.tool, strategy=exp.search_strategy, status="prepared" if mode == "prepare_only" else "running", parameters=params, started_at=self._timestamp())
                trial.config_path = self._write_config(session_id, exp, params, trial_id)
                trial.logs.append(f"Configuration written to {trial.config_path}")
                all_trials.append(trial)
                exp_trials.append(trial)
                session.trials = all_trials
                self._checkpoint(session)
                try:
                    if mode == "simulate":
                        metrics = self._simulate_objectives(baseline, exp, params)
                        self._apply_trial_metrics(
                            trial, exp, metrics,
                            f"Simulated objective scores: {metrics}",
                            simulated=True,
                        )
                    elif mode == "prepare_only":
                        trial.logs.append("Prepared configuration only. External execution is not started by ragdx in this mode.")
                    else:
                        self._execute_external_trial(session_id, exp, trial, baseline)
                except Exception as exc:
                    trial.status = "failed"
                    trial.notes = str(exc)
                    trial.logs.append(f"Execution error: {exc}")
                    logger.exception(
                        "Trial errored tool=%s trial=%s exp=%s",
                        exp.tool, trial.trial_id, exp.name,
                        extra={"trial_id": trial.trial_id, "tool": exp.tool, "session_id": session_id},
                    )
                finally:
                    trial.completed_at = self._timestamp()
                    session.completed_trials += 1
                    self._update_front_and_best(session, exp)
                    self._checkpoint(session)
            exp.status = "done" if mode in {"simulate", "execute"} else "planned"
            completed_experiment_names.add(exp.name)

        # If every trial failed in execute mode, mark the session itself failed.
        if mode == "execute" and session.trials and all(t.status == "failed" for t in session.trials):
            session.status = "failed"
        else:
            session.status = "completed" if mode in {"simulate", "execute"} else "prepared"
        session.current_experiment = None
        self._checkpoint(session)
        logger.info(
            "Session finished session=%s status=%s completed=%s/%s best=%s",
            session_id, session.status, session.completed_trials,
            session.total_trials, session.best_trial_id,
            extra={"session_id": session_id},
        )
        return session
