"""
Heavy Bayesian Optimization Backend

Main Idea:
This module provides a Bayesian optimization backend for efficient parameter space exploration in RAG pipeline optimization. It integrates with the Ax optimization library for sophisticated hyperparameter tuning.

Functionalities:
- Parameter suggestion: Uses Bayesian optimization to suggest promising parameter combinations
- Ax integration: Leverages AxClient for experiment management and optimization
- Multi-objective support: Handles multiple optimization objectives with constraints
- State management: Maintains optimization state across trials
- Fallback handling: Gracefully handles missing dependencies

Supported backends:
- Ax: Facebook's Ax optimization framework
- Internal: Simple fallback for basic parameter sampling

Usage:
Used by the optimization executor for Bayesian optimization:

    from ragdx.optim.heavy_bo import HeavyBOBackend

    backend = HeavyBOBackend()
    suggestion = backend.suggest("ax", experiment, candidates, trials)

The backend provides intelligent parameter suggestions to accelerate optimization convergence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ragdx.core.thresholds import LOWER_IS_BETTER
from ragdx.schemas.models import OptimizationExperiment, OptimizationTrial


@dataclass
class HeavyBOSuggestion:
    parameters: Dict[str, Any]
    trial_key: Optional[str] = None
    backend: str = "internal"


class HeavyBOBackend:
    def __init__(self):
        self._state: Dict[str, Any] = {}

    def suggest(self, backend: str, experiment: OptimizationExperiment, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial]) -> HeavyBOSuggestion | None:
        if backend == "ax":
            return self._suggest_ax(experiment, candidates, trials)
        return None

    def _suggest_ax(self, experiment: OptimizationExperiment, candidates: List[Dict[str, Any]], trials: List[OptimizationTrial]) -> HeavyBOSuggestion | None:
        try:
            from ax.service.ax_client import AxClient
            from ax.service.utils.instantiation import ObjectiveProperties
        except Exception:
            return None

        key = experiment.name
        state = self._state.get(key)
        if state is None:
            client = AxClient(random_seed=7)
            parameters = []
            for name, values in experiment.search_space.items():
                values = list(values)
                entry = {"name": name, "type": "choice", "values": values, "is_ordered": False}
                if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    entry["value_type"] = "float" if any(isinstance(v, float) for v in values) else "int"
                    entry["is_ordered"] = True
                parameters.append(entry)
            objectives = {m: ObjectiveProperties(minimize=(m in LOWER_IS_BETTER), threshold=None) for m in experiment.objectives.keys()}
            constraints = [f"{k[:-4]} <= {v}" if k.endswith("_max") else f"{k[:-4]} >= {v}" for k, v in experiment.constraints.items()]
            try:
                client.create_experiment(name=f"ragdx_{experiment.name}", parameters=parameters, objectives=objectives, outcome_constraints=constraints)
            except TypeError:
                return None
            state = {"client": client, "seen": set()}
            self._state[key] = state

        client = state["client"]
        for t in trials:
            ax_key = json.dumps(t.parameters, sort_keys=True)
            if t.objective_scores and ax_key not in state["seen"]:
                try:
                    _, trial_index = client.attach_trial(parameters=t.parameters)
                    raw_data = {m: (float(v), 0.0) for m, v in t.objective_scores.items()}
                    client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                    state["seen"].add(ax_key)
                except Exception:
                    pass
        try:
            params, trial_index = client.get_next_trial()
            return HeavyBOSuggestion(parameters=params, trial_key=str(trial_index), backend="ax")
        except Exception:
            return None
