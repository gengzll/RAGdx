"""User-defined composite objectives for picking the winner among trials.

The optimisation steps in the demo (and any downstream caller of
``run_autorag_grid_search`` / ``DSPyAdapter.optimize``) used to pick the
best configuration by a single ``objective_metric``. That works for a
toy demo but quickly falls apart in real RAG eval where you typically
care about a *combination* of metrics — and want a hard ceiling on bad
ones (e.g. hallucination must stay below 0.2).

This module gives you that combination as a small, JSON-serialisable
spec the user can either define explicitly or pull from
:func:`default_objective` for a sensible starting point.

Example::

    from ragdx.optim.objectives import CompositeObjective, default_objective

    # explicit spec
    obj = CompositeObjective(
        metrics={
            "faithfulness": 1.0,
            "context_recall": 1.0,
            "context_precision": 0.5,
        },
        constraints={"hallucination": ("max", 0.2)},
        mode="weighted_sum",
    )

    # or take the default for the GT mode and tweak
    obj = default_objective("with_gt")          # sensible default
    obj = obj.with_overrides({"faithfulness": 2.0})  # bump a weight

    # score a candidate's metric dict
    result = obj.evaluate({"faithfulness": 0.9, "context_recall": 1.0, "hallucination": 0.1})
    # result -> {"score": 2.4, "feasible": True, "violations": []}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from ragdx.optim._gt_helpers import GTMode

Direction = Literal["min", "max"]
AggregateMode = Literal["weighted_sum", "weighted_mean"]


@dataclass(frozen=True)
class CompositeObjective:
    """A user-defined recipe for picking the winning trial.

    Attributes
    ----------
    metrics:
        ``metric_name -> weight``. Each metric value is multiplied by its
        weight, then summed (or averaged, see ``mode``). Metrics not in this
        dict are ignored. Metrics that are missing from a candidate's score
        dict — or that are NaN — contribute 0 (their weight is also skipped
        in ``weighted_mean`` mode so they don't deflate the average).
    constraints:
        ``metric_name -> (direction, threshold)`` where ``direction`` is
        ``"min"`` (value must be >= threshold) or ``"max"`` (value must be
        <= threshold). A candidate that violates any constraint is marked
        infeasible; ``best_among`` returns the feasible candidate with the
        highest score, falling back to the highest-scoring infeasible one
        if none are feasible.
    mode:
        ``"weighted_sum"`` (default) preserves the absolute weights;
        ``"weighted_mean"`` normalises by the sum of weights of metrics
        that actually contributed a value.
    """

    metrics: dict[str, float]
    constraints: dict[str, tuple[Direction, float]] = field(default_factory=dict)
    mode: AggregateMode = "weighted_sum"

    # ------------------------------------------------------------------ core
    def score(self, values: dict[str, float]) -> float:
        """Composite score for one candidate.

        Metrics missing from ``values`` or NaN contribute nothing. In
        ``weighted_mean`` mode, only metrics that actually contributed count
        toward the divisor — so missing metrics don't drag the mean down.
        """
        if not self.metrics:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for name, weight in self.metrics.items():
            value = values.get(name)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            total += value * float(weight)
            weight_sum += float(weight)
        if self.mode == "weighted_mean" and weight_sum > 0:
            return total / weight_sum
        return total

    def satisfies_constraints(self, values: dict[str, float]) -> tuple[bool, list[str]]:
        """Check that every constraint holds. Returns ``(ok, violations)``."""
        violations: list[str] = []
        for name, (direction, threshold) in self.constraints.items():
            v = values.get(name)
            if v is None:
                violations.append(f"{name}: missing from scores")
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                violations.append(f"{name}: non-numeric value {v!r}")
                continue
            if math.isnan(v):
                violations.append(f"{name}: NaN")
                continue
            if direction == "min" and v < threshold:
                violations.append(f"{name}={v:.4f} < min {threshold}")
            elif direction == "max" and v > threshold:
                violations.append(f"{name}={v:.4f} > max {threshold}")
        return len(violations) == 0, violations

    def evaluate(self, values: dict[str, float]) -> dict:
        """Bundle ``score`` + ``feasibility`` for nicer logging / display."""
        ok, violations = self.satisfies_constraints(values)
        return {
            "score": self.score(values),
            "feasible": ok,
            "violations": violations,
        }

    # ----------------------------------------------------- picker over trials
    def best_among(self, candidates: list[dict]) -> dict | None:
        """Pick the best candidate dict by score, preferring feasible ones.

        Each candidate must have a ``scores: dict[str, float]`` key.
        Returns the chosen candidate (with extra ``_composite``/``_feasible``
        fields attached for inspection) or ``None`` for an empty list.
        """
        if not candidates:
            return None
        annotated = []
        for c in candidates:
            scores = c.get("scores", {}) or {}
            ev = self.evaluate(scores)
            annotated.append({**c, "_composite": ev["score"], "_feasible": ev["feasible"]})

        feasible = [c for c in annotated if c["_feasible"]]
        pool = feasible if feasible else annotated
        return max(pool, key=lambda c: c["_composite"])

    # ---------------------------------------------------------------- utils
    def with_overrides(
        self,
        metrics: dict[str, float] | None = None,
        constraints: dict[str, tuple[Direction, float]] | None = None,
        mode: AggregateMode | None = None,
    ) -> CompositeObjective:
        """Return a copy with the given fields overridden."""
        return CompositeObjective(
            metrics={**self.metrics, **(metrics or {})},
            constraints={**self.constraints, **(constraints or {})},
            mode=mode or self.mode,
        )

    # ------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return {
            "metrics": dict(self.metrics),
            "constraints": {k: list(v) for k, v in self.constraints.items()},
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompositeObjective:
        cons_raw = data.get("constraints") or {}
        constraints: dict[str, tuple[Direction, float]] = {}
        for name, spec in cons_raw.items():
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                direction, threshold = spec
                if direction in ("min", "max"):
                    constraints[name] = (direction, float(threshold))
        return cls(
            metrics=dict(data.get("metrics") or {}),
            constraints=constraints,
            mode=data.get("mode", "weighted_sum"),
        )


# --------------------------------------------------------------- defaults
def default_objective(mode: GTMode) -> CompositeObjective:
    """Sensible default composite for each GT mode.

    * with-GT prioritises retrieval coverage (``context_recall`` weight 1.0)
      alongside grounding (``faithfulness`` 1.0). Other metrics get smaller
      weights as supporting signals.
    * no-GT prioritises grounding + answer relevance (both 1.0) since
      recall-against-truth isn't available. ``context_precision`` is a
      half-weight tiebreaker.

    These defaults are intentionally simple — adjust weights or add
    constraints with ``.with_overrides(...)`` for any real project.
    """
    if mode == "with_gt":
        return CompositeObjective(
            metrics={
                "faithfulness": 1.0,
                "context_recall": 1.0,
                "context_precision": 0.5,
                "answer_relevancy": 0.5,
            },
            mode="weighted_sum",
        )
    return CompositeObjective(
        metrics={
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 0.5,
        },
        mode="weighted_sum",
    )


__all__ = ["CompositeObjective", "default_objective"]
