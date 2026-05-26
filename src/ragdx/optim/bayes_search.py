"""Lightweight Bayesian-optimisation search for RAG hyperparameters.

Uses scikit-learn's ``GaussianProcessRegressor`` (already a hard dep of
ragdx) so there's no extra install. Supports mixed numeric +
categorical params via one-hot encoding and picks the next trial by
Expected Improvement over the discrete candidate grid.

Why this module exists
----------------------
``run_autorag_grid_search`` in the demo used to enumerate every config
and evaluate them all. That's fine when the search space is 3 ``top_k``
values, but explodes as soon as you add chunk_size / chunk_overlap /
reranker. BO lets you cover the same conceptual space with ~30% of the
evaluations.

The implementation is intentionally compact (~150 lines) and self-
contained so the demo can use it without depending on Ax / BoTorch /
optuna. For a heavier, multi-objective BO loop see
``ragdx.optim.executor.OptimizationExecutor`` which already wraps the
sklearn GP and tracks Pareto fronts.

Example::

    from ragdx.optim.bayes_search import BayesianSearch

    search_space = {
        "chunk_size": [256, 512, 1024],         # numeric
        "top_k": [1, 3, 5, 7, 10],              # numeric
        "reranker": ["none", "bge"],            # categorical
    }
    bo = BayesianSearch(search_space, n_init=3, max_trials=10, seed=7)

    while bo.has_next():
        params = bo.next_params()
        score = my_eval_function(**params)      # your eval
        bo.report(params, score)

    best = bo.best_trial
    print(best.params, best.score)
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Trial:
    """One evaluated configuration. ``score`` is filled by ``report()``."""

    index: int
    params: dict[str, Any]
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def evaluated(self) -> bool:
        return self.score is not None


class BayesianSearch:
    """Discrete-grid BO with sklearn GP + Expected Improvement.

    The search space is a dict ``param_name -> list[allowed_value]``.
    Numeric (int/float) params are standardised; categorical (str/bool)
    params are one-hot encoded. The full cartesian product of allowed
    values forms the discrete candidate pool; BO picks the next
    candidate by maximising EI over still-unseen candidates.

    Parameters
    ----------
    search_space:
        ``{param_name: [values]}``. All combinations of those values are
        the candidate pool.
    n_init:
        Number of random points to evaluate before fitting the GP.
        Sklearn's GP needs at least 2 to start.
    max_trials:
        Total trials (init + BO steps). If max_trials >= len(grid), this
        degenerates to exhaustive grid search.
    seed:
        For reproducibility of init sampling + tie-break.
    objective_kind:
        ``"maximize"`` (default) or ``"minimize"``.
    """

    def __init__(
        self,
        search_space: dict[str, list[Any]],
        *,
        n_init: int = 3,
        max_trials: int = 10,
        seed: int = 7,
        objective_kind: str = "maximize",
    ):
        if not search_space:
            raise ValueError("search_space must be non-empty")
        self.search_space = {k: list(v) for k, v in search_space.items()}
        self.param_names = list(self.search_space)
        self.n_init = max(2, n_init)
        self.max_trials = max(self.n_init, max_trials)
        self.objective_kind = objective_kind
        self._rng = random.Random(seed)

        # Enumerate full candidate grid up-front; BO picks from this pool.
        product = list(itertools.product(*(self.search_space[k] for k in self.param_names)))
        self._candidates: list[dict[str, Any]] = [
            dict(zip(self.param_names, combo, strict=True)) for combo in product
        ]
        self._rng.shuffle(self._candidates)  # randomise init order

        self.max_trials = min(self.max_trials, len(self._candidates))
        self.trials: list[Trial] = []
        self._evaluated_keys: set[tuple] = set()  # candidate fingerprints
        self._pending_params: dict[str, Any] | None = None

        # Cache: column transformer fitted once on the full candidate grid.
        self._numeric_cols = [
            k for k, vs in self.search_space.items()
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs)
        ]
        self._categorical_cols = [k for k in self.param_names if k not in self._numeric_cols]
        self._encoder = self._build_encoder()

    # --------------------------------------------------------- iteration
    def has_next(self) -> bool:
        return len(self.trials) < self.max_trials

    def next_params(self) -> dict[str, Any]:
        """Return the next configuration to evaluate.

        Implementation: random pick for the first ``n_init`` trials, then
        EI-maximising pick from the remaining unseen candidates.
        """
        if not self.has_next():
            raise StopIteration("max_trials reached")
        if self._pending_params is not None:
            return dict(self._pending_params)

        if len(self.trials) < self.n_init:
            params = self._next_unseen_random()
        else:
            params = self._next_ei_pick()
        self._pending_params = params
        return dict(params)

    def report(self, params: dict[str, Any], score: float, extra: dict | None = None) -> Trial:
        """Record the score for the most recently suggested params."""
        if self._pending_params is None:
            raise RuntimeError("report() called before next_params()")
        if params != self._pending_params:
            raise ValueError(
                f"reported params do not match pending suggestion. "
                f"got={params} pending={self._pending_params}"
            )
        trial = Trial(
            index=len(self.trials),
            params=dict(params),
            score=float(score),
            extra=dict(extra or {}),
        )
        self.trials.append(trial)
        self._evaluated_keys.add(self._fingerprint(params))
        self._pending_params = None
        return trial

    # --------------------------------------------------------- best access
    @property
    def best_trial(self) -> Trial | None:
        evaluated = [t for t in self.trials if t.evaluated]
        if not evaluated:
            return None
        if self.objective_kind == "minimize":
            return min(evaluated, key=lambda t: t.score)
        return max(evaluated, key=lambda t: t.score)

    def history(self) -> list[dict]:
        """Compact serialisable list of all evaluated trials."""
        return [
            {"index": t.index, "params": t.params, "score": t.score, **t.extra}
            for t in self.trials
            if t.evaluated
        ]

    # ----------------------------------------------------- internal helpers
    def _fingerprint(self, params: dict[str, Any]) -> tuple:
        return tuple(params[k] for k in self.param_names)

    def _next_unseen_random(self) -> dict[str, Any]:
        for c in self._candidates:
            if self._fingerprint(c) not in self._evaluated_keys:
                return c
        raise RuntimeError("no unseen candidates left")

    def _build_encoder(self) -> ColumnTransformer | None:
        transformers = []
        if self._numeric_cols:
            transformers.append(("num", StandardScaler(), self._numeric_cols))
        if self._categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), self._categorical_cols))
        if not transformers:
            return None
        # Fit the encoder on the FULL candidate space so dimensions stay stable.
        encoder = ColumnTransformer(transformers=transformers, remainder="drop")
        encoder.fit(self._as_frame(self._candidates))
        return encoder

    def _as_frame(self, configs: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert configs to a stable-column DataFrame the encoder accepts."""
        return pd.DataFrame(configs, columns=self.param_names)

    def _vectorise(self, configs: list[dict[str, Any]]) -> np.ndarray:
        if self._encoder is None:
            return np.zeros((len(configs), 0))
        return np.asarray(self._encoder.transform(self._as_frame(configs)))

    def _next_ei_pick(self) -> dict[str, Any]:
        """Fit a GP on the trials so far, pick the next config by Expected
        Improvement over remaining candidates."""
        evaluated = [t for t in self.trials if t.evaluated]
        X_train = self._vectorise([t.params for t in evaluated])
        y_train = np.array([t.score for t in evaluated], dtype=float)
        if self.objective_kind == "minimize":
            y_train = -y_train

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
        gp = Pipeline([
            ("model", GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=7))
        ])
        gp.fit(X_train, y_train)

        unseen = [c for c in self._candidates if self._fingerprint(c) not in self._evaluated_keys]
        if not unseen:
            raise RuntimeError("no unseen candidates left")
        X_test = self._vectorise(unseen)
        mean, std = gp.named_steps["model"].predict(X_test, return_std=True)

        y_best = float(y_train.max())
        ei = _expected_improvement(mean, std, y_best)

        best_idx = int(np.argmax(ei))
        return unseen[best_idx]


def _expected_improvement(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """Standard EI for maximising y. Vectorised. Adds a tiny ``xi`` to
    encourage exploration when std is small but non-zero."""
    std = np.maximum(std, 1e-9)
    z = (mean - y_best - xi) / std
    # Normal CDF / PDF via erf to avoid scipy dependency.
    cdf = 0.5 * (1.0 + _erf_array(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mean - y_best - xi) * cdf + std * pdf


def _erf_array(x: np.ndarray) -> np.ndarray:
    """math.erf, vectorised."""
    return np.array([math.erf(float(v)) for v in np.asarray(x, dtype=float).ravel()]).reshape(np.shape(x))


__all__ = ["BayesianSearch", "Trial"]
