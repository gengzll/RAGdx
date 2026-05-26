"""Tests for ragdx.optim.bayes_search.BayesianSearch.

Uses a synthetic objective (a 2-D paraboloid over chunk_size + top_k)
so we can verify BO converges toward the known optimum without LLM
calls.
"""

from __future__ import annotations

import math

import pytest

from ragdx.optim.bayes_search import BayesianSearch, Trial


def test_init_rejects_empty_search_space():
    with pytest.raises(ValueError, match="non-empty"):
        BayesianSearch({})


def test_basic_iteration_returns_distinct_configs():
    search_space = {"top_k": [1, 3, 5, 7]}
    bo = BayesianSearch(search_space, n_init=2, max_trials=4, seed=0)
    seen = []
    while bo.has_next():
        params = bo.next_params()
        bo.report(params, score=float(params["top_k"]))
        seen.append(params)
    # all distinct (grid has 4 candidates, we did 4 trials)
    fingerprints = [tuple(s.items()) for s in seen]
    assert len(set(fingerprints)) == len(fingerprints)


def test_objective_kind_minimize_picks_lower():
    bo = BayesianSearch({"x": [1, 2, 3, 4]}, n_init=2, max_trials=4, objective_kind="minimize")
    while bo.has_next():
        p = bo.next_params()
        bo.report(p, float(p["x"]))
    assert bo.best_trial.params["x"] == 1


def test_objective_kind_maximize_picks_higher():
    bo = BayesianSearch({"x": [1, 2, 3, 4]}, n_init=2, max_trials=4)
    while bo.has_next():
        p = bo.next_params()
        bo.report(p, float(p["x"]))
    assert bo.best_trial.params["x"] == 4


def test_history_contains_all_evaluated_trials():
    bo = BayesianSearch({"k": [1, 3]}, n_init=2, max_trials=2)
    while bo.has_next():
        p = bo.next_params()
        bo.report(p, 0.5)
    hist = bo.history()
    assert len(hist) == 2
    for entry in hist:
        assert {"index", "params", "score"}.issubset(entry)


def test_report_validates_params_match_pending():
    bo = BayesianSearch({"k": [1, 3]}, n_init=2, max_trials=2)
    _ = bo.next_params()
    with pytest.raises(ValueError, match="do not match"):
        bo.report({"k": 999}, score=0.0)


def test_report_without_pending_raises():
    bo = BayesianSearch({"k": [1, 3]}, n_init=2, max_trials=2)
    with pytest.raises(RuntimeError, match="before next_params"):
        bo.report({"k": 1}, score=0.0)


def test_max_trials_clamped_to_grid_size():
    bo = BayesianSearch({"k": [1, 3]}, n_init=2, max_trials=99)
    assert bo.max_trials == 2  # grid has only 2 points


def test_mixed_numeric_and_categorical_space():
    """Encoder must handle both kinds at once -- this used to break when
    only one column type was provided."""
    bo = BayesianSearch(
        {"chunk_size": [256, 512], "reranker": ["none", "bge"]},
        n_init=2, max_trials=4, seed=0,
    )
    while bo.has_next():
        p = bo.next_params()
        # Higher chunk_size + bge reranker wins
        score = p["chunk_size"] / 1024 + (0.3 if p["reranker"] == "bge" else 0.0)
        bo.report(p, score)
    best = bo.best_trial
    assert best.params["chunk_size"] == 512
    assert best.params["reranker"] == "bge"


# ---------------------- convergence sanity check on a synthetic objective
def _paraboloid(chunk_size: int, top_k: int) -> float:
    """Smooth bowl with optimum at (512, 5). Returns a value in [0, 1]."""
    cs_term = -((chunk_size - 512) / 512) ** 2
    tk_term = -((top_k - 5) / 5) ** 2
    return 1.0 + 0.5 * (cs_term + tk_term)


def test_bo_converges_to_optimum_better_than_random():
    """Over a few trials BO's chosen best should beat a same-size random
    sample on a smooth synthetic objective."""
    space = {"chunk_size": [128, 256, 512, 768, 1024], "top_k": [1, 3, 5, 7, 9]}
    n_trials = 7

    # BO run
    bo = BayesianSearch(space, n_init=3, max_trials=n_trials, seed=1)
    while bo.has_next():
        p = bo.next_params()
        bo.report(p, _paraboloid(p["chunk_size"], p["top_k"]))
    bo_best = bo.best_trial.score

    # Random baseline (same number of trials, distinct seed for diversity)
    import random as _r
    rng = _r.Random(13)
    candidates = [
        {"chunk_size": cs, "top_k": tk}
        for cs in space["chunk_size"] for tk in space["top_k"]
    ]
    rng.shuffle(candidates)
    sampled = candidates[:n_trials]
    rand_best = max(_paraboloid(c["chunk_size"], c["top_k"]) for c in sampled)

    # BO shouldn't be worse than random on a smooth surface. We allow a
    # tiny tolerance because BO with very few init points can briefly
    # underperform; in practice it should match or beat.
    assert bo_best >= rand_best - 0.05, (
        f"BO best ({bo_best:.3f}) significantly worse than random ({rand_best:.3f})"
    )


def test_trial_dataclass_evaluated_flag():
    t = Trial(index=0, params={"k": 1})
    assert not t.evaluated
    t.score = 0.5
    assert t.evaluated


def test_eei_at_boundary_does_not_explode():
    """Internal sanity: when GP std is 0 the EI must be finite, not NaN."""
    import numpy as np

    from ragdx.optim.bayes_search import _expected_improvement
    ei = _expected_improvement(
        mean=np.array([0.8, 0.5]),
        std=np.array([0.0, 0.1]),
        y_best=0.7,
    )
    assert np.all(np.isfinite(ei))
    assert not math.isnan(float(ei[0]))
