"""Per-layer metric aggregation.

``EvaluationResult`` already partitions individual metrics into three
layers — retrieval, generation, e2e — but historically there was no
single "how is the retrieval layer doing overall" number. The
composite objective collapses *all* layers into one score, which is
right for picking an optimization winner but hides *where* the
bottleneck is.

This module computes a per-layer aggregate so a reader (or the
diagnosis) can see at a glance::

    retrieval:  0.71   (context_precision 0.55, context_recall 0.87)
    generation: 0.93   (faithfulness 1.00, answer_relevancy 0.86)
    e2e:        0.65   (answer_correctness 0.65)

Direction normalization: some metrics are lower-is-better
(``hallucination``, ``noise_sensitivity``). Averaging those raw against
higher-is-better metrics would be meaningless, so we invert them
(``1 - value``) before aggregating, giving a uniform "higher = healthier"
layer score in [0, 1].
"""

from __future__ import annotations

from ragdx.core.thresholds import LOWER_IS_BETTER

# Canonical metric name -> layer. The single source of truth for which
# layer a metric belongs to. Mirrors (and supersedes) the ad-hoc
# ``_RETRIEVAL_METRICS`` / ``_GENERATION_METRICS`` / ``_E2E_METRICS``
# sets that were duplicated across ``workflows/evaluate`` and
# ``experiments``; those should defer to this map.
LAYER_OF: dict[str, str] = {
    # --- retrieval ---
    "context_precision": "retrieval",
    "context_recall": "retrieval",
    "context_entities_recall": "retrieval",
    "context_entity_recall": "retrieval",
    "hit_rate_at_k": "retrieval",
    # --- generation ---
    "faithfulness": "generation",
    "answer_relevancy": "generation",
    "response_relevancy": "generation",
    "context_utilization": "generation",
    "noise_sensitivity": "generation",
    "hallucination": "generation",
    "bias": "generation",
    "toxicity": "generation",
    # --- e2e ---
    "answer_correctness": "e2e",
    "answer_accuracy": "e2e",
    "answer_similarity": "e2e",
    "citation_accuracy": "e2e",
    "claim_recall": "e2e",
    "summarization": "e2e",
    "g_eval": "e2e",
    "user_success_rate": "e2e",
}

LAYERS = ("retrieval", "generation", "e2e")


def metric_layer(name: str) -> str | None:
    """Return the layer a metric belongs to, or ``None`` if unknown."""
    return LAYER_OF.get(name)


def _oriented(name: str, value: float) -> float:
    """Flip lower-is-better metrics to higher-is-better.

    ``hallucination=0.1`` (good) becomes ``0.9``; ``faithfulness=0.9``
    (already good) stays ``0.9``. Clamped to [0, 1] so a stray
    out-of-range raw value can't blow up the layer mean.
    """
    v = max(0.0, min(1.0, float(value)))
    return 1.0 - v if name in LOWER_IS_BETTER else v


def compute_layer_scores(
    scores: dict[str, float],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Aggregate a flat ``{metric: value}`` dict into per-layer scores.

    Parameters
    ----------
    scores:
        Flat metric dict (e.g. ``{**result.retrieval, **result.generation,
        **result.e2e}`` or a BO trial's ``scores``). Non-numeric values
        and unknown metric names are skipped.
    weights:
        Optional ``{metric: weight}``. Applied *within* each layer as a
        weighted mean. Metrics absent from this dict default to weight
        ``1.0`` (so by default every layer is a simple mean). Weight
        ``0`` excludes a metric from its layer aggregate.

    Returns
    -------
    ``{layer: {"score": float|None, "metrics": {name: oriented_value},
    "raw": {name: raw_value}, "n": int}}`` for each of retrieval /
    generation / e2e. ``score`` is ``None`` when a layer has no metrics
    (so the caller can render "—" rather than a misleading 0.0).

    The per-layer ``score`` is the weighted mean of the *direction-
    oriented* values (lower-is-better metrics inverted), so it's always
    "higher = healthier" in [0, 1].
    """
    weights = weights or {}
    out: dict[str, dict] = {
        layer: {"score": None, "metrics": {}, "raw": {}, "n": 0}
        for layer in LAYERS
    }
    acc: dict[str, list[tuple[float, float]]] = {layer: [] for layer in LAYERS}

    for name, value in scores.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        layer = LAYER_OF.get(name)
        if layer is None:
            continue
        oriented = _oriented(name, value)
        w = float(weights.get(name, 1.0))
        out[layer]["raw"][name] = float(value)
        out[layer]["metrics"][name] = round(oriented, 4)
        out[layer]["n"] += 1
        if w > 0:
            acc[layer].append((oriented, w))

    for layer in LAYERS:
        pairs = acc[layer]
        wsum = sum(w for _, w in pairs)
        if wsum > 0:
            out[layer]["score"] = round(
                sum(v * w for v, w in pairs) / wsum, 4
            )
    return out


def weakest_layer(layer_scores: dict[str, dict]) -> str | None:
    """Return the layer with the lowest (non-None) aggregate score.

    Used by the diagnosis to prioritize "attack this layer first".
    ``None`` when no layer has a score.
    """
    scored = [
        (layer, d["score"])
        for layer, d in layer_scores.items()
        if d.get("score") is not None
    ]
    if not scored:
        return None
    return min(scored, key=lambda kv: kv[1])[0]


__all__ = [
    "LAYERS",
    "LAYER_OF",
    "compute_layer_scores",
    "metric_layer",
    "weakest_layer",
]
