"""
Evaluation Result Comparison Utilities

Main Idea:
This module provides utilities for comparing evaluation results between different RAG pipeline runs or configurations. It calculates metric deltas and determines whether changes represent improvements or regressions.

Functionalities:
- compare_results: Compare two EvaluationResult objects and return detailed metric comparisons
- Automatic direction detection: Determines if changes are improvements, regressions, or unchanged
- Metric-specific logic: Handles metrics where lower values are better (e.g., noise_sensitivity, hallucination)
- Comprehensive coverage: Compares all metrics across retrieval, generation, and end-to-end categories

Usage:
Compare current and baseline evaluation results:

    from ragdx.core.compare import compare_results

    comparisons = compare_results(current_eval, baseline_eval)
    for comp in comparisons:
        print(f"{comp.metric}: {comp.current:.3f} vs {comp.baseline:.3f} ({comp.direction})")

The function returns a list of MetricComparison objects with current value, baseline value, delta, and direction.
"""

from __future__ import annotations

from ragdx.schemas.models import EvaluationResult, MetricComparison


LOWER_IS_BETTER = {"noise_sensitivity", "hallucination"}


def compare_results(current: EvaluationResult, baseline: EvaluationResult) -> list[MetricComparison]:
    metrics = {}
    for bucket in (current.retrieval, current.generation, current.e2e):
        metrics.update(bucket)
    for bucket in (baseline.retrieval, baseline.generation, baseline.e2e):
        metrics.update(bucket)

    out: list[MetricComparison] = []
    for metric in sorted(metrics):
        c = current.score(metric)
        b = baseline.score(metric)
        if c is None or b is None:
            continue
        raw_delta = c - b
        better = raw_delta < 0 if metric in LOWER_IS_BETTER else raw_delta > 0
        direction = "unchanged" if abs(raw_delta) < 1e-9 else ("improved" if better else "regressed")
        out.append(MetricComparison(metric=metric, current=c, baseline=b, delta=round(raw_delta, 4), direction=direction))
    return out
