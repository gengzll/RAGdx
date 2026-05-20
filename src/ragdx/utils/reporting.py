"""
Reporting and Formatting Utilities

Main Idea:
This module provides utilities for formatting, summarizing, and reporting RAG diagnosis and optimization data in human-readable formats. It converts complex data structures into clear, actionable summaries.

Functionalities:
- Plan summarization: Convert optimization plans into readable text with experiment details
- Experiment formatting: Structure experiment descriptions with parameters and objectives
- Target spec summaries: Format metric targets and constraints clearly
- JSON export: Save data structures to JSON files
- Value formatting: Consistent formatting for numeric values and metrics

Key functions:
- summarize_plan: Create human-readable optimization plan summaries
- summarize_experiment: Format individual experiment details
- summarize_target_spec: Describe metric targets and constraints
- save_json: Export data to JSON format

Usage:
Summarize an optimization plan:

    from ragdx.utils.reporting import summarize_plan

    summary = summarize_plan(plan_dict)
    print(summary)

Save results to JSON:

    save_json(evaluation_result.model_dump(), "results.json")

These utilities are used by the CLI and dashboard for user-friendly output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Any, path: str | Path) -> str:
    path = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def summarize_target_spec(metric: str, spec: Dict[str, Any]) -> str:
    direction = spec.get("direction", "monitor")
    mode = spec.get("mode", "target")
    baseline = spec.get("baseline_value")
    target = spec.get("target_value")
    delta = spec.get("delta_from_baseline")
    bits = [f"{metric}: {direction}/{mode}"]
    if baseline is not None:
        bits.append(f"baseline={_fmt_value(baseline)}")
    if target is not None:
        bits.append(f"target={_fmt_value(target)}")
    if delta is not None:
        bits.append(f"delta={delta:+.4f}")
    if "min_acceptable" in spec:
        bits.append(f"min_acceptable={_fmt_value(spec['min_acceptable'])}")
    if "max_acceptable" in spec:
        bits.append(f"max_acceptable={_fmt_value(spec['max_acceptable'])}")
    return "; ".join(bits)


def summarize_experiment(experiment: Dict[str, Any]) -> str:
    params = experiment.get("parameters", {})
    target_specs = params.get("target_specs", {})
    objective_weights = params.get("objective_weights", experiment.get("objectives", {}))
    constraint_bounds = params.get("constraint_bounds", experiment.get("constraints", {}))
    lines = [
        f"Experiment: {experiment.get('name', '')}",
        f"Stage: {experiment.get('stage', '')} | Tool: {experiment.get('tool', '')} | Strategy: {experiment.get('search_strategy', '')}",
        f"Component: {experiment.get('target_component', '')}",
        f"Description: {experiment.get('description', '')}",
    ]
    if experiment.get("depends_on"):
        lines.append(f"Depends on: {', '.join(experiment['depends_on'])}")
    if experiment.get("baseline_score") is not None:
        lines.append(f"Primary baseline score: {_fmt_value(experiment['baseline_score'])}")
    if target_specs:
        lines.append("Target specs:")
        for metric, spec in target_specs.items():
            lines.append(f"  - {summarize_target_spec(metric, spec)}")
    if objective_weights:
        lines.append("Objective weights (trade-off coefficients, not metric targets):")
        for metric, weight in objective_weights.items():
            lines.append(f"  - {metric}: {_fmt_value(weight)}")
    if constraint_bounds:
        lines.append("Constraint bounds:")
        for key, value in constraint_bounds.items():
            lines.append(f"  - {key}: {_fmt_value(value)}")
    notes = experiment.get("notes")
    if notes:
        lines.append(f"Notes: {notes}")
    return "\n".join(lines)


def summarize_plan(plan: Dict[str, Any]) -> str:
    lines = [f"Objective metric: {plan.get('objective_metric', '')}"]
    rationale = plan.get("rationale", [])
    if rationale:
        lines.append("Rationale:")
        for item in rationale:
            lines.append(f"- {item}")
    experiments = plan.get("experiments", [])
    if not experiments:
        lines.append("No experiments planned.")
        return "\n".join(lines)
    lines.append("")
    for idx, experiment in enumerate(experiments, start=1):
        lines.append(f"[{idx}] {summarize_experiment(experiment)}")
        if idx != len(experiments):
            lines.append("")
    return "\n".join(lines)
