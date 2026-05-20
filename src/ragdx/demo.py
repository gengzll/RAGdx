"""
Demo Script for RAG Diagnosis Library

Main Idea:
This module provides a simple demonstration of the RAG Diagnosis Library's core functionality. It showcases how to diagnose a RAG pipeline's performance issues and generate optimization plans using sample evaluation data.

Functionalities:
- Creates a sample EvaluationResult with typical RAG metrics
- Runs the diagnosis engine to identify issues and generate hypotheses
- Builds an optimization plan with recommended experiments
- Prints a summary of the diagnosis and optimization suggestions

This demo uses hardcoded sample data to illustrate the workflow without requiring actual RAG pipeline evaluation results.

Usage:
Run the demo directly:

    python -m ragdx.demo

Or from the source directory:

    python src/ragdx/demo.py

The output will show:
- Diagnosis summary with identified issues
- Priority actions for immediate improvement
- Optimization plan with specific experiments to run
"""

from __future__ import annotations

from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult


def main() -> None:
    result = EvaluationResult(
        retrieval={"context_precision": 0.63, "context_recall": 0.57, "context_entities_recall": 0.54},
        generation={"faithfulness": 0.79, "response_relevancy": 0.82, "noise_sensitivity": 0.31, "context_utilization": 0.61},
        e2e={"answer_correctness": 0.68, "citation_accuracy": 0.71},
        metadata={"dataset": "demo", "tools": ["ragas", "ragchecker"]},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    plan = OptimizationPlanner().build_plan(report, result=result)
    print("Diagnosis summary:")
    print(report.summary)
    print("\nPriority actions:")
    for item in report.priority_actions:
        print(f"- {item}")
    print("\nOptimization plan:")
    for exp in plan.experiments:
        print(f"- {exp.name} [{exp.tool}] -> {exp.description}")


if __name__ == "__main__":
    main()
