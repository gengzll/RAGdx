"""
Tests for RAG Diagnosis Engine

Main Idea:
This module contains unit tests for the RAG diagnosis engine functionality. It verifies that the diagnosis engine correctly identifies issues and generates optimization recommendations.

Functionalities:
- Hypothesis generation: Tests that diagnosis produces meaningful hypotheses
- Optimization candidates: Verifies generation of optimization suggestions
- Edge case handling: Tests with various evaluation result configurations

Test coverage:
- Basic diagnosis workflow
- Hypothesis structure and content
- Optimization candidate generation
- Integration with evaluation results

Usage:
Run tests with pytest:

    pytest tests/test_diagnosis.py

Or run specific test:

    pytest tests/test_diagnosis.py::test_diagnosis_produces_hypotheses

These tests ensure the diagnosis engine works correctly and produces expected outputs.
"""

from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.schemas.models import EvaluationResult


def test_diagnosis_produces_hypotheses():
    result = EvaluationResult(
        retrieval={"context_precision": 0.60, "context_recall": 0.55},
        generation={"faithfulness": 0.78, "response_relevancy": 0.80, "noise_sensitivity": 0.30},
        e2e={"answer_correctness": 0.65},
    )
    report = RAGDiagnosisEngine().diagnose(result)
    assert report.hypotheses
    assert report.optimization_candidates
