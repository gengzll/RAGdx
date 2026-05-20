"""
Metric Name Normalization

Main Idea:
This module provides normalization mappings for metric names from different RAG evaluation tools to a unified schema. It ensures consistent metric naming across various evaluation engines and enables seamless integration of results.

Functionalities:
- RAGAS metric mapping: Maps RAGAS-specific metric names to standardized categories and names
- RAGChecker metric mapping: Maps RAGChecker-specific metric names to standardized categories and names
- Category organization: Organizes metrics into retrieval, generation, and end-to-end categories
- Consistent naming: Ensures uniform metric names regardless of the evaluation tool used

Supported mappings:
RAGAS metrics: context_precision, context_recall, context_entity_recall, response_relevancy, faithfulness, noise_sensitivity, answer_correctness, answer_accuracy

RAGChecker metrics: precision, recall, claim_recall, context_utilization, hallucination, self_knowledge, faithfulness

Usage:
This module is used internally by evaluation adapters to normalize metric names:

    from ragdx.core.normalization import RAGAS_MAP

    # Get standardized category and name for a RAGAS metric
    category, name = RAGAS_MAP["context_precision"]  # ("retrieval", "context_precision")

The normalized metrics are then stored in EvaluationResult objects with consistent naming.
"""

from __future__ import annotations

from typing import Dict, Tuple

RAGAS_MAP: Dict[str, Tuple[str, str]] = {
    "context_precision": ("retrieval", "context_precision"),
    "context_recall": ("retrieval", "context_recall"),
    "context_entity_recall": ("retrieval", "context_entities_recall"),
    "response_relevancy": ("generation", "response_relevancy"),
    "faithfulness": ("generation", "faithfulness"),
    "noise_sensitivity": ("generation", "noise_sensitivity"),
    "answer_correctness": ("e2e", "answer_correctness"),
    "answer_accuracy": ("e2e", "answer_accuracy"),
}

RAGCHECKER_MAP: Dict[str, Tuple[str, str]] = {
    "precision": ("retrieval", "context_precision"),
    "recall": ("retrieval", "context_recall"),
    "claim_recall": ("e2e", "answer_correctness"),
    "context_utilization": ("generation", "context_utilization"),
    "hallucination": ("generation", "hallucination"),
    "self_knowledge": ("generation", "noise_sensitivity"),
    "faithfulness": ("generation", "faithfulness"),
}
