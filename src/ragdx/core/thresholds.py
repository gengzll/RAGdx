"""
Metric Thresholds and Evaluation Criteria

Main Idea:
This module defines default thresholds and evaluation criteria for RAG pipeline metrics. It provides baseline values for determining whether metric scores indicate acceptable performance or require attention.

Functionalities:
- Default thresholds: Predefined acceptable performance levels for common RAG metrics
- Lower-is-better metrics: Identifies metrics where lower values indicate better performance
- Threshold-based evaluation: Enables automated assessment of metric performance
- Comprehensive coverage: Covers retrieval, generation, end-to-end, and operational metrics

Defined thresholds for metrics like:
- Retrieval: context_precision, context_recall, context_entities_recall
- Generation: faithfulness, response_relevancy, context_utilization, noise_sensitivity
- End-to-end: answer_correctness, answer_accuracy, citation_accuracy
- Operational: latency, cost metrics

Usage:
Used by the diagnosis engine to evaluate metric performance:

    from ragdx.core.thresholds import DEFAULT_THRESHOLDS, LOWER_IS_BETTER

    score = 0.75
    threshold = DEFAULT_THRESHOLDS["context_precision"]  # 0.80
    is_good = (score >= threshold) if "context_precision" not in LOWER_IS_BETTER else (score <= threshold)

Thresholds can be customized for specific use cases or domains.
"""

DEFAULT_THRESHOLDS = {
    "context_precision": 0.80,
    "context_recall": 0.80,
    "context_entities_recall": 0.75,
    "hit_rate_at_k": 0.85,
    "faithfulness": 0.90,
    "response_relevancy": 0.85,
    "context_utilization": 0.75,
    "noise_sensitivity": 0.20,
    "hallucination": 0.10,
    "answer_correctness": 0.85,
    "answer_accuracy": 0.85,
    "citation_accuracy": 0.85,
    "user_success_rate": 0.85,
}

LOWER_IS_BETTER = {"noise_sensitivity", "hallucination", "retrieval_latency_ms", "latency_ms", "cost_usd"}
