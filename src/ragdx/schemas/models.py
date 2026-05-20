"""
Data Models and Schemas

Main Idea:
This module defines all the data models and schemas used throughout the RAG Diagnosis Library. It provides type-safe, validated data structures for evaluation results, diagnosis reports, optimization plans, and related entities.

Functionalities:
- Dataset records: Structures for RAG evaluation data (questions, answers, contexts)
- Evaluation results: Standardized metrics across retrieval, generation, and end-to-end layers
- Diagnosis reports: Structured diagnosis output with hypotheses and recommendations
- Optimization plans: Experiment definitions and parameter spaces
- Traces and feedback: Observability and user feedback data structures
- Causal analysis: Models for root cause analysis and causal graphs

Key model categories:
- Input/Output: DatasetRecord, EvaluationResult, DiagnosisReport
- Optimization: OptimizationPlan, OptimizationExperiment, OptimizationSession
- Observability: QueryTrace, TraceSpan, FeedbackEvent
- Analysis: CausalSignal, CausalGraph, DiagnosisHypothesis

Usage:
Import and instantiate models:

    from ragdx.schemas.models import EvaluationResult, DatasetRecord

    result = EvaluationResult(
        retrieval={"context_precision": 0.85},
        generation={"faithfulness": 0.92}
    )

All models use Pydantic for validation and provide JSON serialization methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, field_validator, model_validator

# Metrics that are bounded to [0, 1] across ragdx; anything outside is a sign
# of an evaluator misconfiguration and is clipped with a logged warning.
_BOUNDED_METRICS = {
    "context_precision", "context_recall", "context_entity_recall",
    "faithfulness", "response_relevancy", "answer_correctness",
    "answer_accuracy", "citation_accuracy", "hallucination",
    "noise_sensitivity", "precision", "recall", "claim_recall",
    "context_utilization", "self_knowledge",
}


def _validate_metric_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for name, raw in bucket.items():
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {name!r} must be numeric, got {raw!r}") from exc
        if name in _BOUNDED_METRICS:
            if value < 0.0 or value > 1.0:
                # Clamp rather than raise — evaluators occasionally drift slightly outside.
                value = max(0.0, min(1.0, value))
        cleaned[name] = value
    return cleaned


LayerName = Literal["retrieval", "generation", "e2e", "pipeline"]
Severity = Literal["low", "medium", "high", "critical"]
ToolName = Literal["ragas", "ragchecker", "dspy", "autorag", "langchain", "llamaindex", "manual"]
SearchStrategy = Literal["bayesian", "pareto_evolutionary"]
ExecutionMode = Literal["simulate", "prepare_only", "execute"]
TrialStatus = Literal["planned", "running", "done", "failed", "prepared"]
SessionStatus = Literal["planned", "running", "completed", "failed", "prepared"]
OptimizerStage = Literal["corpus", "retrieval", "generation", "orchestration", "joint"]
FeedbackKind = Literal["thumbs_up", "thumbs_down", "user_correction", "escalation", "hallucination", "latency", "cost", "policy"]


class DatasetRecord(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    answer: Optional[str] = None
    contexts: List[str] = Field(default_factory=list)
    reference_contexts: List[str] = Field(default_factory=list)
    citations: List[int] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    span_id: str
    parent_span_id: Optional[str] = None
    kind: Literal["query", "retrieve", "rerank", "pack", "generate", "verify", "tool", "judge"] = "query"
    name: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class QueryTrace(BaseModel):
    trace_id: str
    question: str
    answer: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Any] = Field(default_factory=list)
    spans: List[TraceSpan] = Field(default_factory=list)
    token_usage: Dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    labels: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorScore(BaseModel):
    evaluator: str
    metric: str
    score: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackEvent(BaseModel):
    feedback_id: str
    query_id: Optional[str] = None
    kind: FeedbackKind
    severity: Severity = "medium"
    rating: Optional[float] = None
    note: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class EvaluatorCalibration(BaseModel):
    metric: str
    agreement_score: float = 0.0
    audit_sample_size: int = 0
    notes: str = ""


class CausalSignal(BaseModel):
    node: str
    component: LayerName
    posterior: float = 0.0
    prior: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    recommended_experiment: str = ""

    @field_validator("posterior", "prior")
    @classmethod
    def _prob_in_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {value}")
        return value




class CausalEdge(BaseModel):
    source: str
    target: str
    weight: float = 0.0
    rationale: str = ""


class CausalGraph(BaseModel):
    nodes: List[CausalSignal] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    retrieval: Dict[str, float] = Field(default_factory=dict)
    generation: Dict[str, float] = Field(default_factory=dict)
    e2e: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    traces: List[QueryTrace] = Field(default_factory=list)
    evaluator_scores: List[EvaluatorScore] = Field(default_factory=list)
    feedback_events: List[FeedbackEvent] = Field(default_factory=list)
    calibrations: List[EvaluatorCalibration] = Field(default_factory=list)

    @field_validator("retrieval", "generation", "e2e", mode="before")
    @classmethod
    def _check_metric_bucket(cls, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("metric bucket must be a dict[str, float]")
        return _validate_metric_bucket(value)

    def score(self, metric: str, default: float | None = None) -> float | None:
        for bucket in (self.retrieval, self.generation, self.e2e):
            if metric in bucket:
                return bucket[metric]
        return default


class DiagnosisHypothesis(BaseModel):
    component: LayerName
    root_cause: str
    severity: Severity = "medium"
    confidence: float = 0.5
    evidence: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_unit(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")
        return value


class DiagnosisReport(BaseModel):
    summary: str
    expected_thresholds: Dict[str, float] = Field(default_factory=dict)
    metric_gaps: Dict[str, float] = Field(default_factory=dict)
    hypotheses: List[DiagnosisHypothesis] = Field(default_factory=list)
    optimization_candidates: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)
    causal_signals: List[CausalSignal] = Field(default_factory=list)
    causal_graph: CausalGraph = Field(default_factory=CausalGraph)
    evaluator_agreement: Dict[str, float] = Field(default_factory=dict)
    diagnosis_confidence: float = 0.0
    disambiguation_actions: List[str] = Field(default_factory=list)


class OptimizationExperiment(BaseModel):
    name: str
    tool: ToolName
    target_component: LayerName
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    objectives: Dict[str, float] = Field(default_factory=dict)
    search_space: Dict[str, List[Any]] = Field(default_factory=dict)
    search_strategy: SearchStrategy = "bayesian"
    max_trials: NonNegativeInt = 8
    status: Literal["planned", "running", "done", "failed"] = "planned"
    baseline_score: Optional[float] = None
    candidate_score: Optional[float] = None
    notes: str = ""
    config_artifacts: List[str] = Field(default_factory=list)
    stage: OptimizerStage = "joint"
    constraints: Dict[str, float] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("experiment name cannot be empty")
        return value

    @field_validator("objectives")
    @classmethod
    def _weights_non_negative(cls, value: Dict[str, float]) -> Dict[str, float]:
        for metric, weight in value.items():
            if weight < 0:
                raise ValueError(f"objective weight for {metric!r} must be >= 0, got {weight}")
        return value

    @model_validator(mode="after")
    def _self_dep(self) -> "OptimizationExperiment":
        if self.name in self.depends_on:
            raise ValueError(f"experiment {self.name!r} cannot depend on itself")
        return self


class OptimizationPlan(BaseModel):
    objective_metric: str
    experiments: List[OptimizationExperiment] = Field(default_factory=list)
    rationale: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_dependency_graph(self) -> "OptimizationPlan":
        names = {e.name for e in self.experiments}
        for e in self.experiments:
            missing = [d for d in e.depends_on if d not in names]
            if missing:
                raise ValueError(
                    f"experiment {e.name!r} depends on unknown experiment(s): {missing}"
                )
        return self


class ToolRunResult(BaseModel):
    tool: ToolName
    success: bool
    payload: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class MetricComparison(BaseModel):
    metric: str
    current: float
    baseline: float
    delta: float
    direction: Literal["improved", "regressed", "unchanged"]


class OptimizationTrial(BaseModel):
    trial_id: str
    experiment_name: str
    tool: ToolName
    strategy: SearchStrategy
    status: TrialStatus = "planned"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    config_path: Optional[str] = None
    output_path: Optional[str] = None
    log_path: Optional[str] = None
    runner_command: Optional[str] = None
    return_code: Optional[int] = None
    objective_scores: Dict[str, float] = Field(default_factory=dict)
    utility: Optional[float] = None
    feasible: Optional[bool] = None
    constraint_violations: Dict[str, float] = Field(default_factory=dict)
    feasibility_penalty: float = 0.0
    pareto_dominance_count: int = 0
    pareto_front: bool = False
    logs: List[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: str = ""
    simulated: bool = False


class OptimizationSession(BaseModel):
    schema_version: int = 1
    session_id: str
    created_at: str
    run_id: Optional[str] = None
    strategy: SearchStrategy
    mode: ExecutionMode = "simulate"
    status: SessionStatus = "planned"
    plan: OptimizationPlan
    total_trials: NonNegativeInt = 0
    completed_trials: NonNegativeInt = 0
    current_experiment: Optional[str] = None
    trials: List[OptimizationTrial] = Field(default_factory=list)
    best_trial_id: Optional[str] = None
    pareto_front_ids: List[str] = Field(default_factory=list)
    feasible_pareto_front_ids: List[str] = Field(default_factory=list)
    hypervolume: NonNegativeFloat = 0.0
    feasible_hypervolume: NonNegativeFloat = 0.0
    notes: str = ""

    @model_validator(mode="after")
    def _completed_le_total(self) -> "OptimizationSession":
        if self.completed_trials > self.total_trials:
            raise ValueError(
                f"completed_trials ({self.completed_trials}) > total_trials ({self.total_trials})"
            )
        return self


class SavedRun(BaseModel):
    schema_version: int = 1
    run_id: str
    created_at: str
    name: str
    tags: List[str] = Field(default_factory=list)
    notes: str = ""
    baseline_run_id: Optional[str] = None
    latest_session_id: Optional[str] = None
    evaluation: EvaluationResult
    diagnosis: DiagnosisReport
    optimization_plan: OptimizationPlan
