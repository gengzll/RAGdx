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

from typing import Any, Literal

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, field_validator, model_validator

# Forward-reference target for ``SavedRun.rag_config``. rag_config.py is
# a leaf module (it doesn't import from this one) so there's no cycle.
from ragdx.schemas.rag_config import RAGConfig

# Metrics that are bounded to [0, 1] across ragdx; anything outside is a sign
# of an evaluator misconfiguration and is clipped with a logged warning.
_BOUNDED_METRICS = {
    "context_precision", "context_recall", "context_entity_recall",
    "faithfulness", "response_relevancy", "answer_correctness",
    "answer_accuracy", "citation_accuracy", "hallucination",
    "noise_sensitivity", "precision", "recall", "claim_recall",
    "context_utilization", "self_knowledge",
}


def _validate_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
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
    ground_truth: str | None = None
    answer: str | None = None
    contexts: list[str] = Field(default_factory=list)
    reference_contexts: list[str] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    span_id: str
    parent_span_id: str | None = None
    kind: Literal["query", "retrieve", "rerank", "pack", "generate", "verify", "tool", "judge"] = "query"
    name: str
    started_at: str | None = None
    ended_at: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class QueryTrace(BaseModel):
    trace_id: str
    question: str
    answer: str | None = None
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    spans: list[TraceSpan] = Field(default_factory=list)
    token_usage: dict[str, float] = Field(default_factory=dict)
    latency_ms: float | None = None
    cost_usd: float | None = None
    labels: dict[str, Any] = Field(default_factory=dict)


class EvaluatorScore(BaseModel):
    evaluator: str
    metric: str
    score: float
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackEvent(BaseModel):
    feedback_id: str
    query_id: str | None = None
    kind: FeedbackKind
    severity: Severity = "medium"
    rating: float | None = None
    note: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


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
    evidence: list[str] = Field(default_factory=list)
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
    nodes: list[CausalSignal] = Field(default_factory=list)
    edges: list[CausalEdge] = Field(default_factory=list)

class EvaluationResult(BaseModel):
    retrieval: dict[str, float] = Field(default_factory=dict)
    generation: dict[str, float] = Field(default_factory=dict)
    e2e: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_tool_outputs: dict[str, Any] = Field(default_factory=dict)
    traces: list[QueryTrace] = Field(default_factory=list)
    evaluator_scores: list[EvaluatorScore] = Field(default_factory=list)
    feedback_events: list[FeedbackEvent] = Field(default_factory=list)
    calibrations: list[EvaluatorCalibration] = Field(default_factory=list)

    @field_validator("retrieval", "generation", "e2e", mode="before")
    @classmethod
    def _check_metric_bucket(cls, value: Any) -> dict[str, float]:
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

    def layer_scores(
        self, weights: dict[str, float] | None = None
    ) -> dict[str, dict]:
        """Per-layer aggregate scores (retrieval / generation / e2e).

        Each layer's score is the (optionally weighted) mean of its
        metrics, with lower-is-better metrics inverted so the result is
        uniformly "higher = healthier" in [0, 1]. ``weights`` is a flat
        ``{metric: weight}`` map; absent metrics default to weight 1.0
        (simple mean). See :func:`ragdx.core.metrics.compute_layer_scores`.

        Returns ``{layer: {score, metrics, raw, n}}``; ``score`` is
        ``None`` for an empty layer.
        """
        from ragdx.core.metrics import compute_layer_scores

        flat = {**self.retrieval, **self.generation, **self.e2e}
        return compute_layer_scores(flat, weights=weights)


class DiagnosisHypothesis(BaseModel):
    component: LayerName
    root_cause: str
    severity: Severity = "medium"
    confidence: float = 0.5
    evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_unit(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")
        return value


class DiagnosisLayer(BaseModel):
    """One layer of a diagnosis report (rule-based, LLM, or synthesised).

    Carries the same fields a full report does -- so the renderer can
    show them side-by-side without special-casing. Each layer has its
    own ``source`` tag so the reader always knows where a hypothesis
    came from.
    """

    source: Literal["rule", "llm", "synthesis"] = "rule"
    summary: str = ""
    hypotheses: list[DiagnosisHypothesis] = Field(default_factory=list)
    causal_signals: list[CausalSignal] = Field(default_factory=list)
    metric_gaps: dict[str, float] = Field(default_factory=dict)
    optimization_candidates: list[str] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)
    disambiguation_actions: list[str] = Field(default_factory=list)
    diagnosis_confidence: float = 0.0


class DiagnosisReport(BaseModel):
    """Top-level diagnosis report.

    Two views of the same information:

    * **Top-level fields** (``summary`` / ``hypotheses`` / ...) -- the
      "active" view used by downstream consumers (planner, dashboard,
      CLI summary). When ``--use-llm`` / ``--use-both`` is in effect,
      these reflect the LLM / synthesis output; otherwise they reflect
      the rule-based output. Kept for backwards compatibility.

    * **Layer fields** (``rule_based`` / ``llm_based`` / ``synthesis``)
      -- the underlying per-source views. Populated when the producing
      engine wants to preserve the lineage so the renderer can show
      ``"this hypothesis came from the rule engine, this one came from
      the LLM, this combined view came from synthesis"``. ``None``
      when the producer didn't supply that layer.
    """

    summary: str
    expected_thresholds: dict[str, float] = Field(default_factory=dict)
    metric_gaps: dict[str, float] = Field(default_factory=dict)
    hypotheses: list[DiagnosisHypothesis] = Field(default_factory=list)
    optimization_candidates: list[str] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)
    causal_signals: list[CausalSignal] = Field(default_factory=list)
    causal_graph: CausalGraph = Field(default_factory=CausalGraph)
    evaluator_agreement: dict[str, float] = Field(default_factory=dict)
    diagnosis_confidence: float = 0.0
    disambiguation_actions: list[str] = Field(default_factory=list)
    # Per-layer aggregate scores (retrieval / generation / e2e) computed
    # from the evaluated metrics. ``{layer: {score, metrics, raw, n}}``.
    # Drives the "weakest layer first" prioritization in the summary and
    # the three-layer overview in the HTML report.
    layer_scores: dict[str, dict] = Field(default_factory=dict)

    # ----- Per-source layers (additive, optional) --------------------
    rule_based: DiagnosisLayer | None = None
    llm_based: DiagnosisLayer | None = None
    synthesis: DiagnosisLayer | None = None
    # Which layer drove the top-level fields. ``rule`` for the default
    # ``--use-llm`` False path, ``llm`` for ``--use-llm`` alone,
    # ``synthesis`` for ``--use-both``. ``rule`` is the safe default.
    active_source: Literal["rule", "llm", "synthesis"] = "rule"


class OptimizationExperiment(BaseModel):
    name: str
    tool: ToolName
    target_component: LayerName
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)
    search_strategy: SearchStrategy = "bayesian"
    max_trials: NonNegativeInt = 8
    status: Literal["planned", "running", "done", "failed"] = "planned"
    baseline_score: float | None = None
    candidate_score: float | None = None
    notes: str = ""
    config_artifacts: list[str] = Field(default_factory=list)
    stage: OptimizerStage = "joint"
    constraints: dict[str, float] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("experiment name cannot be empty")
        return value

    @field_validator("objectives")
    @classmethod
    def _weights_non_negative(cls, value: dict[str, float]) -> dict[str, float]:
        for metric, weight in value.items():
            if weight < 0:
                raise ValueError(f"objective weight for {metric!r} must be >= 0, got {weight}")
        return value

    @model_validator(mode="after")
    def _self_dep(self) -> OptimizationExperiment:
        if self.name in self.depends_on:
            raise ValueError(f"experiment {self.name!r} cannot depend on itself")
        return self


class OptimizationPlan(BaseModel):
    objective_metric: str
    experiments: list[OptimizationExperiment] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_dependency_graph(self) -> OptimizationPlan:
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
    payload: dict[str, Any] = Field(default_factory=dict)
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
    parameters: dict[str, Any] = Field(default_factory=dict)
    config_path: str | None = None
    output_path: str | None = None
    log_path: str | None = None
    runner_command: str | None = None
    return_code: int | None = None
    objective_scores: dict[str, float] = Field(default_factory=dict)
    utility: float | None = None
    feasible: bool | None = None
    constraint_violations: dict[str, float] = Field(default_factory=dict)
    feasibility_penalty: float = 0.0
    pareto_dominance_count: int = 0
    pareto_front: bool = False
    logs: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    notes: str = ""
    simulated: bool = False


class OptimizationSession(BaseModel):
    schema_version: int = 1
    session_id: str
    created_at: str
    run_id: str | None = None
    strategy: SearchStrategy
    mode: ExecutionMode = "simulate"
    status: SessionStatus = "planned"
    plan: OptimizationPlan
    total_trials: NonNegativeInt = 0
    completed_trials: NonNegativeInt = 0
    current_experiment: str | None = None
    trials: list[OptimizationTrial] = Field(default_factory=list)
    best_trial_id: str | None = None
    pareto_front_ids: list[str] = Field(default_factory=list)
    feasible_pareto_front_ids: list[str] = Field(default_factory=list)
    hypervolume: NonNegativeFloat = 0.0
    feasible_hypervolume: NonNegativeFloat = 0.0
    notes: str = ""

    @model_validator(mode="after")
    def _completed_le_total(self) -> OptimizationSession:
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
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    baseline_run_id: str | None = None
    latest_session_id: str | None = None
    evaluation: EvaluationResult
    diagnosis: DiagnosisReport
    optimization_plan: OptimizationPlan
    rag_config: RAGConfig | None = None
    """The RAGConfig that produced ``evaluation``, stored as a
    scrubbed copy (``scrubbed_for_commit``) so ``ragdx tune
    --from-run <id>`` can inherit it without an explicit
    ``--base-config`` flag. Optional / default ``None`` for
    backward-compatibility with pre-PR6 SavedRun files."""
