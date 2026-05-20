# Data Models and Artifacts

## 1. Overview

`ragdx` uses Pydantic models to make runs, sessions, diagnosis, plans, traces, and feedback explicit and serializable.

The central model file is:

- `src/ragdx/schemas/models.py`

## 2. Evaluation-level models

### 2.1 `DatasetRecord`

Represents one evaluation record.

Fields:

- `question`
- `ground_truth`
- `answer`
- `contexts`
- `reference_contexts`
- `citations`
- `metadata`

### 2.2 `EvaluationResult`

Represents the normalized evaluation state used for diagnosis and optimization.

Buckets:

- `retrieval`
- `generation`
- `e2e`

Additional evidence:

- `metadata`
- `raw_tool_outputs`
- `traces`
- `evaluator_scores`
- `feedback_events`
- `calibrations`

This model is the main input to `diagnose`, `plan`, `optimize`, and `save`.

## 3. Trace models

### 3.1 `TraceSpan`

Represents one span in a query execution path.

Important fields:

- `span_id`
- `parent_span_id`
- `kind`: `query`, `retrieve`, `rerank`, `pack`, `generate`, `verify`, `tool`, `judge`
- `name`
- `started_at`
- `ended_at`
- `attributes`
- `events`

### 3.2 `QueryTrace`

Represents one end-to-end query trace.

Important fields:

- `trace_id`
- `question`
- `answer`
- `retrieved_chunks`
- `citations`
- `spans`
- `token_usage`
- `latency_ms`
- `cost_usd`
- `labels`

## 4. Evaluator models

### 4.1 `EvaluatorScore`

Represents a metric score from a specific evaluator.

Fields:

- `evaluator`
- `metric`
- `score`
- `confidence`
- `metadata`

### 4.2 `EvaluatorCalibration`

Used to capture agreement with audited or gold-labeled data.

Fields:

- `metric`
- `agreement_score`
- `audit_sample_size`
- `notes`

## 5. Feedback models

### 5.1 `FeedbackEvent`

Represents production or review feedback.

Kinds include:

- `thumbs_up`
- `thumbs_down`
- `user_correction`
- `escalation`
- `hallucination`
- `latency`
- `cost`
- `policy`

Fields:

- `feedback_id`
- `query_id`
- `kind`
- `severity`
- `rating`
- `note`
- `metadata`
- `created_at`

## 6. Diagnosis models

### 6.1 `DiagnosisHypothesis`

Represents one hypothesis about a failure source.

Fields:

- `component`
- `root_cause`
- `severity`
- `confidence`
- `evidence`
- `recommended_actions`

### 6.2 `CausalSignal`

Represents one node in the diagnosis causal structure.

Fields:

- `node`
- `component`
- `posterior`
- `prior`
- `evidence`
- `recommended_experiment`

### 6.3 `CausalEdge` and `CausalGraph`

These represent the explicit layered causal graph.

- `CausalEdge`: `source`, `target`, `weight`, `rationale`
- `CausalGraph`: `nodes`, `edges`

### 6.4 `DiagnosisReport`

The central output of diagnosis.

Fields include:

- `summary`
- `expected_thresholds`
- `metric_gaps`
- `hypotheses`
- `optimization_candidates`
- `priority_actions`
- `causal_signals`
- `causal_graph`
- `evaluator_agreement`
- `diagnosis_confidence`
- `disambiguation_actions`

## 7. Optimization models

### 7.1 `OptimizationExperiment`

Represents one planned experiment.

Fields include:

- `name`
- `tool`
- `target_component`
- `description`
- `parameters`
- `objectives`
- `search_space`
- `search_strategy`
- `max_trials`
- `status`
- `baseline_score`
- `candidate_score`
- `notes`
- `config_artifacts`
- `stage`
- `constraints`
- `depends_on`

The `parameters` field contains the most detailed plan semantics, including:

- `baseline_metrics`
- `metric_directions`
- `target_thresholds`
- `target_specs`
- `constraint_bounds`
- `objective_weights`

### 7.2 `OptimizationPlan`

Fields:

- `objective_metric`
- `experiments`
- `rationale`

### 7.3 `OptimizationTrial`

Represents one executed or prepared trial.

Fields include:

- `trial_id`
- `experiment_name`
- `tool`
- `strategy`
- `status`
- `parameters`
- `config_path`
- `output_path`
- `log_path`
- `runner_command`
- `return_code`
- `objective_scores`
- `utility`
- `feasible`
- `constraint_violations`
- `feasibility_penalty`
- `pareto_front`
- `started_at`
- `completed_at`
- `notes`

### 7.4 `OptimizationSession`

Represents the session-level state.

Fields include:

- `session_id`
- `created_at`
- `run_id`
- `strategy`
- `mode`
- `status`
- `plan`
- `total_trials`
- `completed_trials`
- `current_experiment`
- `trials`
- `best_trial_id`
- `pareto_front_ids`
- `feasible_pareto_front_ids`
- `hypervolume`
- `feasible_hypervolume`
- `notes`

## 8. Persistence model

### 8.1 `SavedRun`

Represents the persisted run record.

Fields:

- `run_id`
- `created_at`
- `name`
- `tags`
- `notes`
- `baseline_run_id`
- `latest_session_id`
- `evaluation`
- `diagnosis`
- `optimization_plan`
