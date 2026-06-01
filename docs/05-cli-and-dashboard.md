# CLI and Dashboard

## 1. CLI overview

The Typer application entrypoint is:

- `ragdx`

The Streamlit dashboard entrypoint is:

- `ragdx dashboard`
- or `ragdx-dashboard`

## 2. CLI commands

### `evaluate`

Purpose:

- score a `RAGConfig` YAML against a JSONL eval suite
- emit a normalized `EvaluationResult` JSON
- optionally diagnose + plan + persist to the RunStore in one call

Main options:

- `--config` / `-c` (RAGConfig YAML path)
- `--questions` / `-q` (JSONL eval suite)
- `--corpus` (override YAML's corpus path)
- `--output` / `-o`
- `--api-key`
- `--name`
- `--save` (persist to RunStore via `RunStore.save_run`)
- `--baseline-run-id` (link this run as a delta of another)
- `--use-llm` / `--use-both` / `--use-llm-planner` (only with `--save`)

See [12-evaluate-tune-workflow.md](12-evaluate-tune-workflow.md) for the
full closed-loop workflow.

### `tune`

Purpose:

- optimize one slice of a `RAGConfig` against a JSONL eval suite
- four stages: `joint` (all knobs), `chunking`, `retrieval` (fastest — single pipeline), `generation` (DSPy MIPROv2 over the prompt)
- optionally write the winning config as YAML (credentials scrubbed)
- optionally persist to the RunStore

Main options:

- `--base-config` / `-c`
- `--questions` / `-q`
- `--corpus`
- `--stage` / `-s` (one of `joint` / `chunking` / `retrieval` / `generation`)
- `--budget` / `-b` (BO trial count for BO stages)
- `--bo-init`
- `--seed`
- `--output` / `-o`
- `--write-optimized-config` (winning RAGConfig YAML; credentials scrubbed before write)
- `--api-key`
- `--save` (persist synthesized EvaluationResult to RunStore)
- `--name`
- `--baseline-run-id`
- `--use-llm` / `--use-both` / `--use-llm-planner` (only with `--save`)

### `diagnose`

Purpose:

- load an evaluation file
- run diagnosis
- optionally use LLM diagnosis
- optionally save the run

Main options:

- `--use-llm`
- `--use-both`
- `--use-llm-planner`
- `--save`
- `--name`
- `--baseline-run-id`

### `plan`

Purpose:

- generate an optimization plan from an evaluation file
- optionally use the LLM planner
- optionally print a human-readable explanation

Main options:

- `--strategy`
- `--budget`
- `--use-llm-planner`
- `--human-readable`

### `optimize`

Purpose:

- diagnose
- plan
- execute optimization
- optionally save the run and session

Main options:

- `--strategy`
- `--budget`
- `--mode`
- `--save-run`
- `--use-llm`
- `--use-both`
- `--use-llm-planner`

### `save`

Purpose:

- save evaluation, diagnosis, and plan into the local store

### `compare`

Purpose:

- compare two evaluation files and show metric deltas

### `runs`

Purpose:

- list persisted runs

### `sessions`

Purpose:

- list persisted optimization sessions

### `monitor-session`

Purpose:

- inspect one session via CLI

### `normalize-tools`

Purpose:

- normalize Ragas and or RAGChecker output into a standard evaluation JSON

### `export-report`

Purpose:

- export a run summary as markdown

### `attach-feedback`

Purpose:

- attach one or more feedback events to a persisted run

### `feedback-summary`

Purpose:

- aggregate summary over stored feedback events

### `dashboard`

Purpose:

- launch Streamlit dashboard

### `explain-plan`

Purpose:

- explain a plan JSON in a human-readable form

### `show-runner-templates`

Purpose:

- print runner environment variable templates for external execution

## 3. Dashboard responsibilities

The dashboard is the visual inspection layer for:

- saved runs
- saved sessions
- diagnosis summary and causal signals
- plan details
- traces
- feedback
- optimization progress
- Pareto and feasible Pareto views

## 4. Typical dashboard sections

Depending on implementation state and UI layout, the dashboard focuses on the following logical sections:

- run selection and session selection
- diagnosis summary
- causal graph or causal signal summary
- plan explanation and target specs
- optimization session trial table
- feasible and infeasible candidate breakdown
- traces and feedback
- comparison views

## 5. CLI vs dashboard usage guidance

Prefer the CLI when:

- running in CI
- automating offline diagnosis and planning
- exporting reports
- integrating with shell scripts

Prefer the dashboard when:

- reviewing complex plans
- inspecting optimization trial histories
- comparing feasible candidates
- auditing target specs and constraints
