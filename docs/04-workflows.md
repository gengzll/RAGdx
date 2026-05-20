# Workflows

## 1. Minimal workflow

### Step 1: prepare an evaluation file

Start from a normalized evaluation JSON with `retrieval`, `generation`, and `e2e` metrics.

### Step 2: run diagnosis

```bash
ragdx diagnose examples/demo_evaluation.json
```

### Step 3: generate a plan

```bash
ragdx plan examples/demo_evaluation.json --human-readable
```

### Step 4: execute or simulate optimization

```bash
ragdx optimize examples/demo_evaluation.json --strategy bayesian --mode simulate
```

### Step 5: inspect the session

```bash
ragdx sessions
ragdx monitor-session <SESSION_ID>
ragdx dashboard
```

## 2. Workflow with LLM reasoning

### Diagnosis with LLM only

```bash
ragdx diagnose examples/demo_evaluation.json --use-llm
```

### Diagnosis with both rule and LLM

```bash
ragdx diagnose examples/demo_evaluation.json --use-both
```

### LLM-refined planning

```bash
ragdx plan examples/demo_evaluation.json --use-llm-planner --human-readable
```

## 3. Workflow with saved runs

### Save a diagnosed run

```bash
ragdx save examples/demo_evaluation.json --name baseline-run
```

### List runs

```bash
ragdx runs
```

### Export a markdown report

```bash
ragdx export-report <RUN_ID> report.md
```

## 4. Workflow with feedback

### Attach feedback

```bash
ragdx attach-feedback <RUN_ID> examples/feedback_events.json
```

### Summarize feedback

```bash
ragdx feedback-summary
```

Feedback then becomes part of the run evidence and may affect diagnosis, planning, and learned causal priors.

## 5. Workflow with external execution

### Prepare the runner command

Example for LangChain:

```bash
export RAGDX_LANGCHAIN_RUNNER_CMD='python examples/run_langchain_trial.py --config {config} --output {output}'
```

### Execute optimization

```bash
ragdx optimize examples/demo_evaluation_langchain.json --mode execute --strategy bayesian --budget 6
```

### Monitor progress

```bash
ragdx monitor-session <SESSION_ID>
ragdx dashboard
```

## 6. Workflow for tool-output normalization

If you already have outputs from Ragas or RAGChecker:

```bash
ragdx normalize-tools --ragas-json ragas_output.json --ragchecker-json ragchecker_output.json --output-json normalized.json
```

Then use `normalized.json` as the main evaluation input.

## 7. Workflow for baseline comparison

```bash
ragdx compare current.json baseline.json
```

This prints a metric comparison table, showing deltas and direction of change.

## 8. Recommended enterprise workflow

For an enterprise or production setting, a more realistic workflow is:

1. ingest evaluator output and traces from one batch or release
2. save a run
3. attach production feedback and reviewer feedback
4. run diagnosis and generate a baseline-relative plan
5. execute the plan in simulation first
6. move to external execution with framework-specific runners
7. review feasible Pareto candidates in the dashboard
8. compare candidate run against baseline
9. export markdown reports for governance or review
