# Architecture

## 1. Logical architecture

`ragdx` can be viewed as seven layers:

1. input layer
2. normalization layer
3. diagnosis layer
4. planning layer
5. execution layer
6. persistence layer
7. visualization and reporting layer

## 2. Component diagram

```text
Evaluation JSON / Tool Outputs / Traces / Feedback
                |
                v
        UnifiedEvaluator / Normalization
                |
                v
       RAGDiagnosisEngine
       | rule analyzer
       | causal graph
       | optional LLM diagnosis
                |
                v
        OptimizationPlanner
       | heuristic planner
       | optional LLM planner
                |
                v
       OptimizationExecutor
       | simulate
       | prepare_only
       | execute
       | heavy BO backend
                |
                v
              RunStore
       | runs
       | sessions
       | feedback
       | causal priors
                |
                v
      Dashboard / CLI / Markdown reports
```

## 3. Core modules

### 3.1 `ragdx.core`

Key responsibilities:

- normalize evaluation outputs
- compare result sets
- run diagnosis
- manage metric thresholds and interpretation

Important files:

- `core/evaluator.py`
- `core/diagnosis.py`
- `core/normalization.py`
- `core/compare.py`
- `core/thresholds.py`
- `core/datasets.py`

### 3.2 `ragdx.engines`

Key responsibilities:

- convert evaluator outputs into normalized metrics
- produce diagnosis reports
- enrich diagnosis using LLM reasoning

Important files:

- `engines/root_cause.py`
- `engines/llm_diagnosis.py`
- `engines/ragas_adapter.py`
- `engines/ragchecker_adapter.py`

### 3.3 `ragdx.optim`

Key responsibilities:

- turn diagnosis into optimization experiments
- build baseline-relative targets and constraints
- execute sessions
- support Bayesian and Pareto-style search
- emit configs for external tools and runtimes

Important files:

- `optim/planner.py`
- `optim/executor.py`
- `optim/heavy_bo.py`
- `optim/dspy_adapter.py`
- `optim/autorag_adapter.py`
- `optim/langchain_adapter.py`
- `optim/llamaindex_adapter.py`

### 3.4 `ragdx.storage`

Key responsibilities:

- persist runs
- persist optimization sessions
- persist feedback
- persist learned causal priors
- export markdown reports

Important file:

- `storage/run_store.py`

### 3.5 `ragdx.ui`

Key responsibility:

- Streamlit dashboard for inspection of plans, traces, feedback, sessions, and run comparisons

Important file:

- `ui/dashboard.py`

## 4. Execution modes

The executor supports three modes:

### 4.1 `simulate`

Used to validate planning and session orchestration without launching external tools. Useful for testing, demos, and pipeline validation.

### 4.2 `prepare_only`

Generates config artifacts and session records but does not execute external trials.

### 4.3 `execute`

Runs external tools or runtime-specific trial scripts using configured runner commands.

## 5. Search strategies

Two strategies are exposed at the top level:

- `bayesian`
- `pareto_evolutionary`

Internally, the executor may use the built-in model-based search or a heavier backend if configured.

## 6. Persistence layout

By default, run and session state is stored in hidden local folders such as:

- `.ragdx/runs`
- `.ragdx/optimization/sessions`
- `.ragdx/feedback`
- `.ragdx/causal/priors.json`

This layout makes the tool easy to use locally without a separate metadata database.
