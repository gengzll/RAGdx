# ragdx Overview

## 1. Purpose

`ragdx` is a Python workbench for analyzing and improving Retrieval-Augmented Generation (RAG) systems. It covers the following lifecycle:

1. normalize evaluation signals from one or more external evaluators
2. diagnose failure patterns using rules, an explicit causal graph, and optional LLM reasoning
3. generate a staged optimization plan
4. execute optimization sessions in simulation, preparation, or external execution mode
5. persist runs, sessions, traces, and feedback
6. inspect progress and results in a dashboard

The tool is meant to sit above a RAG application rather than replace the application framework itself.

## 2. Primary use cases

`ragdx` is designed for the following situations:

- evaluating a RAG pipeline with Ragas and or RAGChecker outputs
- diagnosing whether quality loss is caused by retrieval, ranking, chunking, grounding, citation binding, or distribution shift
- generating optimization plans for retrieval, generation, orchestration, and ingestion stages
- tracking optimization sessions over time
- connecting diagnosis and optimization to runtime frameworks such as LangChain or LlamaIndex
- comparing candidate and baseline runs
- attaching production feedback to past runs

## 3. High-level lifecycle

A typical lifecycle in `ragdx` is:

1. prepare an evaluation JSON or normalize external evaluator outputs
2. run diagnosis
3. inspect the generated optimization plan
4. execute the plan in `simulate`, `prepare_only`, or `execute` mode
5. persist the run and session state
6. monitor the session in the dashboard or CLI
7. attach feedback and re-run diagnosis and planning

## 4. Main design ideas

### 4.1 Metric-aware and trace-aware

`ragdx` handles aggregate metrics such as `faithfulness` or `answer_correctness`, but it also supports query traces and spans so diagnosis can use richer evidence than summary scores alone.

### 4.2 Diagnosis before optimization

The system is explicitly not a generic hyperparameter tuner. It attempts to identify the most likely failure region first, then narrows the search plan.

### 4.3 Multi-objective and constraint-aware

Optimization is not treated as a single-score maximization problem. Quality, latency, cost, hallucination, and noise sensitivity can all affect feasibility and candidate selection.

### 4.4 Runtime-agnostic control plane

`ragdx` does not hardcode one application framework. Instead, it can emit config artifacts and orchestrate external trial runs for DSPy, AutoRAG, LangChain, and LlamaIndex.

## 5. What `ragdx` is not

`ragdx` is not:

- a complete RAG application framework
- a vector database
- a document parser by itself
- a replacement for framework-native orchestration
- a full MLOps or deployment platform

It is better understood as a RAG quality and optimization control plane.

## 6. Package structure at a glance

- `src/ragdx/core`: evaluation, normalization, thresholds, comparison, diagnosis engine
- `src/ragdx/engines`: rule-based and LLM-based diagnosis plus evaluator adapters
- `src/ragdx/optim`: planner, executor, heavy BO adapter, framework adapters
- `src/ragdx/schemas`: pydantic data models
- `src/ragdx/storage`: run and session persistence
- `src/ragdx/ui`: Streamlit dashboard
- `src/ragdx/utils`: reporting and plan explanation helpers
- `examples`: runnable examples and dummy runtime pipelines
- `tests`: validation suite
