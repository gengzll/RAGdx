# Runtime Integrations

## 1. Overview

`ragdx` does not require one fixed runtime framework. Instead, it supports tool and runtime adapters so planning and execution can connect to external systems.

Current integration surface includes:

- Ragas
- RAGChecker
- DSPy
- AutoRAG
- LangChain
- LlamaIndex

## 2. Evaluator integrations

### 2.1 Ragas

Purpose:

- broad metric coverage
- evaluation of retrieval, generation, and e2e quality

Integration point:

- `src/ragdx/engines/ragas_adapter.py`

Typical usage:

- normalize Ragas output into `EvaluationResult`
- run diagnosis and planning on top of that normalized structure

### 2.2 RAGChecker

Purpose:

- more fine-grained retriever vs generator diagnostics

Integration point:

- `src/ragdx/engines/ragchecker_adapter.py`

## 3. Optimization tool integrations

### 3.1 DSPy

Purpose:

- generator-side prompt and program optimization

Integration point:

- `src/ragdx/optim/dspy_adapter.py`

Typical usage:

- planner emits DSPy-oriented configs
- external runner executes them
- results are ingested back into the session

### 3.2 AutoRAG

Purpose:

- retrieval and pipeline-oriented configuration search

Integration point:

- `src/ragdx/optim/autorag_adapter.py`

## 4. Runtime framework integrations

### 4.1 LangChain

Integration points:

- `src/ragdx/optim/langchain_adapter.py`
- `examples/langchain_pipeline.py`
- `examples/run_langchain_trial.py`

Usage model:

1. evaluation metadata points to a pipeline module and dataset
2. planner emits configs for the runtime validation experiment
3. external runner command launches a trial script
4. trial output is read back into the session

### 4.2 LlamaIndex

Integration points:

- `src/ragdx/optim/llamaindex_adapter.py`
- `examples/llamaindex_pipeline.py`
- `examples/run_llamaindex_trial.py`

## 5. Example metadata fields

An evaluation file may contain runtime metadata like:

```json
{
  "metadata": {
    "runtime_framework": "langchain",
    "dataset_path": "examples/demo_dataset.jsonl",
    "pipeline_module": "examples.langchain_pipeline:create_pipeline"
  }
}
```

Or:

```json
{
  "metadata": {
    "runtime_framework": "llamaindex",
    "dataset_path": "examples/demo_dataset.jsonl",
    "pipeline_module": "examples.llamaindex_pipeline:create_query_engine"
  }
}
```

## 6. External runner contract

The runner is expected to consume:

- a config artifact
- possibly workdir and trial metadata

It should write an output file containing either:

- `objective_scores`
- or a normalized metrics JSON

Example:

```json
{
  "objective_scores": {
    "answer_correctness": 0.79,
    "citation_accuracy": 0.82
  }
}
```

## 7. Integration guidance

### When to use framework adapters

Use the framework adapters when you already have a RAG application in:

- LangChain
- LlamaIndex

### When to use tool adapters

Use tool adapters when you want to tune:

- retrieval and pipeline configuration with AutoRAG
- prompt and program behavior with DSPy

## 8. Practical recommendation

A common split is:

- use Ragas and RAGChecker for evaluation
- use AutoRAG for retrieval search
- use DSPy for generation-side prompt optimization
- use LangChain or LlamaIndex as the runtime application layer
