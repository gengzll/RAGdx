# Limitations and Roadmap

## 1. Current limitations

### 1.1 Local file store

The default persistence model is simple and local. It is convenient for development, but not a replacement for a shared metadata service.

### 1.2 External execution depends on your runtime

Framework and tool adapters can emit configs and orchestrate trial runs, but actual execution still depends on your environment, installed packages, and runner scripts.

### 1.3 Evaluation quality depends on evaluator quality

`ragdx` does not make a weak evaluator trustworthy by itself. If Ragas or RAGChecker outputs are noisy or the audit sample is too small, diagnosis quality will suffer.

### 1.4 LLM reasoning is helpful but not infallible

LLM diagnosis and LLM planning should be treated as structured reasoning aids, not ground truth.

### 1.5 Heavy optimization backend support is optional

The heavy BO path is configurable, but it still depends on optional packages and runtime support.

## 2. Recommended future improvements

### 2.1 Better trace ingestion

- native OpenTelemetry import
- deeper per-span attribution
- richer trace clustering

### 2.2 Stronger feedback learning

- automatic benchmark evolution from production failures
- persistent posterior learning beyond simple prior updates

### 2.3 Distributed or shared persistence

- database-backed run store
- team-shared experiment history
- artifact versioning

### 2.4 Richer governance workflow

- candidate approval states
- canary and rollback states
- deployment policy checks

### 2.5 More runtime adapters

- Haystack
- custom internal enterprise agent frameworks
- additional vector database and retrieval system integrations

## 3. Practical roadmap suggestion

A practical roadmap for a team adopting `ragdx` is:

1. use normalized offline evaluation only
2. add saved runs and markdown reports
3. add optimization simulation
4. connect one runtime framework in `execute` mode
5. attach feedback and traces
6. introduce LLM diagnosis and LLM planner selectively
7. add heavier BO backend only when search budget and complexity justify it
