# Configuration

## 1. Python version

The project requires:

- Python `>=3.10`

## 2. Base dependencies

Core dependencies include:

- `pydantic`
- `pandas`
- `numpy`
- `pyyaml`
- `plotly`
- `streamlit`
- `typer`
- `rich`
- `scikit-learn`
- `networkx`

## 3. Optional extras

Optional dependency groups in `pyproject.toml` include:

- `langchain`
- `llamaindex`
- `bo`
- `ragas`
- `ragchecker`
- `dspy`
- `autorag`
- `openai`
- `all`

Examples:

```bash
pip install -e .
pip install -e ".[openai]"
pip install -e ".[langchain,llamaindex,bo]"
pip install -e ".[all]"
```

## 4. LLM-related environment variables

### `OPENAI_API_KEY`

Required for:

- `--use-llm`
- `--use-both`
- `--use-llm-planner`

### `RAGDX_OPENAI_MODEL`

Optional. Defaults to:

- `gpt-5.4-thinking`

## 5. External runner environment variables

These commands are used in `execute` mode.

### DSPy

- `RAGDX_DSPY_RUNNER_CMD`

### AutoRAG

- `RAGDX_AUTORAG_RUNNER_CMD`

### LangChain

- `RAGDX_LANGCHAIN_RUNNER_CMD`

### LlamaIndex

- `RAGDX_LLAMAINDEX_RUNNER_CMD`

Runner command templates can use placeholders such as:

- `{config}`
- `{output}`
- `{workdir}`
- `{trial_id}`
- `{session_id}`
- `{tool}`

## 6. Optimization backend configuration

### `RAGDX_BO_BACKEND`

Controls the heavier optimization backend.

Typical values:

- unset: internal default behavior
- `ax`: use Ax when installed and available

## 7. Fallback behavior

### `RAGDX_FALLBACK_SIMULATE_ON_MISSING_RUNNER`

When set to a truthy value, `execute` mode may fall back to simulated scoring if the configured runner is unavailable.

## 8. File-based storage

The store uses local folders by default. You should ensure the process has write access to:

- `.ragdx`
- `.ragdx/runs`
- `.ragdx/optimization/sessions`
- `.ragdx/feedback`
- `.ragdx/causal`

## 9. Evaluation file configuration

The evaluation JSON can include `metadata` fields such as:

- `runtime_framework`
- `dataset_path`
- `pipeline_module`
- any runtime-specific annotations needed by your runner

## 10. Practical configuration guidance

For local development:

- use `simulate` first
- enable `openai` extra only if using LLM diagnosis or planning
- enable runtime extras only for the framework you actually use

For CI and batch use:

- pin package extras explicitly
- set runner commands via environment variables
- export reports for artifact storage
