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

## 10. `RAGConfig` (production RAG description)

PR1–PR5 added `RAGConfig` as a first-class YAML object so users can
describe a production RAG once and reuse it across `ragdx evaluate`,
`ragdx tune`, and as a library import. See
[12-evaluate-tune-workflow.md](12-evaluate-tune-workflow.md) for the
full schema.

Reliability knobs commonly set in the YAML:

- `runtime` (`langchain` | `llamaindex`) — in-process backend dispatch.
- `judge.llm_max_concurrent` — ragas judge throughput. Default `2`
  (survives strict-rate-limit providers like GLM-4-Flash).
- `judge.llm_max_retries` — ragas judge retry budget. Default `5`.
- `generator.temperature` — forwarded to litellm. Automatically
  clamped to `≥ 0.01` in both the litellm and openai client layers so
  providers that reject `temperature=1e-08` don't crash mid-run.
- `generator.system_instruction` — production prompt. Also used as
  the MIPROv2 seed when `ragdx tune --stage generation` runs.

Credentials in YAML are scrubbed by `RAGConfig.scrubbed_for_commit()`
before `ragdx tune --write-optimized-config` writes the optimized
config — the committed YAML is safe.

## 11. Per-project storage (`--project`)

When the same repo hosts multiple production RAGs, isolate each
project's RunStore so runs, sessions, feedback, and causal priors
don't collide:

```bash
ragdx --project esg evaluate ...      # → .ragdx/projects/esg/
ragdx --project legal evaluate ...    # → .ragdx/projects/legal/
ragdx --project esg dashboard         # only the esg project's runs
```

Equivalent: `RAGDX_PROJECT=esg ragdx evaluate ...`

`RAGDX_ROOT` (fully-qualified path) wins over `--project` when both
are set. Run `ragdx show-config` to confirm the resolved root.

See [12-evaluate-tune-workflow.md §8](12-evaluate-tune-workflow.md#8-per-project-storage---project)
for the full design + worked example.

## 12. Practical configuration guidance

For local development:

- use `simulate` first
- enable `openai` extra only if using LLM diagnosis or planning
- enable runtime extras only for the framework you actually use

For CI and batch use:

- pin package extras explicitly
- set runner commands via environment variables
- export reports for artifact storage
