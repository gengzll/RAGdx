![Alt text](docs/logo.png)

`ragdx` is a Python workbench for **RAG evaluation, diagnosis, and
optimization**. It sits above an existing RAG application as a
quality + optimization control plane rather than replacing your
runtime framework, retriever stack, or orchestration layer.

It pairs (a) deterministic rule-based + LLM-augmented diagnosis on
top of an explicit causal graph with (b) stage-targeted Bayesian
search + DSPy prompt evolution, plus (c) per-experiment workspaces
and rendered HTML reports that close the optimize-and-compare loop.

## Install

```bash
pip install -e ".[experiment]"     # everything for an end-to-end run
# or pick extras à la carte:
pip install -e ".[langchain,bo,ragas,deepeval,dspy,openai]"
```

API keys are read from env vars in this fallback order:
`ZHIPU_API_KEY` → `OPENAI_API_KEY` (other providers documented via
`ragdx providers list`).

## 60-second quickstart (per-experiment workspace)

The recommended way to run anything is through a **workspace** — a
single folder under `workspaces/<name>/` that owns your config,
questions, and every artifact every command produces:

```bash
# 1. Create the workspace + drop in your config + eval questions
ragdx workspace init my-exp --config rag.yaml --questions q.jsonl

# 2. Score the baseline (uses the workspace's config + questions)
ragdx workspace eval my-exp

# 3. Diagnose it -- pick rule-based, LLM, or "both" for synthesis
ragdx workspace diagnose my-exp                 # rule-based only
ragdx workspace diagnose my-exp --use-llm       # LLM-refined
ragdx workspace diagnose my-exp --use-both      # rule + LLM + synthesised

# 4. Tune what the diagnosis pointed at
ragdx workspace tune-rag my-exp                 # chunker / retriever / joint
ragdx workspace tune-prompt my-exp              # DSPy prompt + few-shot demos

# 5. Render the HTML report (baseline diagnosis → optimization → comparison)
ragdx workspace report my-exp

# 6. Compare multiple workspaces side-by-side
ragdx workspace compare my-exp my-exp-v2 -o compare.html
```

The legacy `ragdx evaluate / diagnose / tune` commands keep working
unchanged — workspaces are an additive convenience.

## What you get

| Command | What it does |
|---|---|
| `ragdx workspace ...` | Per-experiment workspace (config + history + artifacts in one folder) |
| `ragdx evaluate` | Score a `RAGConfig` against an eval suite (`--evaluator {ragas,deepeval}`) |
| `ragdx diagnose` | Rule-based root-cause + optional LLM refinement, schema with `rule_based` / `llm_based` / `synthesis` layers |
| `ragdx tune --stage X` | Stage-targeted Bayesian search (chunking / retrieval / joint) or DSPy prompt tuning (generation, optimizers: `mipro / copro / bootstrap_fewshot / gepa`) |
| `ragdx experiment` | One-shot end-to-end (corpus → BO → DSPy → diagnosis bundle) |
| `ragdx experiment-report` | Render any bundle as a self-contained HTML report |
| `ragdx providers list/template <name>` | LLM provider catalog (OpenAI, Anthropic, Zhipu, Moonshot, DeepSeek, Qwen, Ollama, vLLM, Groq, Together) |
| `ragdx dashboard` | Streamlit dashboard over the local RunStore |

Diagnosis is layered:
- **rule-based** — deterministic causal-graph reasoning (always run)
- **LLM-refined** — `--use-llm` adds a CoT view
- **synthesis** — `--use-both` adds an LLM-merged final answer
- Reports show all populated layers side-by-side with an `(active)` marker

Evaluation backends are pluggable: `--evaluator ragas` (default) or
`--evaluator deepeval` (G-Eval-backed; resists `faithfulness`
saturation on permissive judges). DSPy inner-loop metrics include
`embed_rubric` (cheap, no saturation) and `geval` (deepeval) on top
of the classic `ragas` / `llm_judge` / `token_f1`.

## Reports

`ragdx workspace report <name>` writes a self-contained HTML
covering:

- baseline diagnosis with per-source layers, causal graph SVG,
  posterior table, hypothesis evidence with anchor links back to the
  metric / signal it references
- Bayesian search trials + per-record metric breakdowns (not just
  trial means)
- DSPy A/B with word-level answer diff (red strikethrough / green
  highlight) and optimizer evolution timeline
- baseline-vs-optimized comparison: resolved / persisted / emerged
  hypotheses, posterior shifts, metric deltas
- optimized diagnosis (what's still broken)
- run cost (wall time, mean per-question latency)

## Documentation

Full details live under [`docs/`](docs/):

- [`12-evaluate-tune-workflow.md`](docs/12-evaluate-tune-workflow.md) — the recommended starting point: hands-on walkthrough of `evaluate → diagnose → tune`
- [`01-overview.md`](docs/01-overview.md) — scope, design goals, lifecycle
- [`02-architecture.md`](docs/02-architecture.md) — component layout
- [`03-data-models.md`](docs/03-data-models.md) — `EvaluationResult` / `DiagnosisReport` / trace schemas
- [`04-workflows.md`](docs/04-workflows.md) — operational workflows
- [`05-cli-and-dashboard.md`](docs/05-cli-and-dashboard.md) — every CLI command + the dashboard
- [`06-configuration.md`](docs/06-configuration.md) — env vars, extras, runners
- [`07-optimization-and-diagnosis.md`](docs/07-optimization-and-diagnosis.md) — metrics, causal graph, optimization strategies
- [`08-runtime-integrations.md`](docs/08-runtime-integrations.md) — DSPy, AutoRAG, LangChain, LlamaIndex
- [`09-extension-guide.md`](docs/09-extension-guide.md) — adding metrics, tools, runtimes
- [`10-examples.md`](docs/10-examples.md) — recipes
- [`11-limitations-and-roadmap.md`](docs/11-limitations-and-roadmap.md) — what doesn't work yet
- `new_demo3/` — fully-executed walkthrough (artifacts + console logs preserved)

## Programmatic API

The CLI is a thin wrapper. The same eval + diagnose + plan calls in
Python:

```python
from ragdx import (
    UnifiedEvaluator,        # ragas / ragchecker / embedding adapters
    RAGDiagnosisEngine,      # rule-based + optional LLM
    OptimizationPlanner,     # diagnosis → optimization plan
    OptimizationExecutor,    # simulate / prepare / execute
    RunStore,                # local persistence
    run_experiment,          # end-to-end shortcut
)
```

## Testing

```bash
python -m pytest tests/ -q
```

CI matrix: Python 3.10 / 3.11 / 3.12 on Ubuntu. `ruff check src tests`
must pass; `mypy src/ragdx` runs informational.

## License

Apache-2.0.
