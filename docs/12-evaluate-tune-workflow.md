# The `evaluate` + `tune` Workflow (`RAGConfig` Control Plane)

The earlier docs describe an **offline-evaluation-first** workflow: you
bring a normalized `EvaluationResult` JSON (from your own pipeline,
from Ragas, from RAGChecker via `ragdx normalize-tools`) and feed it
through `diagnose → plan → optimize → save`.

PR1–PR5 added a second entry point: describe your production RAG as a
**`RAGConfig` YAML**, point `ragdx` at it, and let `ragdx evaluate`
build the pipeline + score it + (optionally) persist into the
RunStore in one call. `ragdx tune` is the matching stage-targeted
optimizer: take a `RAGConfig`, vary one slice of it (chunking /
retrieval / generation / joint), and emit the winning config.

This page covers everything PR1–PR5 introduced and how to compose it
with the existing `diagnose / plan / save / dashboard` flow.

## 1. `RAGConfig`: the single description of a RAG

`src/ragdx/schemas/rag_config.py` defines `RAGConfig` and six
`*Spec` sub-models. A bare `RAGConfig()` reproduces the demo
pipeline, so a minimal config is small:

```yaml
# rag_config.yaml
name: esg-analyst
runtime: langchain          # or "llamaindex"
corpus:
  kind: pdf
  path: docs/asmpt_esg_2024.pdf
chunker:
  strategy: recursive
  chunk_size: 512
  chunk_overlap: 50
embedder:
  kind: huggingface
  model_name: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
retriever:
  vectorstore: faiss
  search_type: similarity
  top_k: 5
generator:
  provider: litellm
  model: openai/glm-4-flash
  api_base: https://open.bigmodel.cn/api/paas/v4
  # api_key: filled in via env var (see §6)
  system_instruction: "You are an ESG analyst. Answer using only the supplied context."
  temperature: 0.05
  max_tokens: 1024
  timeout: 60
judge:
  model: null              # null -> reuse generator.model
  llm_max_concurrent: 2
  llm_max_retries: 5
```

The six `Spec` types are `CorpusSpec`, `ChunkerSpec`, `EmbedderSpec`,
`RetrieverSpec`, `GeneratorSpec`, `JudgeSpec`. All of them have
`extra='forbid'` — typos fail loudly. YAML round-tripping is
lossless (`RAGConfig.from_yaml(...) → to_yaml(...)`).

### `RAGConfig.with_override(...)`

Stage optimizers and BO sweeps swap one slice without mutating the
base:

```python
swapped = base.with_override(retriever=RetrieverSpec(top_k=12))
assert base.retriever.top_k == 5   # base unchanged
assert swapped.retriever.top_k == 12
```

### `RAGConfig.scrubbed_for_commit()`

Strips `api_key` from `generator` and `judge` before serialization.
`ragdx tune --write-optimized-config` calls this automatically — the
YAML it writes is safe to commit. **Always** call it before persisting
a config that was hydrated from env vars or CLI flags.

## 2. `RAGPipeline`: one runtime, two backends

`src/ragdx/runtime/pipeline.py` defines the `RAGPipeline` ABC.
`RAGPipeline.build(config, chunks, embedder, llm_callable)` dispatches
on `config.runtime`:

| `runtime` | Backend | Used for |
|---|---|---|
| `"langchain"` (default) | LangChain FAISS + `similarity_search` | Production default. CI-cheap. |
| `"llamaindex"` | LlamaIndex `VectorStoreIndex.as_retriever()` | When you want LlamaIndex's retriever ecosystem in-process. |

Both backends share the same surface:

```python
pipeline = RAGPipeline.build(cfg, chunks, embedder=..., llm_callable=...)
ctxs = pipeline.retrieve("Q?", top_k=5)         # optional per-call override
ans  = pipeline.generate("Q?", ctxs, system_instruction="...")
out  = pipeline.answer("Q?")                    # retrieve + generate
```

LlamaIndex is an optional extra: install via `pip install
"ragdx[llamaindex]"`. The dispatch is lazy — if you don't set
`runtime: llamaindex` you don't pay for the import.

> Note: the **in-process** LlamaIndex backend is distinct from the
> **subprocess runner** pattern described in `docs/08-runtime-integrations.md`.
> The runner pattern (`RAGDX_LANGCHAIN_RUNNER_CMD`,
> `RAGDX_LLAMAINDEX_RUNNER_CMD`) is for shelling out to a separately
> installed runtime — useful for full enterprise pipelines. The
> in-process backend is for "just score this config quickly."

## 3. `ragdx evaluate`: from YAML to `EvaluationResult`

```bash
ragdx evaluate \
    --config rag_config.yaml \
    --questions my_eval.jsonl \
    --corpus docs/asmpt_esg_2024.pdf \
    --output baseline.json
```

The output `baseline.json` is a normalized `EvaluationResult` — the
same shape `ragdx diagnose / plan / compare` already consume.

### Closed-loop mode: `--save`

To skip the chain `evaluate → diagnose → save`, pass `--save`:

```bash
ragdx evaluate \
    --config rag_config.yaml \
    --questions my_eval.jsonl \
    --corpus docs/asmpt_esg_2024.pdf \
    --output baseline.json \
    --save --name "esg-baseline" \
    --use-llm --use-llm-planner       # optional
```

`--save` causes the command to:

1. Build the `EvaluationResult` (same as without `--save`).
2. Run rule-based or LLM diagnosis (`--use-llm` / `--use-both`).
3. Build the optimization plan (`--use-llm-planner` to refine with an LLM).
4. Persist run + report + plan to the RunStore via `RunStore.save_run(...)`.

After this:

- `ragdx runs` lists the run.
- `ragdx dashboard` shows it in the Scores / Diagnosis / Plan tabs.
- `ragdx export-report <run_id> report.md` exports it.
- `--baseline-run-id <id>` links this run to an earlier baseline so
  `ragdx compare` / the dashboard's delta view work without manual JSON
  copy-paste.

The `--use-llm` / `--use-both` / `--use-llm-planner` /
`--baseline-run-id` flags only make sense in combination with
`--save`; passing them without `--save` fails fast with a clear error
(no expensive eval is wasted).

### Eval-suite shape (`--questions`)

JSONL, one record per line:

```jsonl
{"question": "What was ASMPT's Scope 1 emissions in 2023?", "ground_truth": "..."}
{"question": "Summarize the 2024 governance changes."}
```

`ground_truth` is optional — its presence (per-record) determines
whether reference-based ragas metrics (`context_recall`,
`answer_correctness`) run. A whitespace-only string is treated as
absent.

## 4. `ragdx tune`: stage-targeted optimization

```bash
ragdx tune \
    --base-config rag_config.yaml \
    --questions my_eval.jsonl \
    --corpus docs/asmpt_esg_2024.pdf \
    --stage retrieval --budget 8 \
    --output tune_retrieval.json \
    --write-optimized-config rag_config.optimized.yaml \
    --save --name "retrieval-sweep-v3" \
    --baseline-run-id <baseline_run_id>
```

`--stage` is one of:

| Stage | What it varies | Vstore rebuild per trial? |
|---|---|---|
| `joint` | `chunk_size`, `chunk_overlap`, `top_k` | Yes (chunker changes) |
| `chunking` | `chunk_size`, `chunk_overlap` (retriever held fixed) | Yes |
| `retrieval` | `top_k` only (chunker held fixed) | **No** — one pipeline, fast |
| `generation` | `system_instruction` via DSPy MIPROv2 | No — same retrieval |

`joint` reproduces what `ragdx experiment` does. The three slice
optimizers are the cost-saving paths: pin everything except the slice
you want to improve.

### `--save` on `tune`

Same as `evaluate --save`, but the `EvaluationResult` is **synthesized
from the best trial's ragas scores** (for BO stages) or the MIPROv2
optimized re-run (for `generation`). Metadata stamped onto the saved
run includes:

- `tune_stage`, `best_params`, `best_composite`
- `base_config_path`, `questions_path`, `corpus`
- `bundle_path` (where the full trial bundle was written)
- `optimized_config_path` if `--write-optimized-config` was set

### `--write-optimized-config` (with credential scrubbing)

When set, `tune` writes the winning `RAGConfig` as YAML so you can
diff it against your production config or commit it. Before writing,
`RAGConfig.scrubbed_for_commit()` clears `generator.api_key` and
`judge.api_key`. The original in-memory config is unchanged.

This was a real security bug fix (`23-COMMIT 73ee990`): the
env-resolved API key used to land in the YAML, which would leak if
the file was committed. Tests verify the scrubbing
(`tests/test_rag_config_and_pipeline.py::test_scrubbed_for_commit_*`).

## 5. The closed loop

The README promises this six-step loop. With `evaluate --save` and
`tune --save`, every step is one `ragdx` command:

```bash
# 1. Score the production config
ragdx evaluate --config rag.yaml -q q.jsonl --output base.json \
    --save --name baseline
# -> SavedRun A

# 2. (optional) re-diagnose with LLM, attach to the same run
ragdx diagnose base.json --save --name baseline-llm --use-llm

# 3. Plan
ragdx plan base.json --human-readable

# 4. Tune the bottleneck stage the plan flagged
ragdx tune --base-config rag.yaml -q q.jsonl --stage retrieval \
    --budget 8 --output tune_retrieval.json \
    --write-optimized-config rag.optimized.yaml \
    --save --name retrieval-sweep-v3 --baseline-run-id <A>
# -> SavedRun B, linked to A as baseline

# 5. Score the optimized config and compare
ragdx evaluate --config rag.optimized.yaml -q q.jsonl --output opt.json \
    --save --name optimized --baseline-run-id <A>
# -> SavedRun C

ragdx compare opt.json base.json            # standalone diff
ragdx runs                                  # all three runs
ragdx dashboard                             # Compare tab renders the deltas
```

## 6. Runtime knobs (RAG-LLM-side reliability)

Production RAG against rate-limited or strict providers (GLM,
Anthropic, OpenAI) needs throttling and parameter cleanup. These
ship out of the box; you don't need to wire them.

| Knob | Where set | What it does |
|---|---|---|
| `generator.api_key` | YAML / `--api-key` / env (`ZHIPU_API_KEY`, `OPENAI_API_KEY`) | Authentication. The CLI resolves in that order. |
| `judge.llm_max_concurrent` | YAML | Ragas judge `RunConfig.max_workers`. Default 2 — survives GLM rate limits. |
| `judge.llm_max_retries` | YAML | Ragas judge `RunConfig.max_retries`. Default 5. |
| `generator.temperature` | YAML | Forwarded to litellm. **Clamped to ≥ 0.01** in two layers (litellm + openai client) so providers like GLM that reject `temperature=1e-08` don't crash mid-run. The clamp is idempotent. |
| `generator.system_instruction` | YAML | Production prompt. Used as the seed for `tune --stage generation` (MIPROv2 starts from it instead of silently using a default signature). |

The temperature clamp (`apply_litellm_temperature_clamp` in
`runtime/factories.py`) is applied automatically when `build_runtime`
runs, and verified by
`tests/test_pr4_evaluate_tune.py::test_apply_temperature_clamp_is_idempotent`.

## 7. When to use which command

| Goal | Command |
|---|---|
| "I have ragas/RAGChecker output — score and diagnose it." | `ragdx normalize-tools` → `ragdx diagnose --save` |
| "I have a RAG built in LangChain — score it against my eval suite." | `ragdx evaluate --config rag.yaml --save` |
| "I have a `RAGConfig` and want to improve one slice." | `ragdx tune --base-config rag.yaml --stage retrieval --save` |
| "I want the README's six-step loop." | `evaluate --save` → `tune --save --baseline-run-id` → `evaluate --save` → `compare` |
| "I want full end-to-end BO + DSPy across all stages on a corpus." | `ragdx experiment` (the demo path; see docs/04 §5) |

## 8. Per-project storage (`--project`)

By default every saved run lands in `.ragdx/runs/<id>.json`, every
session in `.ragdx/optimization/sessions/`, and so on. If you work on
multiple production RAGs from the same repo (ESG analyst, legal docs,
internal search...) the global namespace forces you to rely on
the `name` field to keep them apart.

Pass `--project <name>` as a root flag to isolate one project's
artifacts:

```bash
ragdx --project esg evaluate --config esg.yaml -q esg_eval.jsonl --save --name baseline
# Storage root for this command: .ragdx/projects/esg/

ragdx --project esg runs                # only shows esg runs
ragdx --project esg dashboard           # only shows esg's data
ragdx runs                              # shows only the default-project runs (the old shape)
```

Resolution order (highest precedence wins):

1. `RAGDX_ROOT` env var (fully-qualified path) — ignores `--project`.
2. `--project <name>` (root flag) → `.ragdx/projects/<name>/`.
3. `RAGDX_PROJECT` env var → same as above.
4. Default → `.ragdx/`.

The resolved storage path is shown by `ragdx show-config`:

```json
{
  "storage": {
    "root": ".ragdx/projects/esg",
    "project": "esg"
  },
  ...
}
```

Per-project isolation covers **everything** under the storage root:
runs, optimization sessions, feedback events, causal priors. Two
projects can have colliding run names without overwriting each
other's data.

## 9. `tune --from-run`: let the plan drive the optimizer

`evaluate --save` produces a `SavedRun` carrying three things:

1. The `EvaluationResult` (ragas scores).
2. A `DiagnosisReport` + `OptimizationPlan` (which experiments to run
   next, with planned `stage`, `search_space`, `max_trials`).
3. A **scrubbed copy of the `RAGConfig` that produced the eval**
   (new in PR6, persisted automatically).

Before PR6, `ragdx tune` ignored every bit of (1)–(3) and asked the
user to re-pass `--base-config`, `--questions`, `--corpus`,
`--stage`, `--budget`, and `--baseline-run-id`. With PR6 you can just
hand it the run id:

```bash
ragdx --project esg tune --from-run <baseline_run_id> \
    --save --name retrieval-sweep
# That's it.
```

What gets auto-inherited:

| From the SavedRun | Used as | Override |
|---|---|---|
| `rag_config` (PR6+ saves it scrubbed) | `--base-config` | `--base-config <path>` |
| `evaluation.metadata.questions_path` | `--questions` | `--questions <path>` |
| `evaluation.metadata.corpus` | `--corpus` | `--corpus <path>` |
| `optimization_plan.experiments[0].stage` | `--stage` | `--stage <name>` (or `auto`) |
| `optimization_plan.experiments[0].max_trials` | `--budget` | `--budget <N>` |
| `optimization_plan.experiments[0].search_space` | `StageContext.{top_ks, chunk_sizes, chunk_overlaps}` | `--stage` axis defaults |
| The `--from-run` value itself | `--baseline-run-id` | `--baseline-run-id <id>` |

When the plan has multiple experiments (it usually does — corpus +
retrieval + generation), pick which one to inherit from:

```bash
ragdx tune --from-run <id> --experiment retrieval-pipeline-search ...
```

Default is the first experiment (plans are pre-sorted by priority).

### What it requires

- A SavedRun produced by `ragdx evaluate --save` on **PR6+**. Older
  runs don't have `rag_config` populated — pre-PR6 SavedRuns fail
  with a pointer to the fix.
- A valid `api_key` at runtime. Stored `rag_config` is scrubbed by
  design; pass `--api-key` or set `ZHIPU_API_KEY` / `OPENAI_API_KEY`.

### Why this matters

The Scenario C tune in `new_demo3/` (manually configured) swept
`top_k ∈ [1, 3, 5, 7]` (the hardcoded default) and found `top_k=3`.
Scenario F2 (`tune --from-run`) inherited the plan's recommended
range `top_k ∈ [4, 6, 8, 10]` and found `top_k=4`, which scored
**better** (1.675 vs 1.667). The plan's search space was the
informed choice; the manual default was the lazy one.

## 10. Related docs

- `docs/03-data-models.md` — `EvaluationResult` / `SavedRun` / `OptimizationPlan` shapes.
- `docs/04-workflows.md` — the offline-evaluation-first workflows (normalize → diagnose → plan).
- `docs/05-cli-and-dashboard.md` — full command list.
- `docs/06-configuration.md` — settings (storage root, LLM provider).
- `docs/07-optimization-and-diagnosis.md` — diagnosis + planning internals.
- `docs/08-runtime-integrations.md` — subprocess runner pattern (distinct from §2 here).
