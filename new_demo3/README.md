# new_demo3 — End-to-end walkthrough of the PR1–PR5 `evaluate` / `tune` workflow

This directory is a real, executed end-to-end run of every command
added in the `RAGConfig` / `RAGPipeline` / `evaluate` / `tune` control
plane (PR1–PR5 + the `--save` RunStore wiring follow-up). The corpus
is the public **ASMPT 2024 ESG report** (`e0522-asmptesgreport.pdf`,
114 pages); the generator is **GLM-4-Flash** through ZHIPU's
OpenAI-compatible endpoint. All commands were run on
**2026-06-01**.

> Every JSON/log file in this directory was emitted by `ragdx` during
> the run — they are not hand-crafted samples. Run IDs and timings
> are from the live execution.

## What this demo covers

Five scenarios, in dependency order:

| # | Command | What it teaches |
|---|---|---|
| **A** | `ragdx evaluate` | Score a `RAGConfig` and emit a normalized `EvaluationResult` JSON. The simplest mode — no RunStore, no diagnose. |
| **B** | `ragdx evaluate --save` | Same scoring, **plus** diagnose + plan + persist to RunStore in one call. The closed-loop replacement for `evaluate → diagnose → save`. |
| **C** | `ragdx tune --stage retrieval --save` | Stage-targeted Bayesian search (fastest stage — single pipeline rebuild). Synthesizes an EvaluationResult from the best trial and persists it linked to a baseline run. |
| **D** | `ragdx evaluate --save --baseline-run-id` on the tuned config | Re-score the winning config and link it as a delta of the baseline. |
| **E** | `ragdx runs` / `ragdx compare` / `ragdx export-report` | RunStore visibility — list / diff / export. |

The full background and design rationale is in
[`docs/12-evaluate-tune-workflow.md`](../docs/12-evaluate-tune-workflow.md).

## Prerequisites

```bash
# From the project root:
pip install -e ".[langchain,bo,ragas,openai]"
export ZHIPU_API_KEY=<your-zhipu-key>     # or OPENAI_API_KEY
```

The PDF (`e0522-asmptesgreport.pdf`) lives at the repo root.

## Inputs

- [`rag_config.yaml`](rag_config.yaml) — production-style RAG config
  for an ESG analyst (`top_k=3`, GLM-4-Flash, conservative
  `temperature=0.01`, FAISS + MiniLM-L6 embeddings).
- [`questions.jsonl`](questions.jsonl) — 3 ESG questions, no ground
  truth (`no_gt` mode → ragas runs only the no-reference metrics
  `context_precision`, `faithfulness`, `answer_relevancy`).

---

## Scenario A — Pure scoring, no persistence

**When to use it:** quick sanity check on a config before committing
to a full diagnose/save cycle. Or in CI where you only need the
numbers.

```bash
PYTHONPATH=src python -m ragdx.cli evaluate \
    --config new_demo3/rag_config.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/A_baseline.json \
    --name "demo3-A-no-save"
```

Output ([`A_baseline.json`](A_baseline.json), full log in
[`A_console.log`](A_console.log)):

```text
Wrote new_demo3\A_baseline.json
  Retrieval: {"context_precision": 0.5556}
  Generation: {"faithfulness": 1.0}
  E2E: {}

Next: `ragdx diagnose <out>` / `ragdx plan <out>` / `ragdx compare <out> <baseline>` (or re-run with --save to persist to the RunStore).
```

Notes:

- **548 chunks** generated from the PDF (`chunk_size=512`,
  `chunk_overlap=50`). 3 records × 3 metrics = 9 ragas evaluations
  (visible in the progress bar). One metric (`answer_relevancy`)
  returned NaN and was filtered out by
  `_scores_to_evaluation_result`'s non-numeric guard — the partition
  is lossless, dropped scores end up under
  `metadata['unmapped_scores']` if any.
- `e2e` is empty because no ground truth was provided; `context_recall`
  / `answer_correctness` only run when GT is populated.
- The hint at the bottom suggests the next step is either chaining
  `diagnose` / `plan` / `compare` manually, **or** re-running with
  `--save` (which is Scenario B).

---

## Scenario B — `evaluate --save`: the closed loop

**When to use it:** production. Score → diagnose → plan → persist all
in one call so the result shows up in `ragdx runs` and the dashboard
without manual chaining.

```bash
PYTHONPATH=src python -m ragdx.cli evaluate \
    --config new_demo3/rag_config.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/B_baseline_saved.json \
    --save --name "demo3-baseline"
```

Output (full log in [`B_console.log`](B_console.log)):

```text
Wrote new_demo3\B_baseline_saved.json
  Retrieval: {"context_precision": 0.5556}
  Generation: {"faithfulness": 1.0}
  E2E: {}
Saved run: 46c0fcbe5954

Next: `ragdx runs` / `ragdx dashboard` / `ragdx export-report 46c0fcbe5954 report.md`.
```

The metric values are identical to Scenario A (same config, same data
— but with the diagnose + plan + persist side effects). The new
information is **`46c0fcbe5954`**: the SavedRun ID we'll link to in
Scenarios C and D.

Available flags (none used here but worth knowing):

- `--use-llm` / `--use-both` — switch diagnosis from rule-based to
  LLM-only or rule+LLM. Requires `ZHIPU_API_KEY` / `OPENAI_API_KEY`
  in the env so `get_llm_callable()` can build the explainer LLM
  (independent of `--api-key`, which only feeds the **generator**).
- `--use-llm-planner` — refine the optimization plan with an LLM.
- `--baseline-run-id <id>` — set a baseline link at evaluate time.
- These flags only make sense with `--save`; passing them without it
  fails fast.

---

## Scenario C — `tune --stage retrieval --save`: stage-targeted BO

**When to use it:** you've baseline-scored a config and want to
improve **one slice** without re-running everything. `retrieval` is
the cheapest stage because the vector store is built once and reused
across trials (only `top_k` varies per call).

```bash
PYTHONPATH=src python -m ragdx.cli tune \
    --base-config new_demo3/rag_config.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --stage retrieval --budget 4 --bo-init 2 \
    --output new_demo3/C_tune_retrieval.json \
    --write-optimized-config new_demo3/rag_config.optimized.yaml \
    --save --name "demo3-retrieval-sweep" \
    --baseline-run-id 46c0fcbe5954
```

Output (full log in [`C_console.log`](C_console.log)):

```text
Running retrieval optimizer on 548 chunks x 3 records (GT mode: no_gt, budget: 4)...
[4 BO trials over top_k ∈ {1, 3, 5, 7}, each one ragas-evaluating 3 records]
Wrote new_demo3\C_tune_retrieval.json
Wrote optimized config new_demo3\rag_config.optimized.yaml (credentials scrubbed)
Best composite: 1.667
Best params: {'top_k': 3}
Saved run: 77e06e2eecc6 (name: demo3-retrieval-sweep)
```

What happened:

- BO searched `top_k ∈ {1, 3, 5, 7}` (the default `RetrievalOptimizer`
  axis). Best was **`top_k=3`** — same as the baseline. This is a
  perfectly normal outcome: it means our baseline was already at a
  local optimum within the search space, and the tune **confirmed** it
  rather than improving it. The composite score is reported so you
  can compare across stages or sweeps.
- [`rag_config.optimized.yaml`](rag_config.optimized.yaml) is the
  winning `RAGConfig` written to disk. **Credentials are scrubbed**
  (`api_key: null`) by `RAGConfig.scrubbed_for_commit()` before
  serialization — this file is safe to commit.
- SavedRun **`77e06e2eecc6`** is linked to baseline `46c0fcbe5954`
  via `--baseline-run-id`, so the dashboard's compare view auto-pairs
  them.
- [`C_tune_retrieval.json`](C_tune_retrieval.json) is the full
  bundle: 4 trials, scores, elapsed times, the search space, the
  objective spec.

> **⚠ Bug discovered + fixed during this scenario.** The first run
> of `--output` C_tune_retrieval.json leaked
> `generator.api_key` through the embedded `base_config` field
> (PR4–PR5 fixed the YAML path via `scrubbed_for_commit()`, but the
> JSON bundle wasn't going through it). Fixed in `src/ragdx/cli/tune.py`
> the same day — `base_config` is now serialized via
> `rag_config.scrubbed_for_commit().model_dump(...)`. The file in
> this directory is the post-fix shape; a regression test
> (`test_tune_bundle_base_config_is_scrubbed` in
> `tests/test_pr4_evaluate_tune.py`) verifies the source path stays
> scrubbed. **Always grep your tune bundles for secrets before
> committing them**, especially after upgrading ragdx.

---

## Scenario D — Re-score the optimized config + link as delta

**When to use it:** to confirm the tuned config behaves as the tune
report claims, and to feed the comparison view in the dashboard.

```bash
PYTHONPATH=src python -m ragdx.cli evaluate \
    --config new_demo3/rag_config.optimized.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/D_optimized.json \
    --save --name "demo3-optimized" \
    --baseline-run-id 46c0fcbe5954
```

Output (full log in [`D_console.log`](D_console.log)):

```text
Wrote new_demo3\D_optimized.json
  Retrieval: {"context_precision": 0.5556}
  Generation: {"faithfulness": 1.0}
  E2E: {}
Saved run: 5fe0a5837a23
```

Identical scores to Scenario B — expected, because `top_k` is
unchanged. SavedRun **`5fe0a5837a23`** is the third run; the
RunStore now has the full A → C → D chain.

Note: because the optimized config has `api_key: null` (scrubbed),
this command relied on the `ZHIPU_API_KEY` env var to resolve the
credential. That's the intended workflow: scrub before commit, then
re-hydrate from env / `--api-key` at runtime.

---

## Scenario E — RunStore visibility (`runs`, `compare`, `export-report`)

**When to use it:** review or share results.

### `ragdx runs`

```bash
PYTHONPATH=src python -m ragdx.cli runs
```

Output (full log in [`E_runs.log`](E_runs.log)):

```text
                               Saved ragdx runs
+-----------------------------------------------------------------------------+
| Run ID    | Created | Name              | Baseline    | Sess | Feedback     |
|-----------+---------+-------------------+-------------+------+--------------|
| 5fe0a583… | 2026-06 | demo3-optimized   | 46c0fcb…    |      | 0            |
| 77e06e2e… | 2026-06 | demo3-retrieval…  | 46c0fcb…    |      | 0            |
| 46c0fcbe… | 2026-06 | demo3-baseline    |             |      | 0            |
| 60f9431…  | 2026-05 | esg-baseline      |             |      | 0  (e2e,esg) |
+-----------------------------------------------------------------------------+
```

All three demo3 runs are visible, and the **Baseline** column shows
C and D linked back to B (`46c0fcbe…`). Without `--save` they'd be
absent.

### `ragdx compare` (standalone, no RunStore needed)

```bash
PYTHONPATH=src python -m ragdx.cli compare \
    new_demo3/D_optimized.json new_demo3/B_baseline_saved.json
```

Output ([`D_compare.log`](D_compare.log)):

```text
                       Metric comparison
+--------------------------------------------------------------+
| Metric            | Current | Baseline | Delta   | Direction |
|-------------------+---------+----------+---------+-----------|
| context_precision | 0.5556  | 0.5556   | +0.0000 | unchanged |
| faithfulness      | 1.0000  | 1.0000   | +0.0000 | unchanged |
+--------------------------------------------------------------+
```

`compare` works on any two `EvaluationResult` JSONs — they don't
have to be in the RunStore. Useful for CI scripts.

### `ragdx export-report`

```bash
PYTHONPATH=src python -m ragdx.cli export-report 5fe0a5837a23 new_demo3/E_report.md
```

Output: a Markdown summary of Scenario D's run
([`E_report.md`](E_report.md)) including the diagnosis
(`retrieval_precision_defect`, posterior 0.97), causal signals, and
the auto-generated optimization plan. Suitable for sharing in a PR
comment or Slack.

---

## File map

| File | Origin | Purpose |
|---|---|---|
| `rag_config.yaml` | hand-written | Baseline RAGConfig (input). |
| `questions.jsonl` | hand-written | 3-question eval suite (input). |
| `A_baseline.json` | Scenario A `--output` | EvaluationResult, not persisted. |
| `A_console.log` | Scenario A stdout/stderr | Reference log. |
| `B_baseline_saved.json` | Scenario B `--output` | EvaluationResult, also persisted as run `46c0fcbe5954`. |
| `B_console.log` | Scenario B stdout/stderr | Reference log. |
| `C_tune_retrieval.json` | Scenario C `--output` | Tune bundle with 4 BO trials. |
| `rag_config.optimized.yaml` | Scenario C `--write-optimized-config` | Winning config (credentials scrubbed). |
| `C_console.log` | Scenario C stdout/stderr | Reference log including the BO trial sequence. |
| `D_optimized.json` | Scenario D `--output` | EvaluationResult for the tuned config, persisted as run `5fe0a5837a23` linked to baseline B. |
| `D_console.log` | Scenario D stdout/stderr | Reference log. |
| `D_compare.log` | Scenario E `ragdx compare` | Side-by-side delta table. |
| `E_runs.log` | Scenario E `ragdx runs` | RunStore listing. |
| `E_export.log` | Scenario E export confirmation | `Wrote ...` line. |
| `E_report.md` | Scenario E `ragdx export-report` | Markdown report of run `5fe0a5837a23`. |

## Wall-clock budget

The full run (Scenarios A → E, with 3 questions × `chunk_size=512`)
took **~17 minutes** including PDF chunking and FAISS indexing.
Breakdown:

- A: ~1 min (1 evaluate)
- B: ~1 min (1 evaluate + diagnose; diagnose is cheap)
- C: ~6 min (4 BO trials × ~90s each)
- D: ~1.5 min (1 evaluate + diagnose, slightly slower due to ragas pace)
- E: a few seconds total (RunStore reads, markdown formatting)

If you want a faster smoke test, drop the question set to 1 record
and `--budget 2`. If you want a more realistic budget, push to
~20 questions and `--budget 12`.

## Picking the right scenario

| You want to… | Run this |
|---|---|
| Score a YAML quickly, no side effects | **A** |
| Score + diagnose + persist for review | **B** |
| Improve `top_k` on an already-decent config | **C** with `--stage retrieval` |
| Sweep chunker (slower, vstore rebuilt per trial) | **C** with `--stage chunking` |
| Sweep all three knobs jointly (matches `ragdx experiment`) | **C** with `--stage joint` |
| Optimize the system prompt via DSPy MIPROv2 | **C** with `--stage generation` |
| Compare two configs without persisting | **A** twice → `compare` |
| Compare two runs that are already in the RunStore | dashboard (the Compare tab) |

## Reproducing this demo

```bash
git clone https://github.com/gengzll/RAGdx.git && cd RAGdx
pip install -e ".[langchain,bo,ragas,openai]"
export ZHIPU_API_KEY=<your-key>
# Re-run scenarios A → E using the commands in this README.
# Run IDs will differ; everything else (metric values, control flow)
# should match.
```

For the design rationale + every flag's contract see
[`docs/12-evaluate-tune-workflow.md`](../docs/12-evaluate-tune-workflow.md).
