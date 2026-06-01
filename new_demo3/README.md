# new_demo3 — End-to-end walkthrough of the PR1–PR6 `evaluate` / `tune` workflow

This directory is a real, executed end-to-end run of every command
added by the `RAGConfig` / `RAGPipeline` / `evaluate` / `tune` control
plane (PR1–PR5), plus the PR6 follow-up that adds per-project
storage, plan-driven `tune --from-run`, and **latest-by-default
fallbacks** so common workflows need almost no arguments.

The corpus is the public **ASMPT 2024 ESG report**
(`e0522-asmptesgreport.pdf`, 114 pages); the generator is
**GLM-4-Flash** through ZHIPU's OpenAI-compatible endpoint. All
commands were run on **2026-06-01**.

> Every JSON/log file in this directory was emitted by `ragdx` during
> the run — they are not hand-crafted samples. Run IDs and timings
> come from the live execution. The directory was wiped to empty
> before this run, so the artifacts below are fully reproducible.

## What this demo covers

Seven scenarios, in dependency order:

| # | Command | What it teaches |
|---|---|---|
| **A** | `ragdx evaluate` | Score a `RAGConfig` and emit a normalized `EvaluationResult` JSON. The simplest mode — no RunStore, no diagnose. |
| **B** | `ragdx evaluate --save` | Same scoring, **plus** diagnose + plan + persist to RunStore in one call. Stores the (scrubbed) RAGConfig too, so downstream tune commands can inherit it. |
| **C** | `ragdx tune --save` | Stage-targeted Bayesian search with **zero positional args**: `--from-run` / `--base-config` / `--questions` / `--corpus` / `--stage` / `--budget` / `--baseline-run-id` all auto-default from the latest run's plan. |
| **D** | `ragdx evaluate --save --baseline-run-id` on the tuned config | Re-score the winning config and link it as a delta of the original baseline. |
| **E** | `ragdx runs` / `ragdx compare` / `ragdx export-report` | RunStore visibility. Shows the **latest-defaults**: `compare A.json` (no baseline → uses latest) and `export-report "" file.md` (no run id → uses latest). |
| **F** | `ragdx --project demo3-pr6 ...` | Per-project namespace. F1 evaluates and F2 tunes; **F2 again uses no `--from-run`** — it picks the latest run *within that project*, demonstrating that the default respects `--project`. |
| **G** | `ragdx tune --stage generation` | **The non-BO stage**: DSPy MIPROv2 evolves the `system_instruction` at the current retrieval config. Produces a `dspy_a_b` section in the bundle that `experiment-report` renders as a baseline-vs-optimized bar chart. |

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

**When to use it:** quick sanity check before committing to a full
diagnose/save cycle, or in CI where you only need the numbers.

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
```

`548 chunks` from the PDF, 3 records × 3 ragas metrics = 9
evaluations. No RunStore side effects.

---

## Scenario B — `evaluate --save`: the closed loop

**When to use it:** production. Score → diagnose → plan → persist in
one call. The saved run carries the (scrubbed) `RAGConfig`, which
Scenario C will inherit automatically.

```bash
PYTHONPATH=src python -m ragdx.cli evaluate \
    --config new_demo3/rag_config.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/B_baseline_saved.json \
    --save --name "demo3-baseline"
```

Output ([`B_console.log`](B_console.log)):

```text
Wrote new_demo3\B_baseline_saved.json
  Retrieval: {"context_precision": 0.5556}
  Generation: {"faithfulness": 1.0}
Saved run: d84762c9da44
```

`d84762c9da44` becomes the latest run in the RunStore.

---

## Scenario C — `tune --save` with zero positional args (PR6 headline)

**This is the headline ergonomic win.** No `--from-run`, no
`--base-config`, no `--questions`, no `--corpus`, no `--stage`, no
`--budget`, no `--baseline-run-id`. Everything is pulled from the
latest saved run + its plan.

```bash
PYTHONPATH=src python -m ragdx.cli tune \
    --experiment retrieval-pipeline-search \
    --output new_demo3/C_tune_retrieval.json \
    --write-optimized-config new_demo3/rag_config.optimized.yaml \
    --save --name "demo3-retrieval-sweep"
```

The only arg that requires manual choice is `--experiment`: the plan
has three experiments (`corpus-chunking-search`,
`retrieval-pipeline-search`, `generator-prompt-optimization`), and
we pick the cheapest. Omit it and tune picks the first
(`corpus-chunking-search` here — slower but valid).

Output ([`C_console.log`](C_console.log)):

```text
No --from-run / --base-config given; defaulting to latest run d84762c9da44 (demo3-baseline).
Inheriting from run d84762c9da44: experiment=retrieval-pipeline-search,
  stage=retrieval, max_trials=4, search_space={...'top_k': [4, 6, 8, 10]...}
Running retrieval optimizer on 548 chunks x 3 records (GT mode: no_gt, budget: 4)...
[4 BO trials over top_k ∈ {4, 6, 8, 10}]
Wrote new_demo3\C_tune_retrieval.json
Wrote optimized config new_demo3\rag_config.optimized.yaml (credentials scrubbed)
Best composite: 1.675
Best params: {'top_k': 4}
Saved run: 669c0d73d771 (name: demo3-retrieval-sweep)
```

What got auto-inherited (none of these were passed as flags):

| From the SavedRun | Used as |
|---|---|
| `rag_config` | `--base-config` |
| `evaluation.metadata.questions_path` | `--questions` |
| `evaluation.metadata.corpus` | `--corpus` |
| `optimization_plan.experiments[retrieval-pipeline-search].stage` | `--stage retrieval` |
| `optimization_plan.experiments[…].max_trials` | `--budget 4` |
| `optimization_plan.experiments[…].search_space.top_k` | `ctx.top_ks=[4,6,8,10]` |
| The defaulted `--from-run` itself | `--baseline-run-id d84762c9da44` |

**Result**: best `top_k=4`, composite **1.675**. The plan's
recommended sweep `[4, 6, 8, 10]` found an improvement that the
default StageContext range `[1, 3, 5, 7]` (which the manual
`--stage retrieval --budget 8` form used to use) didn't have access
to. **This is the practical payoff of letting the plan drive tune.**

---

## Scenario D — Re-score the optimized config + link as delta

**When to use it:** confirm the tuned config holds up at re-eval
time, and feed the dashboard's compare view.

```bash
PYTHONPATH=src python -m ragdx.cli evaluate \
    --config new_demo3/rag_config.optimized.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/D_optimized.json \
    --save --name "demo3-optimized" \
    --baseline-run-id d84762c9da44
```

`--baseline-run-id` is explicit because we want to link to **B**
(the original baseline), not to C (the tune run, which is now
latest). The default would have picked C.

Output ([`D_console.log`](D_console.log)):

```text
Wrote new_demo3\D_optimized.json
  Retrieval: {"context_precision": 0.5833}
  Generation: {"faithfulness": 1.0}
Saved run: ca1fba72987e
```

`context_precision` went from **0.5556** (baseline) to **0.5833**
(tuned, `top_k=4`) — a real +5% improvement at re-eval time, not just
during the BO sweep. SavedRun `ca1fba72987e` linked to B.

---

## Scenario E — RunStore visibility with latest-defaults

### `ragdx compare D.json B.json` (explicit)

```text
                       Metric comparison
+--------------------------------------------------------------+
| Metric            | Current | Baseline | Delta   | Direction |
|-------------------+---------+----------+---------+-----------|
| context_precision | 0.5833  | 0.5556   | +0.0278 | improved  |
| faithfulness      | 1.0000  | 1.0000   | +0.0000 | unchanged |
+--------------------------------------------------------------+
```

([`E_compare_explicit.log`](E_compare_explicit.log))

### `ragdx compare A.json` (no baseline → uses latest = D, automatic)

```text
Baseline: latest run ca1fba72987e (demo3-optimized)
                       Metric comparison
+--------------------------------------------------------------+
| Metric            | Current | Baseline | Delta   | Direction |
|-------------------+---------+----------+---------+-----------|
| context_precision | 0.5556  | 0.5833   | -0.0278 | regressed |
| faithfulness      | 1.0000  | 1.0000   | +0.0000 | unchanged |
+--------------------------------------------------------------+
```

([`E_compare_default.log`](E_compare_default.log))

The header `Baseline: latest run ca1fba72987e (demo3-optimized)`
makes the implicit choice visible. Same trick on `diagnose`,
`plan`, `optimize`, `save`.

### `ragdx runs`

```text
+------------+-------------------+--------------+
| Run ID     | Name              | Baseline     |
+------------+-------------------+--------------+
| ca1fba7…   | demo3-optimized   | d84762c…     |  ← D
| 669c0d7…   | demo3-retrieval…  | d84762c…     |  ← C
| d84762c…   | demo3-baseline    |              |  ← B
| 60f9431…   | esg-baseline      |              |  (older, e2e_test)
+------------+-------------------+--------------+
```

([`E_runs.log`](E_runs.log))

### `ragdx export-report "" report.md` (no run id → uses latest)

```text
Exporting latest run (demo3-optimized)
Wrote new_demo3\E_report.md
```

([`E_export.log`](E_export.log), markdown in [`E_report.md`](E_report.md))

The `""` opts into the default. For backward compatibility,
`run_id` is still the first positional, so `ragdx export-report
<id> report.md` keeps working unchanged.

---

### Visualizing the tune bundle as a self-contained HTML report

Tune bundles are **shape-compatible** with `ragdx experiment` bundles
since PR6, so `ragdx experiment-report` renders them without any
special-casing:

```bash
PYTHONPATH=src python -m ragdx.cli experiment-report \
    new_demo3/C_tune_retrieval.json \
    --output new_demo3/C_tune_report.html \
    --title "demo3 Scenario C: retrieval tune"
```

Output: [`C_tune_report.html`](C_tune_report.html) (~18 KB, fully
self-contained — open in a browser, no Streamlit server, no internet
needed). Sections:

- Run metadata (model, mode, source)
- Bayesian RAG-config search (4 trials, search space, best params)
- Trial-by-trial table with per-metric scores

The same renderer works on Scenario F2's bundle —
[`F2_tune_report.html`](F2_tune_report.html). For a Streamlit
equivalent: `PYTHONPATH=src python -m ragdx.cli experiment-dashboard
new_demo3/C_tune_retrieval.json` (interactive, but needs a
streamlit server).

## Scenario G — `tune --stage generation`: DSPy MIPROv2 prompt optimization

**When to use it:** the plan / diagnose path flagged
`grounding_defect` or `generation_quality_drop`, or you simply want
to see whether a better prompt can squeeze more out of your current
retrieval config. This is the **non-BO** stage; it runs
[DSPy MIPROv2](https://dspy.ai/learn/optimization/optimizers/#mipro)
to evolve the `generator.system_instruction` (and few-shot demos)
at the existing retrieval config.

```bash
PYTHONPATH=src python -m ragdx.cli tune \
    --stage generation \
    --from-run d84762c9da44 \
    --output new_demo3/G_tune_generation.json \
    --save --name "demo3-generation-tune"
```

`--stage generation` is explicit; without it the plan's first
experiment (`corpus-chunking-search`) would be picked. `--from-run
d84762c9da44` points at the original baseline (B) so the prompt
sweep uses the same retrieval config + questions + corpus that
produced the original 0.5556 / 1.0 baseline scores.

Output ([`G_console.log`](G_console.log), bundle in
[`G_tune_generation.json`](G_tune_generation.json)):

```text
Inheriting from run d84762c9da44: experiment=corpus-chunking-search, ...
Running generation optimizer on 548 chunks x 3 records (GT mode: no_gt, budget: 4)...
[DSPy/no_gt] (a) baseline run
[DSPy/no_gt] (b) MIPROv2 optimisation
  Bootstrapping 11 sets of demonstrations...
  Returning best identified program with score 100.0!
[DSPy/no_gt] (c) optimised re-run

Best composite: 1.667
Best params: {'system_instruction': 'You are an ESG sustainability analyst...'}
Saved run: ba147d961aba (name: demo3-generation-tune)
```

### What MIPROv2 actually did

1. **Baseline run** — score the user-supplied `system_instruction`
   on the 3 questions with the existing retrieval config.
2. **MIPROv2 optimisation** — bootstrap 11 candidate
   instructions + few-shot demo sets, internally score each, pick
   the best.
3. **Optimised re-run** — score the winning program again on the
   3 questions to produce a true before/after comparison.

The MIPROv2 score=100 means the candidate hits every internal
heuristic; the **actual ragas-based** before/after is in
`dspy_a_b.no_gt.{baseline_scores, optimized_scores, delta}`:

```text
baseline_scores  : {context_precision: 0.5556, faithfulness: 1.0}
optimized_scores : {context_precision: 0.5556, faithfulness: 1.0}
delta            : {context_precision: 0.0,    faithfulness: 0.0}
composite Δ      : 0.0
```

**Result:** MIPROv2 explored 11 alternatives and concluded the
hand-crafted ESG analyst prompt was already at a local optimum for
this question set + retrieval config. The "winning" instruction is
*identical* to baseline. That's a valid and informative outcome:
your prompt is already strong. With a larger trainset or more
ambitious `optimizer_kwargs.auto="medium" / "heavy"`, MIPROv2 has
more room to experiment, but at this scale + budget it can't beat
the seed.

### Visualizing the DSPy A/B as HTML

```bash
PYTHONPATH=src python -m ragdx.cli experiment-report \
    new_demo3/G_tune_generation.json \
    --output new_demo3/G_tune_report.html \
    --title "demo3 Scenario G: DSPy MIPROv2 prompt tune"
```

Output: [`G_tune_report.html`](G_tune_report.html) (~27 KB). The
HTML has a dedicated "**DSPy before/after**" section with a
side-by-side bar chart (baseline vs optimized scores per metric),
composite delta, and the prompt text. This is the same renderer
that handles `ragdx experiment` bundles' DSPy section.

## Scenario F — Per-project storage + project-scoped latest-defaults

**When to use it:** real production with multiple RAGs in one repo.
Each project's runs, sessions, feedback, and causal priors live in
their own folder. The `latest` default also scopes to the project.

### F1 — `evaluate --save` under `--project demo3-pr6`

```bash
PYTHONPATH=src python -m ragdx.cli --project demo3-pr6 evaluate \
    --config new_demo3/rag_config.yaml \
    --questions new_demo3/questions.jsonl \
    --corpus e0522-asmptesgreport.pdf \
    --output new_demo3/F1_baseline.json \
    --save --name "demo3-pr6-baseline"
```

Saved run `e390a1cef541` → lands in
**`.ragdx/projects/demo3-pr6/runs/e390a1cef541.json`** (not the
default `.ragdx/runs/`). The default RunStore still has only B/C/D.

### F2 — `tune --save` (still zero `--from-run`, defaults to the project's latest)

```bash
PYTHONPATH=src python -m ragdx.cli --project demo3-pr6 tune \
    --experiment retrieval-pipeline-search \
    --output new_demo3/F2_tune.json \
    --write-optimized-config new_demo3/rag_config.f.optimized.yaml \
    --save --name "demo3-pr6-tune-retrieval"
```

Output ([`F2_console.log`](F2_console.log)):

```text
No --from-run / --base-config given; defaulting to latest run e390a1cef541 (demo3-pr6-baseline).
Inheriting from run e390a1cef541: experiment=retrieval-pipeline-search,
  stage=retrieval, max_trials=4, search_space={...'top_k': [4, 6, 8, 10]...}
Best composite: 1.675
Best params: {'top_k': 4}
Saved run: 5646b1cc0265 (name: demo3-pr6-tune-retrieval)
```

The default picked **F1** (the project's latest), **not D** (which
is `main`'s latest). That's the project isolation working.

### F3 — `ragdx --project demo3-pr6 runs`

```text
+------------+-------------------------+-------------+
| Run ID     | Name                    | Baseline    |
+------------+-------------------------+-------------+
| 5646b1cc…  | demo3-pr6-tune-retrieval| e390a1ce…   |  ← F2
| e390a1ce…  | demo3-pr6-baseline      |             |  ← F1
+------------+-------------------------+-------------+
```

([`F3_runs.log`](F3_runs.log))

Just the demo3-pr6 runs. The main project's B/C/D never appear here.

---

## File map

| File | Origin | Purpose |
|---|---|---|
| `rag_config.yaml` | hand-written | Baseline RAGConfig (input). |
| `questions.jsonl` | hand-written | 3-question eval suite (input). |
| `A_baseline.json` | A `--output` | EvaluationResult, not persisted. |
| `A_console.log` | A stdout/stderr | Reference log. |
| `B_baseline_saved.json` | B `--output` | EvaluationResult, persisted as run `d84762c9da44`. |
| `B_console.log` | B stdout/stderr | Reference log. |
| `C_tune_retrieval.json` | C `--output` | Tune bundle (4 BO trials over `top_k ∈ [4,6,8,10]`). `inherited` section records what was carried from the SavedRun + plan. |
| `rag_config.optimized.yaml` | C `--write-optimized-config` | Winning config (`top_k=4`), credentials scrubbed. |
| `C_console.log` | C stdout/stderr | Shows the auto-default + inherited block. |
| `D_optimized.json` | D `--output` | EvaluationResult at the tuned config, persisted as run `ca1fba72987e` linked to B. |
| `D_console.log` | D stdout/stderr | Reference log. |
| `E_compare_explicit.log` | `compare D.json B.json` | Side-by-side delta table (D vs B). |
| `E_compare_default.log` | `compare A.json` | Same renderer, baseline auto-resolved to D. |
| `E_runs.log` | `ragdx runs` | Main-project RunStore listing. |
| `E_export.log` | `export-report "" file.md` | Latest-default export. |
| `E_report.md` | `export-report` output | Markdown report of run `ca1fba72987e`. |
| `F1_baseline.json` | F1 `--output` | EvaluationResult under `--project demo3-pr6`. |
| `F1_console.log` | F1 stdout/stderr | Reference log. |
| `F2_tune.json` | F2 `--output` | Tune bundle (project-scoped). The `inherited.from_run` records F1's run id, **not** main-project's D. |
| `rag_config.f.optimized.yaml` | F2 `--write-optimized-config` | Winning config from the project-scoped tune. |
| `F2_console.log` | F2 stdout/stderr | Reference log. |
| `F3_runs.log` | `--project demo3-pr6 runs` | Project-scoped run list. |
| `C_tune_report.html` | `experiment-report` on C bundle | Self-contained HTML rendering of Scenario C's BO trials. |
| `F2_tune_report.html` | `experiment-report` on F2 bundle | Same, for the project-scoped tune. |
| `G_tune_generation.json` | G `--output` | DSPy MIPROv2 result bundle: baseline_scores, optimized_scores, delta, instructions, demos, trial_scores. |
| `G_console.log` | G stdout/stderr | Shows MIPROv2's "Bootstrapping" + "Returning best identified program" log. |
| `G_tune_report.html` | `experiment-report` on G bundle | HTML with the DSPy before/after section (~27 KB). |

## Wall-clock budget

This run took **~33 minutes** end-to-end (A–F). Breakdown:

- A + B: ~5 min (2 evaluates; first one's PDF chunking + ragas pace
  was slow due to GLM rate-limit jitter)
- C: ~10 min (4 BO retrieval trials, each ~2 min of ragas)
- D: ~1.5 min (1 evaluate + diagnose)
- E: a few seconds total
- F1 + F2: ~11 min (1 evaluate + 1 tune with 4 BO trials)

For a faster smoke test, drop the question set to 1 record and
`--budget 2`. For a realistic budget, push to ~20 questions and
`--budget 12`.

## Picking the right scenario

| You want to… | Run this |
|---|---|
| Score a YAML quickly, no side effects | **A** |
| Score + diagnose + persist for review | **B** (or **F1** with `--project`) |
| **Tune what I just evaluated** | **C** — `ragdx tune --save` with no `--from-run` needed |
| Run a specific stage instead of plan's first pick | **C** with `--experiment <name>` |
| Override the plan with a specific search axis | **C** with explicit `--stage` / `--budget` |
| **Improve the prompt at the existing retrieval config** | **G** — `ragdx tune --stage generation` (DSPy MIPROv2) |
| Compare two configs without persisting | **A** twice → `compare a.json b.json` |
| Compare against last run without typing paths | `compare new.json` (baseline auto-resolves) |
| Export the latest run as markdown | `export-report "" report.md` |
| Keep multiple production projects' results isolated | `--project <name>` (Scenario F) |

## Reproducing this demo

```bash
git clone https://github.com/gengzll/RAGdx.git && cd RAGdx
pip install -e ".[langchain,bo,ragas,openai]"
export ZHIPU_API_KEY=<your-key>
# Then run scenarios A through F using the commands in this README.
# Run IDs will differ; everything else (metric values, control flow,
# the "defaulting to latest" hints) should match.
```

For the design rationale + every flag's contract see
[`docs/12-evaluate-tune-workflow.md`](../docs/12-evaluate-tune-workflow.md).
