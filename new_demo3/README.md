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
    --mipro-auto medium \
    --dspy-metric ragas \
    --output new_demo3/G_tune_generation.json \
    --save --name "demo3-generation-tune-v2"
```

The two new (PR6+) flags that make MIPROv2 productive in no-GT mode:

- `--mipro-auto medium` — bump from `light` (3 candidates) to ~10-20
  proposed instructions. Gives the grounded proposer more room to
  beat the seed.
- `--dspy-metric ragas` — replace DSPy's built-in single-call
  `FaithfulnessJudge` metric (which **saturates at 1.0** on permissive
  judge LMs like GLM-4-Flash, so MIPROv2 has no discriminative
  signal and falls back to the seed) with the **ragas composite**
  (context_precision + faithfulness + answer_relevancy as a weighted
  sum). The first run on demo3 with the default `llm_judge` metric
  produced trial scores `[100.0, 100.0, ..., 100.0]` — every
  candidate tied → MIPROv2 returns the seed by tie-break rule
  (strict `>` in `mipro_optimizer_v2.py:579`). With `--dspy-metric
  ragas` the same data produces `[240.97, 247.27, 247.45, ...,
  249.7, ...]` and MIPROv2 actually picks a non-seed winner.

### What MIPROv2 actually did (full per-candidate evidence)

The 2026-06-02 run with `--mipro-auto medium --dspy-metric ragas`:

1. **Step 1 — Bootstrap few-shot examples**: built **12** candidate
   demo sets from successful baseline answers (vs. 6 in `light`).
2. **Step 2 — Propose instructions**: the grounded proposer LLM
   generated **6 distinct candidate `system_instruction`s** captured
   in `dspy_a_b.no_gt.proposed_instructions_by_predictor`:

   | # | First 80 chars | Note |
   |---|---|---|
   | 0 | `You are an ESG sustainability analyst reviewing ASMPT's 2024 report...` | seed |
   | 1 | `You are an electronics manufacturing process analyst tasked with interpreting...` | |
   | 2 | `In the role of an electronics manufacturing process analyst, provide a detailed analysis...` | |
   | 3 | `Prompt the Language Model to act as an ESG sustainability analyst...` | **★ winner** |
   | 4 | `Develop an instruction that prompts a Language Model to act as a sustainability consultant...` | |
   | 5 | `Prompt the Language Model to act as an ESG sustainability analyst, focusing on comprehensive...` | |

3. **Step 3 — Bayesian search**: 13 of 18 planned minibatch trials
   completed before the run was aborted (see Salvage box below). Each
   row is one trial choosing an `(Instruction, Few-Shot Set)` combo.
   **Note the scores are not all tied** — exactly the discriminative
   signal `--dspy-metric ragas` was supposed to produce:

   | Trial | Score | (Instruction, Few-Shot Set) |
   |---|---|---|
   | 1 | **240.97** | seed (Instr 0, Set 0) — default |
   | 2 | 247.27 | (Instr 1, Set 6) |
   | 3 | 247.45 | (Instr 5, Set 11) |
   | 4 | 240.97 | (Instr 2, Set 8) |
   | 5 | 247.75 | (Instr 1, Set 9) |
   | 6 | 235.25 | (Instr 4, Set 6) |
   | 7 | 247.45 | (Instr 5, Set 11) |
   | 8 | 247.55 | (Instr 2, Set 0) |
   | 9 | 220.88 | (Instr 3, Set 7) |
   | **10** | **249.7** | **(Instr 3, Set 10) ★ best** |
   | 11 | 247.45 | (Instr 5, Set 11) |
   | 12 | 247.75 | (Instr 1, Set 9) |
   | 13 | 247.75 | (Instr 1, Set 9) |

   Score range: **220.88 — 249.7** = real spread of 28.82 points.
   Trial 10 (Instr 3 + Set 10) scored **strictly higher** than the
   default (240.97), so MIPROv2's `if score > best_score` check
   triggers and **`best_program` becomes Instruction 3** — **not the
   seed**.

### Did the prompt actually change? Yes — finally

| | Baseline (Instr 0) | Winner (Instr 3) |
|---|---|---|
| Length | 68 chars | 532 chars |
| Persona | "ESG sustainability analyst" | "ESG sustainability analyst" |
| Method | "Quote exact wording when possible" | **"provide a detailed reasoning process"** |
| Discipline | "If not in context, reply 'Not in report'" | "**Ensure step-by-step**, when quoting use exact wording" |

The winning prompt adds **chain-of-thought-style reasoning** and
explicit step-by-step structure. Substantively different content.

### Heavy budget breakthrough (2026-06-02): no regression, real win

Re-running with `--mipro-auto heavy --dspy-metric ragas` produced
**28 trials** (vs medium's 13) and found a winner that's both
strictly higher than the seed at minibatch eval AND matches the
seed on full ragas eval — no minibatch overfit this time:

```bash
ragdx tune --stage generation --from-run d84762c9da44 \
    --mipro-auto heavy --dspy-metric ragas \
    --output new_demo3/G_heavy_tune.json --save --name "demo3-generation-heavy"
```

Bundle: [`G_heavy_tune.json`](G_heavy_tune.json) · HTML:
[`G_heavy_report.html`](G_heavy_report.html) (38 KB). Saved run
`9fd2d342acef`. Full console log:
[`G_heavy_console.log`](G_heavy_console.log).

**Result**:

| Metric | Baseline (Instr 0) | Heavy winner | Δ |
|---|---|---|---|
| context_precision | 0.5556 | 0.5556 | 0.0 |
| faithfulness | **1.0** | **1.0** | **0.0** |
| composite (weighted) | 1.6667 | 1.6667 | 0.0 |

**`Best score so far: 247.75`** (vs default's `240.97`) — strictly
higher → MIPROv2 picks a non-seed winner on the inner BO loop AND
that winner also matches the baseline on the full ragas re-eval.

**Winning prompt** (539 chars, substantively different from the 189-char seed):

> "You are an ESG sustainability analyst **specializing in the
> analysis of Industry 4.0 concepts and digital transformation in
> electronics manufacturing**. Your task is to review passages from
> ASMPT's 2024 report and **extract meaningful insights**. For each
> question provided, quote exact wording from the context when
> possible, and if the answer is not present in the context, reply
> 'Not in report'. Based on the context, generate a reasoning that
> outlines the **step-by-step thought process** leading to the
> answer, and then provide the answer itself."

Added: **domain specialization** (Industry 4.0 + electronics
manufacturing), **insight extraction**, **explicit reasoning
structure**. Kept: **persona** (ESG analyst), **safety net** ("Not
in report"), **fidelity discipline** (exact wording).

### Three runs at three budgets — the full picture

| Run | budget | trials | Winner | full-eval faithfulness | Wall |
|---|---|---|---|---|---|
| previous-1 | `light` + `llm_judge` | 11 | Instr 0 (seed) — tie-break | 1.0 (no change) | ~12 min |
| previous-2 | `medium` + `ragas` | 13 (of 18) | Instr 3 — strict win on minibatch | **0.91 (regressed!)** | ~75 min |
| **current** | **`heavy` + `ragas`** | **28** (of 27 planned, 1 retry) | New Industry-4.0-themed prompt | **1.0 (no regression)** | **~73 min** |

The pattern:
1. `light` + `llm_judge` couldn't even discriminate candidates.
2. `medium` + `ragas` discriminated, picked a winner, but the
   winner overfit the minibatch (Method F / Hold-out validation
   would have caught this).
3. `heavy` + `ragas` discriminated better (more candidates + more
   trials reduce the per-trial-sampling noise), found a winner
   that doesn't overfit.

**The structural fix to MIPROv2's tie-to-seed problem is `ragas`
metric. The structural fix to minibatch overfit is `heavy` budget
(or full-set eval, or hold-out validation as Method F would
implement).**

Also: **3 ragas calls timed out at 90s during this run** (see "ragas
faithfulness scoring hit 90s timeout" in `G_heavy_console.log`).
Each timeout was treated as score=0 for that one metric/sample
combination (visible as the 172.45 / 165.97 / 172.45 outliers in
`Scores so far`). Without the PR6+ `_PER_CALL_TIMEOUT_S = 90` fix in
`_ragas_dspy_metric.py`, each of those would have hung indefinitely
and a 73-minute run would have become an unfinishable run.

### Honest follow-up finding: winner ≠ better on full eval

We re-evaluated the winner (Instruction 3) on the same 3 questions
via `ragdx evaluate`:

| Metric | Baseline (Instr 0) | Winner (Instr 3) | Δ |
|---|---|---|---|
| context_precision | 0.5556 | 0.5556 | 0.0 (retrieval unchanged) |
| faithfulness | **1.0000** | **0.9111** | **−0.0889** |
| composite (weighted) | 1.6667 | 1.5333 | −0.1333 |

**The winner did slightly worse on full ragas eval.** Why?

MIPROv2 uses **minibatch evaluation** during its inner search — only
2 random sample(s) per trial, not the full 3-question set. Trial 10
saw 2 samples and got 249.7; the 3rd question (left out in trial 10's
minibatch) is the one where Instruction 3's reasoning-style prompt
introduces a slightly less-faithful claim. The minibatch said
"winner", the full eval says "marginally worse".

This is a known minibatch-overfit phenomenon in MIPROv2. To avoid it:
crank `--mipro-auto heavy` (more trials → less minibatch noise) or
use full-set eval (`auto=None` + manual `minibatch=False`).

So the user's original question — "did DSPy actually change the
prompt?" — gets a definitive **yes** answer (Instruction 3 ≠ seed,
text proves it), and a follow-up nuance: the "winner" by MIPROv2's
internal scoring may not be the best choice on full evaluation.

> **⚠ Salvage box.** The 2026-06-02 run was aborted at Trial 14 / 18
> after a 13-minute idle on a GLM HTTP connection (network switch
> mid-run). Trial 14's metric call hit no timeout, so the process
> hung. We:
>
> 1. Killed the stuck process (PID 28408).
> 2. Parsed `G_console.log` to recover **6 proposed instructions**
>    + **13 trial logs**.
> 3. Re-evaluated the **winner Instruction 3** on the 3 questions
>    via `ragdx evaluate` ([`G_winner_eval.json`](G_winner_eval.json)).
> 4. Merged into [`G_tune_generation.json`](G_tune_generation.json)
>    with a `_salvage` block recording the recovery.
> 5. Added `_PER_CALL_TIMEOUT_S = 90` to
>    `src/ragdx/optim/_ragas_dspy_metric.py` so a future hung HTTP
>    call dies after 90 seconds and the run continues. The next
>    `medium + ragas` run will be hang-proof.
>
> All findings above use only data that was actually captured during
> the live run — no fabrication. The remaining 5 trials would have
> sampled more (instruction, demo set) combos but would not have
> changed the structural conclusion that ragas metric breaks the
> tie-to-seed problem.

### Picking a different optimizer (PR7+): COPRO / BootstrapFewShot

`--dspy-optimizer [mipro|copro|bootstrap_fewshot]` switches the
underlying DSPy teleprompter. **Each writes a different shape of
optimized config**, aligned with what it actually optimizes:

| Optimizer | What it optimizes | What writes to `optimized.yaml` |
|---|---|---|
| `mipro` (default) | (instruction × demos) | both `system_instruction` AND `few_shot_demos` |
| `copro` | instruction only | only `system_instruction`; `few_shot_demos: []` |
| `bootstrap_fewshot` | demos only | only `few_shot_demos: [...]`; `system_instruction` unchanged |
| `gepa` (experimental) | instruction only (reflective evolution) | only `system_instruction`; `few_shot_demos: []` |

This closes the architecture gap noted earlier: previously every
run wrote only `system_instruction` even when the optimizer found
demos. Now the schema field that gets populated reflects what the
algorithm actually evolved.

#### Side-by-side: same baseline, four optimizers

| Run | Optimizer | Wall | What was written | Outcome |
|---|---|---|---|---|
| `9fd2d342acef` | `mipro` heavy | 73 min | `system_instruction` (new Industry-4.0 prompt) + `few_shot_demos` populated | full eval matched baseline (1.0 / 1.0) |
| `7083a760fdfb` | `copro` | 17 min | only `system_instruction` (returned seed — no improvement) | no change |
| `9797f4e99a43` | `bootstrap_fewshot` | 9 min | only `few_shot_demos: [3 demos]` | demos available at deploy time |
| `8c4204179655` | `gepa` light (30 calls) | 53 min | only `system_instruction` (new 2136-char reflective prompt vs 189-char seed) | minibatch winner; full eval faithfulness 1.0 → 0.958 (minor regression) |

Inspect the three `rag_config.*_winner.yaml` files in this directory
to see exactly what each optimizer wrote.

#### When to use which

| Production deployment | Optimizer |
|---|---|
| Static system prompt, no few-shot | **`copro`** — fastest, single field to deploy |
| Static prompt, dynamic few-shot examples | **`bootstrap_fewshot`** — produces a clean demos list |
| Both knobs available, want max performance | **`mipro`** (default) — searches the full grid |
| Want reflective per-trace LLM analysis | **`gepa`** — Pareto evolution + trace reflection (experimental; requires `gepa` package) |

> **GEPA budget note.** DSPy GEPA's `auto="light"` plans ~392 rollouts —
> reasonable on GPT-4 / Claude but produces 16+ hour runs on
> rate-limited LMs like GLM-4-Flash. ragdx translates
> `--mipro-auto {light, medium, heavy}` into explicit
> `max_metric_calls = {30, 100, 300}` so production iteration stays
> bounded. Power users can edit the mapping in
> `src/ragdx/optim/stages/generation.py`.

The renderer (`ragdx experiment-report`) handles all four bundle
shapes — see `G_copro_report.html` (18 KB), `G_bootstrap_report.html`
(~21 KB), `G_gepa_report.html` (~20 KB), `G_heavy_report.html` (38 KB).

### Visualizing the DSPy A/B as HTML

```bash
PYTHONPATH=src python -m ragdx.cli experiment-report \
    new_demo3/G_tune_generation.json \
    --output new_demo3/G_tune_report.html \
    --title "demo3 Scenario G: DSPy MIPROv2 prompt tune"
```

Output: [`G_tune_report.html`](G_tune_report.html) (~30 KB). The
HTML has dedicated sections for:

- **DSPy before/after bar chart** — baseline (Instr 0) vs optimized
  (Instr 3) ragas scores, plus composite Δ.
- **Candidate instructions MIPROv2 proposed** — all **6** candidates
  side-by-side as expandable blocks; **Instruction 3 marked with ★**
  as winner.
- **Trial-by-trial decisions** — table of 13 completed trials with
  the chosen (Instruction, Few-Shot Set) tuple and score per trial.
  Scores span 220.88 — 249.7 (no tie).
- **Prompts: baseline vs optimized** — text diff between Instr 0
  and Instr 3 side-by-side.

Same renderer as `ragdx experiment` bundles' DSPy section, plus the
candidate-trace UI added in this commit.

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
