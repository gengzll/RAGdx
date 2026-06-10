# ragdx demos

Two end-to-end walkthroughs that reflect **two different real
workflows**, both run against the public **ASMPT 2024 ESG report**
(`../e0522-asmptesgreport.pdf`, 114 pages) with **GLM-4-Flash** as the
generator. Every JSON / YAML / HTML / log in these folders was produced
by `ragdx` during the run — real artifacts, not hand-crafted samples.

| folder | The workflow it shows | Command surface |
|---|---|---|
| [`end2end_demo/`](end2end_demo) | **"just optimize my RAG."** Run BO → DSPy prompt tuning → diagnose-at-the-end, all in one command, one bundle, one report. **No upfront baseline / diagnosis.** | `ragdx experiment` |
| [`diagnosis_demo/`](diagnosis_demo) | **"diagnose first, then fix the specific bottleneck, then check it worked."** eval → diagnose → targeted tune → re-eval → re-diagnose (escalates) → report. | `ragdx workspace …` |

The distinction is the point: end2end is the *blind full-optimization*
path; diagnosis is the *diagnose-driven targeted* path. They are not
the same demo with different names.

---

## Prerequisites

```bash
# From the repo root:
pip install -e ".[experiment]"
export ZHIPU_API_KEY=<your-zhipu-key>          # or OPENAI_API_KEY
```

The PDF lives at the repo root. Run all commands from the repo root.

---

## Demo 1: `end2end_demo/` — one-shot full optimization

**Workflow:** you have a RAG and just want it optimized end to end.
`ragdx experiment` does the whole pipeline in one command: load corpus
→ synthesize questions → **Bayesian search over RAG params** → **DSPy
prompt tuning at the BO winner** → diagnose the result. The diagnosis
is computed at the END, from the optimized system — there is no
upfront baseline-and-diagnose ceremony.

```bash
# One command: BO -> DSPy -> diagnosis, written to one bundle.
ragdx experiment e0522-asmptesgreport.pdf \
    --no-gt --n-questions 3 --bo-trials 4 --bo-init 2 \
    --output-dir workspaces/end2end_demo --seed 11 --name end2end_demo

# Render the single unified report.
ragdx experiment-report workspaces/end2end_demo/result.json \
    -o workspaces/end2end_demo/report.html
```

### What you get

| file | what it is |
|---|---|
| `result.json` | ONE bundle with `bayes_search` (BO trials) + `dspy_a_b` (prompt before/after) + `diagnosis` (baseline → optimized → comparison), all from the single run |
| `report.html` | the complete report: run metadata, baseline diagnosis, BO search + per-record metrics, **DSPy answer diff + candidate evolution timeline**, comparison, optimized diagnosis, causal-graph SVG, run cost |

The report's "Diagnosis (baseline)" section here describes the
post-BO-pre-DSPy state, and "Diagnosis (optimized)" + "comparison"
describe the final system — so you read the optimization story
top-to-bottom without ever running a separate diagnose command.

---

## Demo 2: `diagnosis_demo/` — diagnose-driven targeted loop

**Workflow:** you suspect a specific bottleneck, want the tool to
confirm it, fix exactly that, and tell you whether it worked — and if
not, **what to escalate to**. This is where the workspace verbs shine:
short commands, all artifacts in one folder, history that drives the
next step.

```bash
# 0. Create the workspace once (config + questions copied in).
ragdx workspace init diagnosis_demo \
    --config rag_config.demo.yaml \
    --questions questions.demo.jsonl \
    --corpus e0522-asmptesgreport.pdf

# 1. Baseline evaluation.
ragdx workspace eval diagnosis_demo

# 2. Diagnose. History is empty -> base-level advice.
#    Read `optimization_candidates`: ["autorag_pipeline_search"]
#    -> the recommended next step is tune-rag.
ragdx workspace diagnose diagnosis_demo

# 3. Run exactly what the diagnosis pointed at.
ragdx workspace tune-rag diagnosis_demo --stage joint --budget 4

# 4. Re-evaluate at the optimized config.
ragdx workspace eval diagnosis_demo

# 5. Re-diagnose. The workspace history now contains a tune-rag, so
#    if precision is STILL below threshold the advice ESCALATES:
#    "basic precision tuning already ran -> add a cross-encoder
#    reranker / switch to hybrid retrieval (BM25 + dense)".
ragdx workspace diagnose diagnosis_demo

# 6. Render the report (baseline -> comparison -> optimized).
ragdx workspace report diagnosis_demo
```

### What the escalation looks like

| step | summary | candidates / advice |
|---|---|---|
| D2 (baseline diagnose) | "retrieval noise or weak ranking quality" | `autorag_pipeline_search` → "add or tune a reranker; reduce top-k" |
| D5 (re-diagnose, after tune-rag) | **"precision still failing after retrieval tuning — escalate retrieval complexity"** | `autorag_pipeline_search` + `corpus_chunking_search` → **"escalate: add a cross-encoder reranker (not just a score cutoff); switch to hybrid retrieval (BM25 + dense)"** |

The re-diagnosis is **history-aware**: it knows the basic retrieval
tune already ran and didn't fully fix precision, so it climbs the
escalation ladder instead of repeating the same advice.

### What you get

| file | what it is |
|---|---|
| `baseline.eval.json` / `baseline.eval.diagnose.json` | before optimization (base advice) |
| `tune_rag_joint.json` | the BO tune the diagnosis recommended (+ in-bundle baseline/optimized/comparison diagnoses) |
| `rag_optimized.tune_rag.yaml` | the tuned config (api_key scrubbed) |
| `post_1.eval.json` / `post_1.eval.diagnose.json` | after optimization (**escalated** advice) |
| `report.html` | baseline → comparison → optimized diagnoses side-by-side |
| `.ragdx/` | workspace-local store: RunStore, **causal priors**, and **checkpoints** — see below |

---

## Why two different command surfaces?

* `ragdx experiment` is the right tool for **"optimize everything,
  show me the result"** — it's one shot, and the diagnosis is a
  post-hoc summary, not a driver.
* `ragdx workspace` is the right tool for the **diagnose → fix →
  re-check loop**, because each step's output (and the *history* of
  what you've tried) feeds the next. The escalation in step 5 only
  works because the workspace remembers step 3.

## Workspace-local storage (priors + checkpoints)

Each workspace scopes `RAGDX_ROOT` to `<workspace>/.ragdx`, so its
RunStore, **causal priors**, and **checkpoints** all live inside the
workspace folder:

```
workspaces/diagnosis_demo/.ragdx/
├── runs/            # saved EvaluationResult + diagnosis + plan per run
├── causal/priors.json   # causal-graph priors learned from THIS workspace only
└── checkpoints/     # per-trial BO / per-phase DSPy resume state
```

This matters for two reasons:

1. **Causal priors don't leak across experiments.** Each workspace
   starts from the clean `base_priors`, so a fresh diagnose produces
   meaningful posteriors instead of values saturated by every other
   experiment you've ever run. (A global, shared prior store saturates
   every node to ~0.95 after a handful of runs, which made baseline
   and optimized causal graphs look identical — fixed by this scoping.)

2. **Long tunes are resumable.** `tune-rag` / `tune-prompt` checkpoint
   after every BO trial (or DSPy phase). If a run dies mid-way:

   ```bash
   ragdx workspace tune-rag diagnosis_demo --resume auto
   ```

   replays the completed trials and continues. `--no-checkpoint`
   disables it.

## Common workspace verbs

```bash
ragdx workspace list                        # all workspaces
ragdx workspace show <name>                 # one workspace's config + history
ragdx workspace report <name> [--source X]  # render HTML
ragdx workspace compare <a> <b> [c...]      # cross-experiment metric / posterior delta
```

## Reproduce / clean up

```bash
# Re-run both demos from the repo root (≈40 min on GLM-4-Flash):
#   end2end:   the single ragdx experiment command above
#   diagnosis: ragdx workspace init + the 6-step loop above

# Wipe and start over:
rm -rf workspaces/end2end_demo workspaces/diagnosis_demo
```

Credentials are never written to disk — the demos rely on the
`ZHIPU_API_KEY` env var via ragdx's standard fallback chain. Verify
with `grep -r "<your-key-prefix>" workspaces/` (zero matches).
