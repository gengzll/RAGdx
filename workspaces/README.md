# ragdx workspace demos

Two end-to-end walkthroughs of the `ragdx workspace` command family,
both executed against the public **ASMPT 2024 ESG report**
(`../e0522-asmptesgreport.pdf`, 114 pages) with **GLM-4-Flash** as
the generator. Every JSON, log, YAML, and HTML file in these folders
was produced by `ragdx` during the run — they are real artifacts,
not hand-crafted samples.

| folder | What it teaches | Wall time |
|---|---|---|
| [`end2end_demo/`](end2end_demo) | the full optimization pipeline in one chain: eval → diagnose → tune-rag (BO) → tune-prompt (DSPy MIPROv2) → report | ~35 min |
| [`diagnosis_demo/`](diagnosis_demo) | the diagnosis-driven loop: eval → diagnose → tune what diagnosis points at → re-eval → re-diagnose → compare | ~15 min |

The two demos share `rag_config.yaml` + `questions.jsonl` (3 ESG
questions, no ground truth → ragas runs in `no_gt` mode with
`context_precision` + `faithfulness` + `answer_relevancy`).

---

## Prerequisites

```bash
# From the repo root:
pip install -e ".[experiment]"
export ZHIPU_API_KEY=<your-zhipu-key>          # or OPENAI_API_KEY
```

The PDF (`e0522-asmptesgreport.pdf`) lives at the repo root and is
referenced by `corpus: e0522-asmptesgreport.pdf` in each workspace's
`rag_config.yaml`. The workspace `corpus_path` resolver tries the
workspace folder first, then the cwd, so a bare filename works
when you `cd` to the repo root before running commands.

---

## Demo 1: `end2end_demo/` — the full pipeline

**Goal:** show the complete optimization chain, from a baseline
score through both retrieval BO and DSPy prompt tuning, all written
to one folder.

```bash
# 1. Create the workspace + drop the config + questions in.
ragdx workspace init end2end_demo \
    --config rag_config.demo.yaml \
    --questions questions.demo.jsonl \
    --corpus e0522-asmptesgreport.pdf

# 2. Baseline evaluation (ragas no-GT).
ragdx workspace eval end2end_demo

# 3. Diagnose the baseline (rule-based).
ragdx workspace diagnose end2end_demo
# Optional richer modes (slower; need a working OPENAI_BASE_URL / key):
#   ragdx workspace diagnose end2end_demo --use-llm
#   ragdx workspace diagnose end2end_demo --use-both

# 4. Tune the RAG (chunker + retriever + top_k, joint BO, 4 trials).
ragdx workspace tune-rag end2end_demo --stage joint --budget 4

# 5. Tune the prompt (DSPy MIPROv2 light + embed_rubric inner-loop).
ragdx workspace tune-prompt end2end_demo \
    --dspy-optimizer mipro \
    --dspy-metric embed_rubric \
    --mipro-auto light

# 6. Render the HTML report (covers the BO stage by default; pass
#    --source tune_prompt_mipro.json to see the prompt A/B view).
ragdx workspace report end2end_demo
```

### What you get in `end2end_demo/`

| file | what it is |
|---|---|
| `workspace.yaml` | manifest: config + questions paths, evaluator, mode, command history, `current.*` pointers to each verb's latest output |
| `rag_config.yaml` | the baseline RAG description (under test) |
| `questions.jsonl` | the eval suite |
| `baseline.eval.json` | normalized `EvaluationResult` from step 2 |
| `baseline.eval.diagnose.json` | rule-based `DiagnosisReport` from step 3 |
| `tune_rag_joint.json` | BO bundle from step 4 (per-trial scores, records, per-record metric breakdowns, baseline + optimized + comparison diagnoses) |
| `rag_optimized.tune_rag.yaml` | tuned RAG config from step 4 (api_key scrubbed) |
| `tune_prompt_mipro.json` | DSPy A/B bundle from step 5 (MIPROv2 candidates, before/after answers) |
| `rag_optimized.tune_prompt.yaml` | tuned config from step 5 (winner instruction + few-shot demos) |
| `report.html` | self-contained HTML covering the chosen source bundle |
| `console.log` | full stdout of the 5-step chain |

---

## Demo 2: `diagnosis_demo/` — the diagnosis-driven loop

**Goal:** show how the diagnosis output drives the next step, and
how `comparison` answers "did the optimization fix what we set out
to fix?".

```bash
# 1. Init.
ragdx workspace init diagnosis_demo \
    --config rag_config.demo.yaml \
    --questions questions.demo.jsonl \
    --corpus e0522-asmptesgreport.pdf

# 2. Baseline evaluation.
ragdx workspace eval diagnosis_demo

# 3. Diagnose. Read the JSON's `optimization_candidates` field:
#    e.g. ["autorag_pipeline_search"] -> tune-rag is what to run next.
ragdx workspace diagnose diagnosis_demo

# 4. Run what diagnosis recommended.
ragdx workspace tune-rag diagnosis_demo --stage joint --budget 4

# 5. Re-evaluate at the optimized config.
ragdx workspace eval diagnosis_demo

# 6. Re-diagnose. The bundle now has `baseline` + `optimized` +
#    `comparison` showing which hypotheses got resolved.
ragdx workspace diagnose diagnosis_demo

# 7. Render the HTML report.
ragdx workspace report diagnosis_demo
```

### What you get in `diagnosis_demo/`

| file | what it is |
|---|---|
| `baseline.eval.json` / `baseline.eval.diagnose.json` | before optimization |
| `tune_rag_joint.json` | the optimization the diagnosis pointed at |
| `rag_optimized.tune_rag.yaml` | tuned config |
| `post_1.eval.json` / `post_1.eval.diagnose.json` | after optimization |
| `report.html` | shows baseline → comparison → optimized diagnoses side-by-side |

The naming `post_<N>.eval.json` is automatic: the workspace knows
the first eval is the "baseline" and subsequent evals are "post"
runs, so chained `tune` → `eval` cycles don't overwrite anything.

---

## Common workspace verbs

```bash
ragdx workspace list                        # all workspaces under ./workspaces/
ragdx workspace show <name>                 # one workspace's config + history
ragdx workspace report <name>               # render the latest tune > eval as HTML
ragdx workspace report <name> --source X    # render a specific bundle
ragdx workspace compare <a> <b> [c...]      # cross-experiment metric / posterior delta
```

`workspaces_root` defaults to `./workspaces/`; override with
`RAGDX_WORKSPACES=/some/path`. Each workspace is fully
self-contained, so you can `tar czf` the folder, ship it
to a teammate, and they can `ragdx workspace report <name>` to
inspect the same artifacts.

## Re-rendering the report from a specific bundle

The default `report` verb picks the most recently-produced bundle
(`tune-prompt > tune-rag > eval`). To pin a particular source:

```bash
# render the tune-rag BO view explicitly
ragdx workspace report end2end_demo --source tune_rag_joint.json -o tune_rag_report.html

# render the DSPy A/B view from the prompt tune
ragdx workspace report end2end_demo --source tune_prompt_mipro.json -o tune_prompt_report.html
```

## Cleanup

Workspaces are local-only and ignored by `.gitignore` patterns other
than the two demo folders here. To clean up:

```bash
rm -rf workspaces/end2end_demo workspaces/diagnosis_demo
# or wipe everything:
rm -rf workspaces/
```

Re-init with `ragdx workspace init <name>` whenever you want to
start fresh.
