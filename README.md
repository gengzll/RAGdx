![ragdx](docs/logo.png)

`ragdx` runs an **end-to-end RAG optimization experiment** on your own
documents and shows you the result. Point it at a PDF (optionally with a
ground-truth question set), and it will:

1. **Load & chunk** the document and resolve an evaluation question set —
   either from your ground truth, or synthesized from the corpus.
2. **Search** RAG configurations (chunk size / overlap / top-k) with
   Bayesian optimization.
3. **Optimize the prompt** at the winning config with DSPy (before/after).
4. **Evaluate** everything with a composite objective (Ragas / DeepEval)
   and **diagnose** what's still weak.
5. **Report**: a self-contained HTML report you can view and download,
   plus the ship-ready optimized `RAGConfig` + prompt.

Two ground-truth modes, chosen automatically:

- **with-GT** — you provide questions + reference answers (Excel/CSV or
  JSONL); metrics include answer correctness.
- **no-GT** — questions are synthesized from the document; metrics use
  reference-free signals (faithfulness, relevancy, …).

## Install

```bash
pip install -e ".[experiment]"     # recommended: everything for a run + the UI
```

API keys are read from the environment in this fallback order:
`ZHIPU_API_KEY` → `OPENAI_API_KEY`. The defaults target Zhipu GLM
(`openai/glm-4-airx`, Zhipu's fast-inference model; `openai/glm-4-flash`
is the free-tier alternative); override the model / base URL in the UI
or via CLI flags for OpenAI, Anthropic, etc.

## Quickstart — the studio (recommended)

```bash
ragdx ui
```

This opens the Streamlit studio in your browser:

1. **Upload** a PDF. Optionally upload an Excel/CSV ground-truth file —
   columns named `question` / `ground_truth` (and optional `contexts`)
   are auto-detected; if they're named differently you map them in the UI.
2. Set the model / API key and trial budget in the sidebar, then click
   **Run experiment**.
3. Watch **live progress** (Bayesian search → prompt optimization →
   evaluation) with a running log.
4. When it finishes, the **report renders inline** and you can download
   the HTML report, the raw `result.json` bundle, and the optimized
   config + prompt.

## Headless CLI

Same pipeline without the UI:

```bash
# no-GT: questions synthesized from the PDF
ragdx experiment path/to/report.pdf --no-gt --bo-trials 8 --n-questions 5

# with-GT: labelled questions in a JSONL ({question, ground_truth, contexts?})
ragdx experiment path/to/report.pdf --has-gt \
    --questions data/labelled_qa.jsonl --mode with_gt

# render / re-render any bundle as a standalone HTML report
ragdx experiment-report .ragdx_experiment/result.json -o report.html
```

`ragdx experiment` writes a `schema_version: 1` bundle to
`--output-dir/result.json` plus a `final/` folder (optimized
`rag_config.yaml` + `prompt.md`). `ragdx experiment-dashboard` opens the
Streamlit viewer on an existing bundle.

## Programmatic API

```python
from ragdx import run_experiment

result = run_experiment(
    corpus="report.pdf",
    has_gt=False,               # no-GT: synthesize questions from the PDF
    n_bo_trials=8,
    api_key="<your-key>",
    progress_callback=lambda ev: print(ev["pct"], ev["stage"]),
)
print(result.bundle["bayes_search"]["no_gt"]["best_params"])
```

For with-GT runs, pass `questions_path=` (a JSONL). To turn an
Excel/CSV into that JSONL, use the loader the studio uses:

```python
from ragdx.loaders import load_gt_table, records_from_table, write_questions_jsonl

df = load_gt_table("labelled_qa.xlsx")
records = records_from_table(df)                 # auto-detects columns
write_questions_jsonl(records, "questions.jsonl")
```

## Testing

```bash
python -m pytest -q
ruff check src tests
```

## License

Apache-2.0.
