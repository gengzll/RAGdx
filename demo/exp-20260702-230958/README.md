# Demo experiment: `exp-20260702-230958`

A complete, real end-to-end run of the ragdx studio pipeline, committed as a
reference for what an experiment produces.

**Setup**: no-GT mode over a single PDF (ASMPT ESG report, ~6.3 MB — not
committed for repo-size reasons), 5 synthesized questions, 8 Bayesian search
trials, GEPA/light prompt optimization, GLM-4-Flash as the LLM. This run was
also interrupted mid-way and **resumed from its checkpoints** (the 8 BO trials
replayed without LLM calls), so it doubles as a resume demo.

## Files

| File | What it is |
|---|---|
| `report.html` | Self-contained HTML report — **open this in a browser** (TOC, Bayesian search trials, prompt before/after, diagnosis, run cost) |
| `result.json` | The full `schema_version: 1` experiment bundle the report renders from |
| `final/rag_config.yaml` | Ship-ready optimized RAG config (winning chunk size / overlap / top-k + prompt) |
| `final/prompt.md` | The winning system prompt |
| `questions_synthesized.jsonl` | The evaluation questions synthesized from the PDF (persisted so resume reuses them) |
| `meta.json` | The studio's run record: settings, status, timestamps (no secrets) |

## Reproduce

```bash
pip install -e ".[experiment]"
ragdx ui                      # upload a PDF, keep defaults, Run
# or headless:
ragdx experiment <your.pdf> --no-gt --bo-trials 8 --n-questions 5
# view any bundle:
ragdx experiment-dashboard --bundle demo/exp-20260702-230958/result.json
```
