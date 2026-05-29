"""Self-contained HTML report renderer for ``ragdx experiment`` bundles.

Mirrors the content of :mod:`ragdx.ui.experiment_dashboard` but emits a
single static HTML file with embedded Plotly charts -- no Streamlit
runtime, no external assets required at view-time (Plotly JS is
fetched from a CDN by the rendered page itself).

Use cases:

* **Experiment verification reports** -- attach the HTML to a PR /
  email so reviewers can see the run without spinning up Python.
* **PDF export** -- open the HTML in a browser and Ctrl-P -> "Save
  as PDF". Modern browsers handle the embedded Plotly figures.

Entry point: :func:`render_report` (bundle -> HTML string). The CLI
``ragdx experiment-report`` is a thin wrapper that loads the bundle
JSON, calls this, and writes the result.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.express as px

from ragdx.experiments import SCHEMA_VERSION, migrate_legacy_bundle

# --------------------------------------------------------------- shell
_CSS = """
:root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff;
        --accent:#1a73e8; --warn:#b45309; --ok:#15803d; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: var(--bg); color: var(--fg);
             font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                          Roboto, "Helvetica Neue", Arial, sans-serif;
             font-size: 14px; line-height: 1.55; }
.page { max-width: 1100px; margin: 32px auto; padding: 0 24px; }
h1 { font-size: 28px; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 32px 0 12px; padding-top: 16px;
     border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 20px 0 8px; color: #111827; }
h4 { font-size: 14px; margin: 14px 0 6px; color: #374151; }
.subtle { color: var(--muted); font-size: 13px; }
.caption { color: var(--muted); font-size: 12px; margin: 4px 0 12px; }
.metric-row { display: flex; gap: 16px; margin: 12px 0 20px; flex-wrap: wrap; }
.metric { flex: 1 1 180px; padding: 12px 16px; background: #f9fafb;
          border: 1px solid var(--line); border-radius: 8px; min-width: 160px; }
.metric .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase;
               letter-spacing: 0.05em; }
.metric .val { font-size: 18px; font-weight: 600; margin-top: 2px;
               word-break: break-all; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px;
        font-size: 13px; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px;
         text-align: left; vertical-align: top; }
th { background: #f3f4f6; font-weight: 600; }
tr:hover td { background: #fafbfc; }
pre, code { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
            font-size: 12px; }
pre { background: #f3f4f6; padding: 10px 12px; border-radius: 6px;
      overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
.box { border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px;
       margin: 12px 0; background: #fafbfc; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.two-col > div { min-width: 0; }
details { margin: 8px 0; border: 1px solid var(--line); border-radius: 6px;
          padding: 8px 12px; background: #fafbfc; }
details > summary { cursor: pointer; font-weight: 500; color: #374151; }
details[open] > summary { margin-bottom: 10px; }
.warn { color: var(--warn); }
.ok { color: var(--ok); }
.tag { display: inline-block; padding: 2px 8px; background: #eef2ff;
       color: #3730a3; border-radius: 4px; font-size: 12px; }
.tag.warn { background: #fef3c7; color: #92400e; }
.tag.ok { background: #d1fae5; color: #065f46; }
.divider { border-top: 1px dashed var(--line); margin: 16px 0; }
@media print {
  .page { max-width: none; margin: 0; padding: 12mm; }
  h2 { page-break-before: auto; }
  details { page-break-inside: avoid; }
}
"""


def _esc(s: Any) -> str:
    """HTML-escape ``s`` (calling ``str()`` first)."""
    return html.escape("" if s is None else str(s))


def _wrap(title: str, body: str, generated_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_esc(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_CSS}</style>
</head>
<body><div class="page">
<header>
  <h1>{_esc(title)}</h1>
  <div class="subtle">Generated {_esc(generated_at)} · ragdx schema v{SCHEMA_VERSION}</div>
</header>
{body}
</div></body></html>"""


def _metric(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<div class="lbl">{_esc(label)}</div>'
        f'<div class="val">{_esc(value)}</div>'
        '</div>'
    )


def _table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    """Render a list-of-dicts as an HTML table. ``cols`` controls order."""
    if not rows:
        return '<p class="subtle">_(no data)_</p>'
    cols = cols or list(rows[0].keys())
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_esc(r.get(c, ''))}</td>" for c in cols)
        + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _details(summary: str, body: str, open_: bool = False) -> str:
    o = " open" if open_ else ""
    return f"<details{o}><summary>{_esc(summary)}</summary>{body}</details>"


def _fig_html(fig: Any) -> str:
    """Embed a Plotly figure as a script-only HTML fragment (no <html>)."""
    return fig.to_html(include_plotlyjs=False, full_html=False)


# ----------------------------------------------------------- sections
def _render_meta(bundle: dict) -> str:
    meta = bundle.get("meta") or {}
    parts: list[str] = []
    parts.append('<h2>Run metadata</h2>')

    parts.append('<div class="metric-row">')
    parts.append(_metric("Experiment mode", meta.get("experiment_mode", "?")))
    parts.append(_metric("Modes run", ", ".join(meta.get("modes_run") or []) or "—"))
    parts.append(_metric("Has GT", "Yes" if meta.get("has_gt") else "No"))
    parts.append(_metric("Detected GT", meta.get("detected_gt_mode", "—")))
    parts.append('</div>')

    parts.append('<div class="metric-row">')
    parts.append(_metric("Model", meta.get("model", "?")))
    parts.append(_metric("Endpoint", meta.get("model_endpoint", "?")))
    parts.append('</div>')

    rc = meta.get("run_config") or {}
    if rc:
        parts.append('<h3>Run config (reproducibility)</h3>')
        parts.append(
            '<p class="caption">These are the values of <code>ragdx '
            'experiment</code>\'s flags at the time this bundle was '
            'produced. The same values plus the bundle\'s seed reproduce '
            'this run exactly.</p>'
        )
        rc_rows = [{"flag": k, "value": v} for k, v in rc.items()]
        parts.append(_table(rc_rows, cols=["flag", "value"]))

    source = meta.get("source") or {}
    if source:
        # PDFs: relabel chunk_size/overlap so "initial load" is honest.
        display = json.loads(json.dumps(source))
        pm = display.get("pdf_meta") if isinstance(display, dict) else None
        if isinstance(pm, dict):
            for k in ("chunk_size", "chunk_overlap"):
                if k in pm:
                    pm[f"initial_load_{k}"] = pm.pop(k)
        parts.append(_details(
            "Corpus source",
            f'<pre>{_esc(json.dumps(display, indent=2, ensure_ascii=False))}</pre>'
            '<p class="caption">PDF <code>initial_load_*</code> fields '
            'reflect the chunking used at load-time for question synthesis. '
            'The Bayesian search below explores alternate chunk_size / '
            'chunk_overlap values; see <strong>Bayesian search → '
            'best_params</strong> for the winner.</p>',
            open_=True,
        ))

    if meta.get("_migrated_from"):
        parts.append(
            f'<p class="caption">_Migrated from legacy bundle '
            f'(<code>{_esc(meta["_migrated_from"])}</code>)_</p>'
        )

    return "\n".join(parts)


def _render_diagnostics(bundle: dict) -> str:
    diag = bundle.get("data_diagnostics") or {}
    if not diag:
        return ""
    rows = [{"mode": m, **d} for m, d in diag.items()]
    return (
        '<h2>Data diagnostics</h2>'
        + _table(rows)
    )


def _render_questions(bundle: dict) -> str:
    qs = bundle.get("questions") or []
    if not qs:
        return ""
    rows = []
    for i, q in enumerate(qs, 1):
        rows.append({
            "#": i,
            "question": q.get("question", ""),
            "ground_truth": q.get("ground_truth") or "_(none)_",
        })
    out = ['<h2>Questions</h2>', _table(rows, cols=["#", "question", "ground_truth"])]
    extras = bundle.get("extras") or {}
    syn = extras.get("synthesized_questions") or []
    if syn:
        rows_s = [{"question": s.get("question", ""), "source_chunk_ids":
                   ", ".join(map(str, s.get("source_chunk_ids", [])))} for s in syn]
        out.append(_details(
            f"Synthesized question provenance ({len(syn)})",
            _table(rows_s, cols=["question", "source_chunk_ids"]),
        ))
    return "\n".join(out)


def _render_objectives(bundle: dict) -> str:
    objs = bundle.get("objectives") or {}
    if not objs:
        return ""
    parts = ['<h2>Composite objectives</h2>']
    parts.append(
        '<p class="caption"><strong>Metric weights</strong> combine into '
        'the composite score (<code>sum of weight_i * metric_i</code>) -- '
        'what BO and DSPy optimize against. <strong>Constraints</strong> '
        'are hard pass/fail thresholds: a trial that violates one is '
        'still scored but marked <strong>infeasible</strong> so the '
        'planner can filter unsafe configs.</p>'
    )
    for m, spec in objs.items():
        weights = (spec or {}).get("metrics") or {}
        constraints = (spec or {}).get("constraints") or {}
        body = '<div class="two-col">'
        body += '<div><h4>Metric weights</h4>'
        body += _table(
            [{"metric": k, "weight": v} for k, v in weights.items()],
            cols=["metric", "weight"],
        )
        body += '</div><div><h4>Constraints</h4>'
        if constraints:
            crows = []
            for k, v in constraints.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    crows.append({"metric": k, "direction": v[0], "bound": v[1]})
                else:
                    crows.append({"metric": k, "direction": "—", "bound": v})
            body += _table(crows, cols=["metric", "direction", "bound"])
        else:
            body += '<p class="caption">_(no hard constraints)_</p>'
        body += '</div></div>'
        parts.append(_details(f"`{m}` objective", body, open_=(len(objs) <= 2)))
    return "\n".join(parts)


def _trial_param_label(params: dict) -> str:
    if not params:
        return "—"
    return " · ".join(f"{k}={v}" for k, v in params.items())


def _render_bayes_search(bundle: dict) -> str:
    bs = bundle.get("bayes_search") or {}
    if not bs:
        return ""
    parts = ['<h2>Bayesian RAG-config search</h2>']

    for mode, payload in bs.items():
        payload = payload or {}
        parts.append(f'<h3>Mode: <code>{_esc(mode)}</code></h3>')
        if payload.get("_legacy_kind") == "grid":
            parts.append('<p class="caption">_Legacy bundle: grid search '
                         '(only top_k varied)._</p>')

        trials = payload.get("trials") or []
        best_params = payload.get("best_params") or {}
        best_comp = payload.get("best_composite")
        search_space = payload.get("search_space") or {}

        parts.append('<div class="metric-row">')
        parts.append(_metric("Trials", len(trials)))
        parts.append(_metric(
            "Best composite",
            f"{best_comp:.3f}" if isinstance(best_comp, (int, float)) else "—",
        ))
        parts.append(_metric("Best params", _trial_param_label(best_params)))
        parts.append('</div>')

        if search_space:
            parts.append(_details(
                "Search space",
                f'<pre>{_esc(json.dumps(search_space, indent=2))}</pre>',
            ))

        if not trials:
            parts.append('<p class="subtle">No trials recorded.</p>')
            continue

        # Trial table
        tbl_rows = []
        for i, t in enumerate(trials):
            row = {
                "#": t.get("trial_index", i),
                "params": _trial_param_label(t.get("params") or {}),
                "composite": (
                    f"{t.get('composite_score'):.3f}"
                    if isinstance(t.get("composite_score"), (int, float))
                    else "—"
                ),
                "feasible": "✓" if t.get("feasible", True) else "✗",
            }
            for k, v in (t.get("scores") or {}).items():
                row[k] = f"{v:.3f}" if isinstance(v, (int, float)) else v
            tbl_rows.append(row)
        cols = ["#", "params", "composite", "feasible"] + [
            k for k in tbl_rows[0] if k not in {"#", "params", "composite", "feasible"}
        ]
        parts.append(_table(tbl_rows, cols=cols))

        # Progression chart (composite per trial + running best)
        comp_series = [t.get("composite_score") for t in trials]
        if any(isinstance(v, (int, float)) for v in comp_series):
            rb = float("-inf")
            running: list[float | None] = []
            for v in comp_series:
                if isinstance(v, (int, float)):
                    rb = max(rb, v)
                running.append(rb if rb != float("-inf") else None)
            progress = pd.DataFrame({
                "trial": list(range(1, len(trials) + 1)),
                "composite": comp_series,
                "running best": running,
            }).melt("trial", value_name="score", var_name="series")
            fig = px.line(
                progress, x="trial", y="score", color="series",
                markers=True, title=f"BO progression — {mode}",
            )
            fig.update_layout(height=360, margin=dict(t=60, b=40))
            parts.append(_fig_html(fig))
            parts.append(
                '<p class="caption"><strong>composite</strong> = this '
                "trial's composite score (per the objective above). "
                "<strong>running best</strong> = highest composite seen "
                "so far -- the non-decreasing curve shows BO convergence.</p>"
            )

        # Per-trial Q/A inspector (compact: first 3 trials)
        records_keyed = [t for t in trials if t.get("records")]
        if records_keyed:
            show = records_keyed[:3]
            inspector = ""
            for t in show:
                inspector += (
                    f"<h4>Trial #{t.get('trial_index')} "
                    f"({_esc(_trial_param_label(t.get('params') or {}))})</h4>"
                )
                rec_rows = []
                for j, r in enumerate(t.get("records") or [], 1):
                    rec_rows.append({
                        "#": j,
                        "question": (r.get("question") or "")[:200],
                        "ground_truth": (r.get("ground_truth") or "—")[:200],
                        "answer": (r.get("answer") or "")[:300],
                    })
                inspector += _table(
                    rec_rows, cols=["#", "question", "ground_truth", "answer"]
                )
            parts.append(_details(
                f"Per-record outputs (first {len(show)} trials)",
                inspector,
            ))

    return "\n".join(parts)


def _render_dspy_a_b(bundle: dict) -> str:
    ab = bundle.get("dspy_a_b") or {}
    if not ab:
        return ""
    parts = ['<h2>DSPy before/after (at BO winner config)</h2>']

    for mode, payload in ab.items():
        payload = payload or {}
        parts.append(f'<h3>Mode: <code>{_esc(mode)}</code></h3>')

        bs = payload.get("baseline_scores") or {}
        os_ = payload.get("optimized_scores") or {}
        delta = payload.get("delta") or {}
        comp = payload.get("composite") or {}

        if not bs and not os_:
            parts.append('<p class="subtle">No before/after scores recorded.</p>')
            continue

        # Bar chart
        metrics = sorted(set(bs) | set(os_))
        rows = []
        for m in metrics:
            rows.append({"metric": m, "phase": "baseline", "score": bs.get(m)})
            rows.append({"metric": m, "phase": "optimized", "score": os_.get(m)})
        df = pd.DataFrame(rows)
        df["score_display"] = df["score"].fillna(0.0)
        fig = px.bar(
            df, x="metric", y="score_display", color="phase", barmode="group",
            text=df["score"].apply(
                lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"
            ),
            color_discrete_map={"baseline": "#9aa0a6", "optimized": "#1a73e8"},
            title=f"DSPy baseline vs optimized — {mode}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(range=[0, 1.05], title="score"),
            xaxis_tickangle=-25, height=420, margin=dict(t=60, b=80),
            legend_title=None,
        )
        parts.append(_fig_html(fig))

        # Composite headline
        if comp:
            b = (comp.get("baseline") or {}).get("score")
            o = (comp.get("optimized") or {}).get("score")
            d = comp.get("delta")
            parts.append('<div class="metric-row">')
            parts.append(_metric(
                "Composite baseline",
                f"{b:.3f}" if isinstance(b, (int, float)) else "—",
            ))
            parts.append(_metric(
                "Composite optimized",
                f"{o:.3f}" if isinstance(o, (int, float)) else "—",
            ))
            parts.append(_metric(
                "Composite Δ",
                f"{d:+.3f}" if isinstance(d, (int, float)) else "—",
            ))
            parts.append('</div>')

        # Delta table
        rows_d = []
        for m in metrics:
            d = delta.get(m)
            arrow = "→"
            if isinstance(d, (int, float)):
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
            rows_d.append({
                "metric": m,
                "baseline": (
                    f"{bs.get(m):.3f}"
                    if isinstance(bs.get(m), (int, float))
                    else "N/A"
                ),
                "optimized": (
                    f"{os_.get(m):.3f}"
                    if isinstance(os_.get(m), (int, float))
                    else "N/A"
                ),
                "delta": (
                    f"{arrow} {d:+.3f}"
                    if isinstance(d, (int, float))
                    else "N/A"
                ),
            })
        parts.append(_table(rows_d))

        # MIPROv2 trial-scores progression
        trial_scores = payload.get("trial_scores") or []
        if trial_scores:
            rb = float("-inf")
            running = []
            for s in trial_scores:
                rb = max(rb, s)
                running.append(rb)
            line_df = pd.DataFrame({
                "trial": list(range(1, len(trial_scores) + 1)),
                "per-trial": trial_scores,
                "running best": running,
            }).melt("trial", value_name="score", var_name="series")
            fig2 = px.line(
                line_df, x="trial", y="score", color="series",
                markers=True, title=f"MIPROv2 trial scores — {mode}",
            )
            fig2.update_layout(height=320, margin=dict(t=60, b=40))
            parts.append(_fig_html(fig2))
            parts.append(
                '<p class="caption"><strong>per-trial</strong> = MIPROv2\'s '
                'inner-loop training-time score for that trial (token-F1 '
                'with-GT, LLM-judge without-GT — <strong>NOT</strong> the '
                'ragas composite). <strong>running best</strong> = highest '
                'score seen so far.</p>'
            )

        # Prompts: baseline vs optimized
        base_instr = payload.get("baseline_instructions") or {}
        opt_instr = payload.get("instructions") or {}
        base_demos = payload.get("baseline_demos") or {}
        opt_demos = payload.get("demos") or {}
        if base_instr or opt_instr:
            parts.append('<h4>Prompts: baseline vs MIPROv2-optimized</h4>')
            parts.append(
                '<p class="caption"><strong>Baseline</strong> = the DSPy '
                'signature\'s instruction at run start (your '
                '<code>--system-instruction</code> if set, else ragdx '
                'default). <strong>Optimized</strong> = what MIPROv2 picked '
                'after its inner-loop search. Identical columns = MIPROv2 '
                'found no improvement and kept the seed (the right default '
                'when scores are tied).</p>'
            )
            predictor_names = sorted(set(base_instr) | set(opt_instr))
            for name in predictor_names:
                body = (
                    '<div class="two-col"><div><h4>Baseline instruction</h4>'
                    f'<pre>{_esc(base_instr.get(name) or "(empty — default signature)")}</pre>'
                    '</div><div><h4>Optimized instruction</h4>'
                    f'<pre>{_esc(opt_instr.get(name) or "(empty)")}</pre>'
                    '</div></div>'
                    f'<p class="caption">Few-shot demos: baseline = '
                    f'{len(base_demos.get(name) or [])} · optimized = '
                    f'{len(opt_demos.get(name) or [])}</p>'
                )
                parts.append(_details(
                    f"Predictor `{name}`", body, open_=(len(predictor_names) == 1)
                ))

        # Per-record before/after inspector
        recs = payload.get("records") or []
        if recs:
            show = recs[:3]
            inspector = ""
            for j, r in enumerate(show, 1):
                inspector += f"<h4>Record {j}</h4>"
                gt = r.get("ground_truth") or ""
                qa_rows = [{
                    "field": "question",
                    "content": r.get("question", ""),
                }]
                if gt:
                    qa_rows.append({"field": "ground_truth", "content": gt})
                qa_rows.append({"field": "baseline answer",
                                "content": r.get("baseline_answer", "")})
                qa_rows.append({"field": "optimized answer",
                                "content": r.get("optimized_answer", "")})
                inspector += _table(qa_rows, cols=["field", "content"])
            parts.append(_details(
                f"Per-record before/after (first {len(show)} of {len(recs)})",
                inspector,
                open_=True,
            ))

    return "\n".join(parts)


def _render_extras(bundle: dict) -> str:
    extras = bundle.get("extras") or {}
    if not extras:
        return ""
    parts = ['<h2>Extras</h2>']
    if "pdf_meta" in extras:
        parts.append(_details(
            "PDF metadata",
            f'<pre>{_esc(json.dumps(extras["pdf_meta"], indent=2))}</pre>',
        ))
    leftover = {k: v for k, v in extras.items()
                if k not in {"pdf_meta", "synthesized_questions"}}
    if leftover:
        parts.append(_details(
            "Other extras",
            f'<pre>{_esc(json.dumps(leftover, indent=2, ensure_ascii=False))}</pre>',
        ))
    return "\n".join(parts) if len(parts) > 1 else ""


# --------------------------------------------------------- public entry
def render_report(bundle: dict, *, title: str | None = None) -> str:
    """Render ``bundle`` (a v1 ragdx experiment bundle) as a self-contained
    HTML document string.

    Legacy bundles are upgraded via :func:`migrate_legacy_bundle` first.
    """
    if bundle.get("schema_version") != SCHEMA_VERSION:
        bundle = migrate_legacy_bundle(bundle)

    meta = bundle.get("meta") or {}
    auto_title = (
        f"ragdx experiment report — {meta.get('model', '?')} · "
        f"{', '.join(meta.get('modes_run') or []) or 'unknown'}"
    )
    sections = "\n".join(
        s for s in (
            _render_meta(bundle),
            _render_diagnostics(bundle),
            _render_objectives(bundle),
            _render_bayes_search(bundle),
            _render_dspy_a_b(bundle),
            _render_questions(bundle),
            _render_extras(bundle),
        ) if s
    )
    return _wrap(
        title or auto_title,
        sections,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


__all__ = ["render_report"]
