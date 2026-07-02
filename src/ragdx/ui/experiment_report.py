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
.callout { padding: 10px 14px; border-radius: 8px; font-size: 13px; }
.callout.warn { background: #fef3c7; border: 1px solid #f59e0b; color: #78350f; }
.callout.ok { background: #dcfce7; border: 1px solid #22c55e; color: #14532d; }
h1.part { font-size: 24px; margin: 48px 0 4px; padding: 12px 16px;
          background: #eef2ff; border-left: 5px solid var(--accent);
          border-radius: 6px; }
nav.toc { background: #f9fafb; border: 1px solid var(--line);
          border-radius: 8px; padding: 14px 20px; margin: 20px 0; }
nav.toc .toc-title { font-weight: 600; margin-bottom: 6px; }
nav.toc ol { margin: 0; padding-left: 20px; }
nav.toc a { color: var(--accent); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
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


_REPORT_JS = """
// Per-trial visibility checkboxes toggle each trial block.
document.addEventListener('change', function(e){
  var t = e.target;
  if(t && t.classList && t.classList.contains('trial-toggle')){
    var id = t.getAttribute('data-target');
    var el = document.getElementById(id);
    if(el) el.style.display = t.checked ? '' : 'none';
  }
});
"""


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
</div>
<script>{_REPORT_JS}</script>
</body></html>"""


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
    parts.append(_metric("Model", meta.get("model", "?")))
    parts.append(_metric("Endpoint", meta.get("model_endpoint", "?")))
    parts.append('</div>')

    emb = meta.get("embedding") or {}
    if emb.get("model"):
        parts.append('<div class="metric-row">')
        parts.append(_metric("Chunk embedding model", emb["model"]))
        if emb.get("max_seq_tokens"):
            parts.append(_metric("Embedding max input", f"{emb['max_seq_tokens']} tokens"))
        parts.append('</div>')
        if emb.get("max_seq_tokens"):
            parts.append(
                '<p class="caption">Chunk text beyond the embedding '
                'model\'s max input is silently truncated before '
                'embedding — chunk sizes far above this limit stop '
                'improving retrieval.</p>'
            )

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


def _severity_class(severity: str) -> str:
    s = (severity or "").lower()
    if s == "high":
        return "warn"
    if s in ("medium", "med"):
        return ""
    return "ok"


def _render_causal_graph_svg(signals: list[dict], graph: dict | None = None) -> str:
    """Phase 4c: render the 8-node causal graph as an SVG diagram.

    Layout: fixed positions for the 8 canonical nodes (the analyser
    has a hardcoded topology). Node circles are sized by posterior
    (bigger = stronger signal) and tinted by component (retrieval /
    generation / e2e / pipeline -> blue / orange / green / grey).
    Edges drawn from ``graph.edges`` when supplied; otherwise from the
    hardcoded analyzer topology so legacy reports still render.

    Returns ``""`` when ``signals`` is empty (caller skips the section).
    """
    if not signals:
        return ""

    # Hardcoded coordinates -- four columns roughly: upstream / mid /
    # downstream / side. Matches the 8-node causal topology.
    layout = {
        "corpus_chunking_defect":      (110, 80),
        "retrieval_recall_defect":     (110, 180),
        "retrieval_precision_defect":  (110, 280),
        "context_packing_defect":      (310, 130),
        "grounding_defect":            (310, 240),
        "citation_binding_defect":     (510, 240),
        "judge_or_metric_instability": (510, 80),
        "distribution_shift":          (510, 160),
    }
    comp_colors = {
        "retrieval":  "#1d4ed8",   # blue
        "generation": "#c2410c",   # orange
        "e2e":        "#15803d",   # green
    }
    edges_default = [
        ("corpus_chunking_defect", "retrieval_recall_defect"),
        ("retrieval_recall_defect", "context_packing_defect"),
        ("retrieval_precision_defect", "context_packing_defect"),
        ("retrieval_precision_defect", "grounding_defect"),
        ("context_packing_defect", "grounding_defect"),
        ("grounding_defect", "citation_binding_defect"),
        ("judge_or_metric_instability", "distribution_shift"),
        ("distribution_shift", "retrieval_recall_defect"),
        ("distribution_shift", "grounding_defect"),
    ]
    if graph and isinstance(graph.get("edges"), list):
        edges = [(e.get("source"), e.get("target")) for e in graph["edges"]]
    else:
        edges = edges_default

    sig_by_node = {s.get("node"): s for s in signals}

    # SVG primitives.
    width, height = 640, 360
    bits: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'style="border:1px solid var(--line);border-radius:6px;background:#fff" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    # Arrow marker definition.
    bits.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>'
    )
    # Edges first so nodes overlay them.
    for src, tgt in edges:
        if src not in layout or tgt not in layout:
            continue
        x1, y1 = layout[src]
        x2, y2 = layout[tgt]
        bits.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#cbd5e1" stroke-width="1.5" marker-end="url(#arrow)" />'
        )
    # Nodes.
    for node, (cx, cy) in layout.items():
        sig = sig_by_node.get(node) or {}
        post = sig.get("posterior")
        post_val = float(post) if isinstance(post, (int, float)) else 0.0
        # Radius: 12..28 px scaled by posterior.
        r = 12 + int(16 * max(0.0, min(1.0, post_val)))
        comp = sig.get("component") or "e2e"
        fill = comp_colors.get(comp, "#9ca3af")
        opacity = 0.4 + 0.6 * post_val  # stronger fill for higher posterior
        bits.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'fill-opacity="{opacity:.2f}" stroke="{fill}" stroke-width="1.5"><title>'
            f'{_esc(node)} (post={post_val:.2f}, component={comp})</title></circle>'
        )
        # Posterior label inside the node.
        bits.append(
            f'<text x="{cx}" y="{cy + 4}" font-size="11" font-weight="600" '
            f'fill="#fff" text-anchor="middle">{post_val:.2f}</text>'
        )
        # Node name label below.
        label = node.replace("_defect", "").replace("_", " ")
        bits.append(
            f'<text x="{cx}" y="{cy + r + 14}" font-size="11" '
            f'fill="#374151" text-anchor="middle">{_esc(label)}</text>'
        )
    # Legend.
    legend_x, legend_y = width - 150, 12
    for i, (comp, color) in enumerate(comp_colors.items()):
        ly = legend_y + i * 18
        bits.append(
            f'<rect x="{legend_x}" y="{ly}" width="12" height="12" fill="{color}" '
            f'fill-opacity="0.7" />'
            f'<text x="{legend_x + 16}" y="{ly + 10}" font-size="11" fill="#374151">'
            f'{_esc(comp)}</text>'
        )
    bits.append('</svg>')

    # Node glossary (item 3): what each node means + how to read the SVG.
    from ragdx.core.metrics import CAUSAL_NODE_GLOSSARY
    present = {s.get("node") for s in signals}
    gloss_rows = []
    for node, meta in CAUSAL_NODE_GLOSSARY.items():
        sig = sig_by_node.get(node) or {}
        post = sig.get("posterior")
        post_str = (
            f"{float(post):.2f}" if isinstance(post, (int, float)) else "—"
        )
        here = " " if node in present else ""
        gloss_rows.append(
            f'<tr><td><code>{_esc(node)}</code></td>'
            f'<td>{_esc(meta["layer"])}</td>'
            f'<td>{_esc(post_str)}</td>'
            f'<td class="subtle">{_esc(meta["desc"])}{_esc(here)}</td></tr>'
        )
    gloss = (
        '<table><thead><tr><th>node</th><th>layer</th>'
        '<th>posterior</th><th>what it means</th></tr>'
        f'</thead><tbody>{"".join(gloss_rows)}</tbody></table>'
    )

    return (
        '<h4>Causal graph (SVG)</h4>'
        '<p class="caption"><strong>How to read it:</strong> each circle is '
        'a hypothesized <em>defect</em> (failure mode). The circle\'s size '
        'and fill opacity scale with its <em>posterior probability</em> — '
        'how likely that defect is, given the metrics + traces — so the '
        'biggest, boldest node is the prime suspect. Colour = which layer '
        'the defect sits in (retrieval blue / generation orange / e2e '
        'green). Arrows are causal-propagation edges: a defect upstream '
        'raises the probability of the ones it points to. Lower posterior '
        '= healthier. The number inside each node is its posterior.</p>'
        + "".join(bits)
        + '<div style="margin-top:10px">'
        + _details("Causal node glossary (what each node means)", gloss)
        + '</div>'
    )


def _render_answer_diff(baseline: str, optimized: str) -> str:
    """Phase 4b: word-level baseline-vs-optimized diff with colours.

    Uses ``difflib.ndiff`` on word tokens. The shape:
    * equal tokens render plain.
    * tokens only in baseline render struck-through in red ("removed").
    * tokens only in optimized render bold in green ("added").
    Whitespace is preserved verbatim. Empty diff (identical strings)
    returns ``""`` so the caller can skip the section entirely.
    """
    import difflib
    import re as _re

    if not baseline and not optimized:
        return ""
    if baseline == optimized:
        return (
            '<p class="caption"><em>(Baseline and optimized answers '
            'are identical -- prompt change had no textual effect on '
            'this record.)</em></p>'
        )

    # Tokenise by word boundaries while keeping the separators. Newlines
    # get their own token so paragraph breaks survive.
    def _tokens(text: str) -> list[str]:
        return [t for t in _re.split(r'(\s+)', text) if t != ""]

    base_toks = _tokens(baseline)
    opt_toks = _tokens(optimized)

    sm = difflib.SequenceMatcher(a=base_toks, b=opt_toks, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(_esc("".join(base_toks[i1:i2])))
        elif tag == "delete":
            parts.append(
                f'<span style="background:#fee2e2;text-decoration:line-through;color:#991b1b">'
                f'{_esc("".join(base_toks[i1:i2]))}</span>'
            )
        elif tag == "insert":
            parts.append(
                f'<span style="background:#dcfce7;font-weight:600;color:#166534">'
                f'{_esc("".join(opt_toks[j1:j2]))}</span>'
            )
        elif tag == "replace":
            parts.append(
                f'<span style="background:#fee2e2;text-decoration:line-through;color:#991b1b">'
                f'{_esc("".join(base_toks[i1:i2]))}</span>'
            )
            parts.append(
                f'<span style="background:#dcfce7;font-weight:600;color:#166534">'
                f'{_esc("".join(opt_toks[j1:j2]))}</span>'
            )
    return (
        '<h4>Answer diff (baseline -> optimized)</h4>'
        '<p class="caption">Red strikethrough = removed by optimization. '
        'Green bold = added. Plain = unchanged.</p>'
        f'<pre style="white-space:pre-wrap;background:#fafbfc;padding:12px;'
        f'border:1px solid var(--line);border-radius:6px">{"".join(parts)}</pre>'
    )


def _render_run_cost(bundle: dict) -> str:
    """Phase 4e: surface "what did this run cost" -- per-trial wall
    time, mean per-question latency, total runtime. Token / dollar
    accounting only renders when traces carry it; otherwise we just
    show the timing we always have.
    """
    bayes = bundle.get("bayes_search") or {}
    dspy = bundle.get("dspy_a_b") or {}
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    for mode, payload in bayes.items():
        trials = (payload or {}).get("trials") or []
        if not trials:
            continue
        elapsed = [t.get("elapsed_seconds") for t in trials if isinstance(t.get("elapsed_seconds"), (int, float))]
        n_records = max(
            (len(t.get("records") or []) for t in trials), default=0
        )
        per_question = (sum(elapsed) / (len(trials) * n_records)) if elapsed and n_records else 0.0
        total = sum(elapsed)
        total_seconds += total
        rows.append({
            "section": f"BO ({mode})",
            "trials": len(trials),
            "questions / trial": n_records,
            "total runtime (s)": f"{total:.1f}" if total else "-",
            "mean per question (s)": f"{per_question:.1f}" if per_question else "-",
        })
    for mode, payload in dspy.items():
        if not payload:
            continue
        n_q = len((payload or {}).get("records") or [])
        # GenerationOptimizer records per-phase wall time in
        # ``stage_elapsed`` (baseline_s / optimize_s / re_eval_s /
        # total_s). Older bundles lack it -> "-".
        se = (payload or {}).get("stage_elapsed") or {}
        total_s = se.get("total_s")
        if isinstance(total_s, (int, float)):
            total_seconds += float(total_s)
            phase_bits = " / ".join(
                f"{k.replace('_s','')} {se[k]:.0f}s"
                for k in ("baseline_s", "optimize_s", "re_eval_s")
                if isinstance(se.get(k), (int, float))
            )
            total_str = f"{total_s:.1f}" + (f"  ({phase_bits})" if phase_bits else "")
            per_q = f"{total_s / n_q:.1f}" if n_q else "-"
        else:
            total_str = "-"
            per_q = "-"
        rows.append({
            "section": f"DSPy ({mode})",
            "trials": "phases: baseline / optimize / re-eval",
            "questions / trial": n_q or "-",
            "total runtime (s)": total_str,
            "mean per question (s)": per_q,
        })
    if not rows:
        return ""
    out = ['<h2>Run cost</h2>']
    out.append('<p class="caption">Wall-clock cost of the optimization run, '
               'derived from per-trial elapsed times. Token / dollar accounting '
               'only renders when traces carry it (not by default).</p>')
    out.append(_table(rows, cols=[
        "section", "trials", "questions / trial",
        "total runtime (s)", "mean per question (s)",
    ]))
    if total_seconds:
        out.append(
            f'<p class="caption"><strong>Total tracked runtime:</strong> '
            f'{total_seconds:.1f}s '
            f'({total_seconds / 60.0:.1f} min)</p>'
        )
    return "\n".join(out)


def _diagnosis_view(bundle: dict, phase: str) -> dict[str, dict | None]:
    """Return ``{mode: report_dump | None}`` for the requested ``phase``
    (``"baseline"`` or ``"optimized"``). Handles both the new
    baseline/optimized-keyed schema and the legacy flat-per-mode schema
    (where the value is just the optimized report).
    """
    out: dict[str, dict | None] = {}
    diag_by_mode = bundle.get("diagnosis") or {}
    for mode, entry in diag_by_mode.items():
        if isinstance(entry, dict) and (
            "baseline" in entry or "optimized" in entry
        ):
            out[mode] = entry.get(phase)
        elif phase == "optimized":
            # Legacy: a flat per-mode report dict means the "optimized"
            # view (this was the original schema before we split into
            # baseline-vs-optimized; keep it readable to old bundles).
            out[mode] = entry
        else:
            out[mode] = None
    return out


def _render_layer_panel(layer: dict, *, label: str, sigil: str) -> str:
    """One per-source diagnosis layer rendered as a labelled panel.

    ``sigil`` is the short tag shown next to the layer label so the
    reader doesn't have to read the whole title to know whether they're
    looking at rule output vs LLM output vs synthesised output.
    """
    parts: list[str] = [f'<h4>{_esc(label)} <span class="tag">{_esc(sigil)}</span></h4>']
    summary = layer.get("summary") or ""
    if summary:
        parts.append(f'<div class="box">{_esc(summary)}</div>')

    hyps = layer.get("hypotheses") or []
    if hyps:
        rows = [
            {
                "severity": h.get("severity", "?"),
                "component": h.get("component", "?"),
                "root cause": h.get("root_cause", ""),
                "confidence": f"{h.get('confidence', 0.0):.2f}",
            }
            for h in hyps
        ]
        parts.append(_table(rows, cols=["severity", "component", "root cause", "confidence"]))

    cands = layer.get("optimization_candidates") or []
    if cands:
        parts.append(
            '<p><strong>Candidates:</strong> '
            + " ".join(f'<span class="tag">{_esc(c)}</span>' for c in cands)
            + '</p>'
        )
    return "\n".join(parts)


def _render_diagnosis_layers(report: dict) -> str:
    """Render the per-source layers (rule / LLM / synthesis) side-by-side.

    Phase 2: the JSON now carries ``rule_based`` / ``llm_based`` /
    ``synthesis`` subsections. Show whichever ones are populated so the
    reader can tell where each hypothesis came from. ``active_source``
    is highlighted to match the top-level fields downstream consumers
    see.
    """
    active = report.get("active_source") or "rule"
    panels: list[str] = []
    for key, label, sigil in (
        ("rule_based", "Rule-based", "rule"),
        ("llm_based", "LLM", "llm"),
        ("synthesis", "Synthesis", "both"),
    ):
        layer = report.get(key)
        if not isinstance(layer, dict):
            continue
        active_marker = " (active)" if key.split("_")[0] == active else ""
        panels.append(_render_layer_panel(
            layer, label=f"{label}{active_marker}", sigil=sigil,
        ))
    if not panels:
        return ""
    return (
        '<h4>Per-source diagnosis layers</h4>'
        '<p class="caption">Which signals come from where. <strong>Rule</strong> '
        '= deterministic causal-graph analysis. <strong>LLM</strong> = LLM-refined '
        'view of the same data. <strong>Synthesis</strong> = LLM-merged final '
        'answer combining both. The active layer drives the top-level summary above.</p>'
        + "\n".join(panels)
    )


def _render_layer_overview(layer_scores: dict) -> str:
    """Three-layer (retrieval / generation / e2e) score overview.

    Each layer renders as a horizontal bar sized + coloured by its
    aggregate score, with a ``<details>`` drop-down exposing the
    per-metric breakdown (oriented values, so lower-is-better metrics
    already inverted). Returns ``""`` when no layer has a score.
    """
    from ragdx.core.metrics import layer_catalog

    layers = ("retrieval", "generation", "e2e")
    have_any = any(
        isinstance((layer_scores.get(lyr) or {}).get("score"), (int, float))
        for lyr in layers
    )
    if not have_any and not layer_scores:
        return ""

    # The full catalog per layer is keyed off whatever metrics were
    # actually computed (the ``raw`` values in layer_scores).
    computed: dict[str, float] = {}
    for lyr in layers:
        computed.update((layer_scores.get(lyr) or {}).get("raw") or {})
    catalog = layer_catalog(computed)

    def _bar_color(s: float) -> str:
        # red < 0.6 < amber < 0.8 < green
        if s < 0.6:
            return "#dc2626"
        if s < 0.8:
            return "#d97706"
        return "#16a34a"

    from ragdx.core.metrics import METRIC_GLOSSARY

    parts: list[str] = [
        '<h4>Three-layer score overview</h4>',
        '<p class="caption">Aggregate score per layer (mean of the '
        'layer\'s <em>computed</em> metrics; lower-is-better metrics '
        'like <code>hallucination</code> are inverted so higher = '
        'healthier). Expand a layer to see every metric, its value, and '
        'its meaning. Metric weighting is configured up-front in the '
        'studio\'s experiment settings.</p>',
    ]
    for lyr in layers:
        d = layer_scores.get(lyr) or {}
        s = d.get("score")
        raw = d.get("raw") or {}
        oriented = d.get("metrics") or {}

        # The aggregate bar (id-less; the JS finds .ovbar-fill / .ovscore
        # within this .ovlayer container).
        if isinstance(s, (int, float)):
            pct = round(s * 100)
            color = _bar_color(s)
            score_txt = f"{s:.2f}"
        else:
            pct = 0
            color = "#cbd5e1"
            score_txt = "—"
        bar = (
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0">'
            f'<div style="width:90px;font-weight:600;text-transform:capitalize">{_esc(lyr)}</div>'
            f'<div style="flex:1;background:#eef0f2;border-radius:5px;height:20px;position:relative">'
            f'<div class="ovbar-fill" style="width:{pct}%;background:{color};height:100%;border-radius:5px"></div>'
            f'</div>'
            f'<div class="ovscore" style="width:48px;text-align:right;font-variant-numeric:tabular-nums">{score_txt}</div>'
            f'</div>'
        )

        # Full catalog detail: computed value OR requires-badge, each row
        # carrying meaning + direction.
        entries = catalog.get(lyr) or []
        n_computed = sum(1 for e in entries if e["computed"])
        rows = []
        for e in entries:
            name = e["name"]
            gloss = METRIC_GLOSSARY.get(name, {})
            direction = gloss.get("direction", "higher")
            arrow = "↑ higher better" if direction == "higher" else "↓ lower better"
            desc = gloss.get("desc", "")
            if e["computed"]:
                ov = oriented.get(name)
                rv = raw.get(name)
                if isinstance(rv, (int, float)):
                    inv = (
                        " (inverted for score)"
                        if isinstance(ov, (int, float)) and abs(rv - ov) > 1e-9
                        else ""
                    )
                    val_cell = _esc(f"{rv:.3f}{inv}")
                else:
                    val_cell = "—"
            else:
                badge = e["requires_label"] or "not computed"
                val_cell = (
                    f'<span class="tag" style="background:#f3f4f6;color:#6b7280">'
                    f'{_esc(badge)}</span>'
                )
            rows.append(
                f'<tr><td><code>{_esc(name)}</code></td>'
                f'<td>{val_cell}</td>'
                f'<td class="subtle">{_esc(arrow)}</td>'
                f'<td class="subtle">{_esc(desc)}</td></tr>'
            )
        detail = (
            '<table><thead><tr><th>metric</th><th>value / status</th>'
            '<th>direction</th><th>what it measures</th></tr>'
            f'</thead><tbody>{"".join(rows)}</tbody></table>'
        )
        parts.append(
            f'<div class="ovlayer" data-layer="{_esc(lyr)}">'
            + bar
            + _details(
                f"{lyr} — {n_computed}/{len(entries)} metric(s) computed",
                detail,
            )
            + '</div>'
        )
    return "\n".join(parts)


def _render_diagnosis_body(report: dict | None, *, mode: str) -> str:
    """One mode's diagnosis content (used by both baseline and optimized
    renderers). Returns an empty string when ``report`` is None or empty.
    """
    if not report:
        return ""

    parts: list[str] = [f'<h3>Mode: {_esc(mode)}</h3>']

    summary = report.get("summary") or ""
    confidence = report.get("diagnosis_confidence")
    active = report.get("active_source") or "rule"
    parts.append('<div class="metric-row">')
    parts.append(_metric(
        "Diagnosis confidence",
        f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "—",
    ))
    parts.append(_metric(
        "Causal signals", len(report.get("causal_signals") or [])
    ))
    parts.append(_metric(
        "Hypotheses", len(report.get("hypotheses") or [])
    ))
    parts.append(_metric(
        "Source", active,
    ))
    parts.append('</div>')
    if summary:
        parts.append(f'<div class="box">{_esc(summary)}</div>')

    # Three-layer overview (retrieval / generation / e2e). Bars sized
    # by each layer's aggregate score, expandable to the per-metric
    # breakdown. Renders nothing for legacy reports without layer_scores.
    overview_html = _render_layer_overview(report.get("layer_scores") or {})
    if overview_html:
        parts.append(overview_html)

    # Per-source layer panels (Phase 2). Renders nothing for legacy
    # reports that don't have the layers populated.
    layers_html = _render_diagnosis_layers(report)
    if layers_html:
        parts.append(layers_html)

    gaps = report.get("metric_gaps") or {}
    if gaps:
        parts.append('<h4>Metric gaps (below threshold)</h4>')
        # Phase 4d: each row gets an anchor id so the hypothesis
        # evidence below can hyperlink back to the underlying gap.
        gap_head = '<tr><th>metric</th><th>gap</th></tr>'
        gap_body = "".join(
            f'<tr id="gap-{_esc(k)}"><td>{_esc(k)}</td>'
            f'<td>{(_esc(f"{v:.4f}") if isinstance(v, (int, float)) else _esc(v))}</td></tr>'
            for k, v in gaps.items()
        )
        parts.append(f"<table><thead>{gap_head}</thead><tbody>{gap_body}</tbody></table>")

    signals = report.get("causal_signals") or []
    # Phase 4c: SVG diagram BEFORE the table -- visual reading first,
    # numeric drill-down second.
    svg = _render_causal_graph_svg(signals, report.get("causal_graph"))
    if svg:
        parts.append(svg)
    if signals:
        from ragdx.core.metrics import CAUSAL_NODE_GLOSSARY

        parts.append('<h4>Causal signals (top 8 by posterior)</h4>')
        parts.append(
            '<p class="caption">Each row is a hypothesized defect. '
            '<strong>posterior</strong> = how likely that defect is '
            'after weighing the metrics, traces, and causal propagation. '
            'Lower is healthier; the top row is the prime suspect. '
            '(The pre-evidence prior is omitted — it tracks the learned '
            'store and adds little per-report signal.)</p>'
        )
        # Phase 4d: anchor per node so evidence can link back.
        sig_head = (
            '<tr><th>node</th><th>layer</th>'
            '<th>posterior</th><th>what it means</th>'
            '<th>recommended experiment</th></tr>'
        )
        sig_rows_html: list[str] = []
        for s in signals[:8]:
            node = s.get("node", "")
            post_val = s.get("posterior", 0.0)
            post_str = f"{post_val:.2f}" if isinstance(post_val, (int, float)) else str(post_val)
            meaning = (CAUSAL_NODE_GLOSSARY.get(node) or {}).get("desc", "")
            sig_rows_html.append(
                f'<tr id="signal-{_esc(node)}">'
                f'<td><code>{_esc(node)}</code></td>'
                f'<td>{_esc(s.get("component", ""))}</td>'
                f'<td>{_esc(post_str)}</td>'
                f'<td class="subtle">{_esc(meaning)}</td>'
                f'<td>{_esc(s.get("recommended_experiment", ""))}</td>'
                '</tr>'
            )
        parts.append(
            f"<table><thead>{sig_head}</thead><tbody>{''.join(sig_rows_html)}</tbody></table>"
        )

    hyps = report.get("hypotheses") or []
    if hyps:
        # Phase 4d: evidence linkifier. Each evidence string typically
        # mentions a metric ("context_precision=0.55 is below ...") or
        # a causal-graph node ("Upstream propagation from
        # retrieval_recall_defect ..."). We turn matching tokens into
        # hyperlinks pointing at the metric_gaps / causal_signals rows
        # above (those rows now carry id="gap-<name>" / "signal-<node>"
        # anchors -- see the gap / signal renderers).
        gap_names = set(gaps.keys()) if gaps else set()
        signal_names = {s.get("node", "") for s in signals if s.get("node")}

        def _linkify_evidence(text: str) -> str:
            """Replace metric and signal mentions with anchor links.

            HTML-escapes first, then layers anchors over the escaped
            string so the user-supplied evidence text can never inject
            markup. Longest tokens first so e.g. ``retrieval_recall_defect``
            wins over ``recall``.
            """
            import re as _re
            esc_text = _esc(text)
            tokens = sorted(gap_names | signal_names, key=len, reverse=True)
            for tok in tokens:
                if not tok:
                    continue
                anchor = (
                    f'gap-{tok}' if tok in gap_names else f'signal-{tok}'
                )
                # Word-boundary match against the already-escaped text.
                pattern = _re.compile(rf'\b{_re.escape(tok)}\b')
                esc_text = pattern.sub(
                    f'<a href="#{anchor}"><code>{tok}</code></a>',
                    esc_text,
                )
            return esc_text

        parts.append('<h4>Hypotheses</h4>')
        for i, h in enumerate(hyps, 1):
            sev_class = _severity_class(h.get("severity", ""))
            tag = (
                f'<span class="tag {sev_class}">{_esc(h.get("severity", "?"))}'
                f'</span> · component: <code>{_esc(h.get("component", "?"))}</code> '
                f'· confidence {h.get("confidence", 0.0):.2f}'
            )
            title = f"{i}. {h.get('root_cause', '?')}"
            inner: list[str] = [f'<p>{tag}</p>']
            ev = h.get("evidence") or []
            if ev:
                inner.append('<h4>Evidence</h4><ul>')
                inner.extend(f'<li>{_linkify_evidence(e)}</li>' for e in ev)
                inner.append('</ul>')
            acts = h.get("recommended_actions") or []
            if acts:
                inner.append('<h4>Recommended actions</h4><ul>')
                inner.extend(f'<li>{_esc(a)}</li>' for a in acts)
                inner.append('</ul>')
            parts.append(_details(title, "".join(inner), open_=(i == 1)))

    prio = report.get("priority_actions") or []
    if prio:
        parts.append('<h4>Priority actions</h4><ul>')
        parts.extend(f'<li>{_esc(a)}</li>' for a in prio)
        parts.append('</ul>')

    disambig = report.get("disambiguation_actions") or []
    if disambig:
        parts.append('<h4>Disambiguation actions</h4><ul>')
        parts.extend(f'<li>{_esc(a)}</li>' for a in disambig)
        parts.append('</ul>')

    cands = report.get("optimization_candidates") or []
    if cands:
        parts.append(
            '<h4>Optimization candidates</h4><p>'
            + " ".join(f'<span class="tag">{_esc(c)}</span>' for c in cands)
            + '</p>'
        )
    return "\n".join(parts)


def _render_baseline_diagnosis(bundle: dict) -> str:
    """Diagnosis on the baseline (pre-optimization) config -- the *why*
    behind the optimization sections that follow. Placed near the top
    of the report so the reader sees the problem before the solution.
    """
    per_mode = _diagnosis_view(bundle, "baseline")
    bodies = [
        _render_diagnosis_body(rep, mode=m)
        for m, rep in per_mode.items()
    ]
    bodies = [b for b in bodies if b]
    if not bodies:
        return ""
    header = '<h2>Diagnosis (baseline) — what we set out to fix</h2>'
    caption = (
        '<p class="caption">Rule-based root-cause analysis at the '
        '<em>baseline</em> config (post-retrieval-BO, pre-prompt-tuning). '
        'This is the bottleneck picture that motivates the optimization '
        'sections below: causal-graph nodes with high posterior probability '
        'are what to attack first. The <strong>Optimization candidates</strong> '
        'tags map directly to the tools used in the sections below '
        '(Bayesian search, DSPy A/B). For an LLM-refined view, run '
        '<code>ragdx diagnose --use-both</code> on the saved run.</p>'
    )
    return header + caption + "\n".join(bodies)


def _render_optimized_diagnosis(bundle: dict) -> str:
    """Diagnosis on the optimized winner -- the *follow-up*: did we
    actually move the right metrics, and what (if anything) is still
    broken? Placed after the optimization sections.
    """
    per_mode = _diagnosis_view(bundle, "optimized")
    bodies = [
        _render_diagnosis_body(rep, mode=m)
        for m, rep in per_mode.items()
    ]
    bodies = [b for b in bodies if b]
    if not bodies:
        return ""
    # When the bundle carries no baseline diagnosis (the one-shot
    # ``ragdx experiment`` flow produces a single final diagnosis), this
    # IS the diagnosis -- title it plainly rather than implying a
    # baseline comparison that doesn't exist.
    has_baseline = any(_diagnosis_view(bundle, "baseline").values())
    if has_baseline:
        header = '<h2>Diagnosis (optimized) — what still needs attention</h2>'
        caption = (
            '<p class="caption">Same analysis re-run on the optimized config. '
            'Compare the posteriors here with the baseline diagnosis above: '
            'a node whose posterior dropped is a hypothesis we successfully '
            'attacked; a node whose posterior is still high is the next '
            'thing to tune.</p>'
        )
    else:
        header = '<h2>Diagnosis — final system</h2>'
        caption = (
            '<p class="caption">Rule-based root-cause analysis of the '
            'fully-optimized system (after Bayesian search + DSPy prompt '
            'tuning). The weakest layer and highest-posterior causal nodes '
            'are what to attack next. For an LLM-refined view, run '
            '<code>ragdx diagnose --use-both</code> on the saved run.</p>'
        )
    return header + caption + "\n".join(bodies)


def _render_one_comparison(mode: str, comp: dict) -> str:
    """Render a single mode's baseline vs optimized diagnosis delta."""
    parts: list[str] = [f'<h3>Mode: {_esc(mode)}</h3>']

    # Top-line story
    summary = comp.get("summary") or ""
    if summary:
        parts.append(f'<div class="box">{_esc(summary)}</div>')

    top_imp = comp.get("top_improvement")
    top_reg = comp.get("top_regression")
    parts.append('<div class="metric-row">')
    if top_imp:
        parts.append(_metric(
            f'Top improvement · {top_imp.get("node", "")}',
            f'{top_imp.get("baseline", 0.0):.2f} → '
            f'{top_imp.get("optimized", 0.0):.2f}  '
            f'(Δ {top_imp.get("delta", 0.0):+.2f})',
        ))
    if top_reg:
        parts.append(_metric(
            f'Top regression · {top_reg.get("node", "")}',
            f'{top_reg.get("baseline", 0.0):.2f} → '
            f'{top_reg.get("optimized", 0.0):.2f}  '
            f'(Δ {top_reg.get("delta", 0.0):+.2f})',
        ))
    parts.append(_metric("Resolved", len(comp.get("resolved_hypotheses") or [])))
    parts.append(_metric("Persisted", len(comp.get("persisted_hypotheses") or [])))
    parts.append(_metric("Emerged", len(comp.get("emerged_hypotheses") or [])))
    parts.append('</div>')

    # Hypothesis status grid
    def _hyp_rows(hyps: list, status: str) -> list[dict]:
        return [{
            "status": status,
            "component": h.get("component", ""),
            "root cause": h.get("root_cause", ""),
            "severity": h.get("severity", ""),
            "confidence": f'{h.get("confidence", 0.0):.2f}',
        } for h in hyps]

    all_rows: list[dict] = []
    all_rows += _hyp_rows(comp.get("resolved_hypotheses") or [], "✓ resolved")
    all_rows += _hyp_rows(comp.get("emerged_hypotheses") or [], "✗ emerged")
    all_rows += _hyp_rows(comp.get("persisted_hypotheses") or [], "→ persisted")
    if all_rows:
        parts.append('<h4>Hypothesis status</h4>')
        parts.append(_table(
            all_rows,
            cols=["status", "component", "root cause", "severity", "confidence"],
        ))

    # Posterior shifts (already filtered + sorted by |delta|)
    shifts = comp.get("posterior_shifts") or []
    if shifts:
        parts.append('<h4>Posterior shifts (|Δ| ≥ 0.05, sorted by magnitude)</h4>')
        rows = []
        for s in shifts:
            direction = s.get("direction", "")
            arrow = "↓" if direction == "improved" else "↑"
            sigil = "ok" if direction == "improved" else "warn"
            rows.append({
                "node": s.get("node", ""),
                "baseline": f'{s.get("baseline", 0.0):.2f}',
                "optimized": f'{s.get("optimized", 0.0):.2f}',
                "Δ": f'<span class="tag {sigil}">{arrow} {s.get("delta", 0.0):+.2f}</span>',
                "direction": direction,
            })
        # _table HTML-escapes values; build manually so the tag stays as HTML.
        head = "".join(
            f"<th>{_esc(c)}</th>"
            for c in ["node", "baseline", "optimized", "Δ", "direction"]
        )
        body = "".join(
            "<tr>"
            + f"<td>{_esc(r['node'])}</td>"
            + f"<td>{_esc(r['baseline'])}</td>"
            + f"<td>{_esc(r['optimized'])}</td>"
            + f"<td>{r['Δ']}</td>"
            + f"<td>{_esc(r['direction'])}</td>"
            + "</tr>"
            for r in rows
        )
        parts.append(
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
        )

    # Metric deltas (with LOWER_IS_BETTER direction-aware arrow)
    deltas = comp.get("metric_deltas") or {}
    lower_better = set(comp.get("lower_is_better_metrics") or [])
    if deltas:
        parts.append('<h4>Metric deltas (optimized - baseline)</h4>')
        rows = []
        for m, d in sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True):
            if abs(d) < 1e-9:
                continue
            improved = (d < 0) if m in lower_better else (d > 0)
            arrow = "↑" if d > 0 else "↓"
            sigil = "ok" if improved else "warn"
            rows.append({
                "metric": m,
                "delta": f'<span class="tag {sigil}">{arrow} {d:+.4f}</span>',
                "note": "lower-is-better" if m in lower_better else "",
            })
        if rows:
            head = "<th>metric</th><th>delta</th><th>note</th>"
            body = "".join(
                "<tr>"
                + f"<td>{_esc(r['metric'])}</td>"
                + f"<td>{r['delta']}</td>"
                + f"<td>{_esc(r['note'])}</td>"
                + "</tr>"
                for r in rows
            )
            parts.append(
                f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
            )

    return "\n".join(parts)


def _render_diagnosis_comparison(bundle: dict) -> str:
    """Render the baseline-vs-optimized delta story.

    Closes the optimization loop: did we actually move the metrics
    the baseline diagnosis pointed at? Did anything regress as a
    side-effect (a common failure mode: fixing recall exposes
    precision noise, so retrieval_precision_defect's posterior rises
    even though faithfulness went up)?
    """
    diag_by_mode = bundle.get("diagnosis") or {}
    bodies: list[str] = []
    for mode, entry in diag_by_mode.items():
        if not isinstance(entry, dict):
            continue
        comp = entry.get("comparison")
        if not comp:
            continue
        bodies.append(_render_one_comparison(mode, comp))
    if not bodies:
        return ""
    header = (
        '<h2>Diagnosis comparison — did the optimization move what it should?</h2>'
    )
    caption = (
        '<p class="caption">Baseline vs optimized diagnosis delta. '
        '<strong>Resolved</strong> = hypothesis present at baseline but '
        'gone after optimization (a clear win). '
        '<strong>Persisted</strong> = still present (next target). '
        '<strong>Emerged</strong> = appeared only after optimization '
        '(common after fixing one bottleneck: the next one becomes '
        'visible). Posterior shifts identify which causal-graph nodes '
        'moved most; a <em>regression</em> (posterior rising) is a side-effect '
        'worth investigating before celebrating.</p>'
    )
    return header + caption + "\n".join(bodies)


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
    from ragdx.core.metrics import METRIC_GLOSSARY

    parts = ['<h2>Composite objective (RAG Bayesian search)</h2>']
    parts.append(
        '<p class="caption"><strong>Metric weights</strong> combine into '
        'the composite score (<code>sum of weight_i * metric_i</code>) '
        'that the <strong>RAG-config Bayesian search</strong> optimizes '
        'and that the report uses for headline comparisons. The prompt '
        'optimizer (DSPy) selects its winner with its own inner-loop '
        'metric. Weights follow the practical '
        'priority hierarchy <em>groundedness &gt; correctness &gt; '
        'retrieval quality &gt; auxiliary judges</em>. Lower-is-better '
        'metrics (hallucination / bias / toxicity) are inverted '
        '(<code>1 - value</code>) before weighting, so every weight '
        'rewards the healthy direction. <strong>Constraints</strong> '
        'are hard pass/fail thresholds: a trial that violates one is '
        'still scored but marked <strong>infeasible</strong> so the '
        'planner can filter unsafe configs.</p>'
    )
    for m, spec in objs.items():
        weights = (spec or {}).get("metrics") or {}
        constraints = (spec or {}).get("constraints") or {}
        body = '<div>'
        body += '<h4>Metric weights</h4>'
        body += _table(
            [
                {
                    "metric": k,
                    "weight": v,
                    "what it measures": (METRIC_GLOSSARY.get(k) or {}).get("desc", "—"),
                }
                for k, v in sorted(weights.items(), key=lambda kv: -kv[1])
            ],
            cols=["metric", "weight", "what it measures"],
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
        body += '</div>'
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
                color_discrete_map={"composite": "#9aa0a6", "running best": "#1a73e8"},
            )
            fig.update_layout(height=360, margin=dict(t=60, b=40))
            parts.append(_fig_html(fig))
            parts.append(
                '<p class="caption"><strong>composite</strong> = this '
                "trial's composite score (per the objective above). "
                "<strong>running best</strong> = highest composite seen "
                "so far -- the non-decreasing curve shows BO convergence.</p>"
            )

        # Per-trial Q/A inspector — ALL trials, each toggleable via a
        # checkbox (item 6). First 3 are shown by default; the rest are
        # hidden until ticked, so the section stays readable but every
        # trial is one click away.
        records_keyed = [t for t in trials if t.get("records")]
        if records_keyed:
            checkboxes: list[str] = []
            blocks: list[str] = []
            for idx, t in enumerate(records_keyed):
                tindex = t.get("trial_index")
                block_id = f"trial-{_esc(mode)}-{idx}"
                shown_default = idx < 3
                checkboxes.append(
                    f'<label style="margin-right:14px;white-space:nowrap">'
                    f'<input type="checkbox" class="trial-toggle" '
                    f'data-target="{block_id}"{" checked" if shown_default else ""}> '
                    f'Trial #{_esc(tindex)}</label>'
                )

                # Phase 4a: union of metric columns across the trial's records.
                t_records = t.get("records") or []
                metric_cols: list[str] = []
                for r in t_records:
                    for k in (r.get("scores") or {}).keys():
                        if k not in metric_cols:
                            metric_cols.append(k)
                rec_rows = []
                for j, r in enumerate(t_records, 1):
                    row: dict[str, Any] = {
                        "#": j,
                        "question": r.get("question") or "",
                        "ground_truth": r.get("ground_truth") or "—",
                        "answer": r.get("answer") or "",
                    }
                    rec_scores = r.get("scores") or {}
                    for m in metric_cols:
                        v = rec_scores.get(m)
                        row[m] = f"{v:.3f}" if isinstance(v, (int, float)) else "-"
                    rec_rows.append(row)
                cols = ["#", "question", "ground_truth", "answer", *metric_cols]
                display = "" if shown_default else "display:none"
                blocks.append(
                    f'<div id="{block_id}" style="{display}">'
                    f'<h4>Trial #{_esc(tindex)} '
                    f'({_esc(_trial_param_label(t.get("params") or {}))})</h4>'
                    + _table(rec_rows, cols=cols)
                    + '</div>'
                )

            inspector = (
                '<p class="caption">Tick a trial to show / hide its '
                'per-record answers + metrics. The first three are shown '
                'by default.</p>'
                f'<div style="margin:8px 0;line-height:2">{"".join(checkboxes)}</div>'
                + "".join(blocks)
            )
            parts.append(_details(
                f"Per-record outputs ({len(records_keyed)} trials, toggleable)",
                inspector,
                open_=True,
            ))

    return "\n".join(parts)


def _render_dspy_a_b(bundle: dict) -> str:
    ab = bundle.get("dspy_a_b") or {}
    if not ab:
        return ""
    # Algorithm-aware label: MIPROv2 / COPRO / BootstrapFewShot / GEPA.
    # Falls back to a neutral "DSPy" when the bundle's pre-PR8 and
    # doesn't record which optimizer ran.
    _algo_label_map = {
        "mipro": "MIPROv2",
        "copro": "COPRO",
        "bootstrap_fewshot": "BootstrapFewShot",
        "gepa": "GEPA",
    }
    # Inspect the first mode's extras for ``dspy_optimizer_used``.
    _first_payload = next(iter(ab.values()), {}) or {}
    _algo_key = _first_payload.get("dspy_optimizer_used") or "mipro"
    _algo_label = _algo_label_map.get(_algo_key, "DSPy")
    parts = [f'<h2>Prompt optimization — before/after ({_esc(_algo_label)}, at the RAG winner config)</h2>']
    parts.append(
        '<p class="caption">Both columns are evaluated with the same '
        'pipeline, questions, and metric set as the rest of this report. '
        f'The final config ships <strong>the prompt {_esc(_algo_label)} '
        'selected with its own inner-loop metric</strong>; the composite '
        'scores below are shown for comparison.</p>'
    )

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

        # When the optimizer's winner IS the seed prompt (every candidate
        # rejected), say so — otherwise identical prompts with slightly
        # different scores read as a contradiction (the difference is
        # pure evaluation noise).
        _base_instr = next(iter((payload.get("baseline_instructions") or {}).values()), None)
        _win_instr = next(iter((payload.get("instructions") or {}).values()), None)
        if _base_instr is not None and _base_instr == _win_instr:
            parts.append(
                f'<p class="callout warn"><strong>{_esc(_algo_label)} kept the '
                'seed prompt</strong> — none of its candidates beat the baseline '
                'on its inner-loop metric, so both columns below evaluate the '
                '<em>same</em> prompt text. Any score difference between them '
                'is evaluation noise (LLM-judge nondeterminism), not a real '
                'improvement or regression.</p>'
            )

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

        # Optimizer trial-scores progression
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
                markers=True, title=f"{_algo_label} trial scores — {mode}",
                color_discrete_map={"per-trial": "#9aa0a6", "running best": "#1a73e8"},
            )
            fig2.update_layout(height=320, margin=dict(t=60, b=40))
            parts.append(_fig_html(fig2))
            parts.append(
                f'<p class="caption"><strong>per-trial</strong> = {_esc(_algo_label)}\'s '
                'inner-loop training-time score for that trial (token-F1 '
                'with-GT, LLM-judge without-GT — <strong>NOT</strong> the '
                'ragas composite). <strong>running best</strong> = highest '
                'score seen so far.</p>'
            )

        # Candidate instructions MIPROv2 explored (PR6+ capture).
        # This is the user-visible answer to "what prompts did DSPy
        # actually try?". The winner is one of these entries.
        proposed = payload.get("proposed_instructions_by_predictor") or {}
        opt_instr = payload.get("instructions") or {}
        trial_log = payload.get("trial_log") or []
        if proposed:
            parts.append(f'<h4>Candidate instructions {_esc(_algo_label)} proposed</h4>')
            parts.append(
                f'<p class="caption">{_esc(_algo_label)} generates candidate '
                '<code>system_instruction</code>s and evaluates each with '
                'its inner-loop metric. The winner (highlighted) is what '
                'ends up in <code>generator.system_instruction</code>. '
                f'A candidate count of 1 means {_esc(_algo_label)} only '
                'tried the seed — usually because the budget was too tight '
                'or the trainset was too small for the proposer.</p>'
            )
            for pname, cands in sorted(proposed.items()):
                winner = (opt_instr or {}).get(pname)
                body_parts = [f'<p class="subtle">{len(cands)} candidate(s) for predictor <code>{_esc(pname)}</code>.</p>']
                for i, cand in enumerate(cands):
                    is_winner = (winner is not None and cand.strip() == (winner or "").strip())
                    marker = ' ★ <strong>winner</strong>' if is_winner else ''
                    body_parts.append(
                        f'<details {"open" if is_winner else ""}>'
                        f'<summary>Candidate {i}{marker}</summary>'
                        f'<pre>{_esc(cand)}</pre>'
                        f'</details>'
                    )
                parts.append(_details(
                    f"Predictor `{pname}` — {len(cands)} candidate(s)",
                    "\n".join(body_parts),
                    open_=True,
                ))

        # Phase 4f: optimizer evolution timeline. Renders the candidate
        # sequence as a vertical timeline with per-candidate score, so
        # the reader can see "instruction N scored X, succeeded by N+1
        # scoring Y, ..." at a glance. Complements the table below.
        if proposed and trial_log:
            # Map params → score so we can attach a score to each
            # candidate index. Two formats:
            # * MIPROv2: params like 'Predictor 0: Instruction 1'.
            # * GEPA: params like 'Iter 2: proposed score 5.96 vs ...';
            #   candidates = [seed] + one proposal per iteration, so
            #   "Iter N" maps to candidate index N, and the seed
            #   (index 0) takes the baseline-program score (the trial
            #   whose kind mentions 'default'/'baseline').
            import re as _re
            score_by_idx: dict[str, dict[int, float]] = {}
            for t in trial_log:
                p = str(t.get("params") or "")
                k = str(t.get("kind") or "")
                s = t.get("score")
                if not isinstance(s, (int, float)):
                    continue
                for m in _re.finditer(r"Instruction (\d+)", p):
                    score_by_idx.setdefault("predict", {}).setdefault(
                        int(m.group(1)), float(s),
                    )
                gm = _re.match(r"Iter (\d+):", p)
                if gm:
                    score_by_idx.setdefault("predict", {}).setdefault(
                        int(gm.group(1)), float(s),
                    )
                if "default" in k or "baseline" in k or "seed" in p.lower():
                    score_by_idx.setdefault("predict", {}).setdefault(
                        0, float(s),
                    )

            timeline_html: list[str] = [
                '<h4>Candidate evolution timeline</h4>',
                '<p class="caption">Each row = one proposed instruction '
                'in proposal order. Score (when available) is the best '
                'trial score that picked this candidate. Winner marked '
                'with a star.</p>',
                '<ol style="border-left:2px solid var(--line);padding-left:18px;list-style:none">',
            ]
            for pname, cands in sorted(proposed.items()):
                winner = (opt_instr or {}).get(pname)
                scores = score_by_idx.get(pname, {})
                for i, cand in enumerate(cands):
                    is_winner = (
                        winner is not None
                        and cand.strip() == (winner or "").strip()
                    )
                    score = scores.get(i)
                    badge_color = "#15803d" if is_winner else "#6b7280"
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                    marker = " &#x2605; <strong>winner</strong>" if is_winner else ""
                    snippet = (cand[:160] + "...") if len(cand) > 160 else cand
                    timeline_html.append(
                        f'<li style="margin:8px 0;position:relative">'
                        f'<span style="position:absolute;left:-26px;top:4px;width:14px;'
                        f'height:14px;border-radius:50%;background:{badge_color};'
                        f'border:2px solid #fff"></span>'
                        f'<div><span class="tag">#{i}</span> '
                        f'<strong>score:</strong> {score_str}{marker}</div>'
                        f'<pre style="margin:4px 0;font-size:11px">{_esc(snippet)}</pre>'
                        f'</li>'
                    )
            timeline_html.append('</ol>')
            parts.append("".join(timeline_html))

        # Per-trial log: trial # → chosen (instruction_idx, demo_idx)
        # → score. Lets users see "trial 3 picked instruction 4, scored
        # 0.78". Built from the MIPROv2 logger.
        if trial_log:
            parts.append('<h4>Trial-by-trial decisions</h4>')
            parts.append(
                f'<p class="caption">Each row is one {_esc(_algo_label)} trial. '
                '<code>params</code> describes what the optimizer tried on '
                'that trial (candidate indices for MIPROv2, proposal-vs-prior '
                'summaries for GEPA). Combine with the candidate list above '
                'to see which prompt got tried on which trial.</p>'
            )
            rows_t = []
            for t in trial_log:
                rows_t.append({
                    "trial": t.get("trial", "?"),
                    "kind": t.get("kind", "?"),
                    "score": (
                        f"{t.get('score'):.3f}"
                        if isinstance(t.get("score"), (int, float))
                        else "—"
                    ),
                    "params": _esc(str(t.get("params") or "—")),
                })
            parts.append(_table(rows_t, cols=["trial", "kind", "score", "params"]))

        # Prompts: baseline vs optimized
        base_instr = payload.get("baseline_instructions") or {}
        base_demos = payload.get("baseline_demos") or {}
        opt_demos = payload.get("demos") or {}
        if base_instr or opt_instr:
            parts.append(f'<h4>Prompts: baseline vs {_esc(_algo_label)}-optimized</h4>')
            parts.append(
                '<p class="caption"><strong>Baseline</strong> = the DSPy '
                'signature\'s instruction at run start (your '
                '<code>--system-instruction</code> if set, else ragdx '
                f'default). <strong>Optimized</strong> = what {_esc(_algo_label)} '
                f'picked after its inner-loop search. Identical columns = '
                f'{_esc(_algo_label)} found no improvement and kept the seed '
                '(the right default when scores are tied).</p>'
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
                # Retrieved contexts: shared by baseline and optimized
                # answers in DSPy stages (the retriever is held fixed).
                # Rendering them lets the reader judge "did the
                # optimised answer use the same evidence" -- the most
                # frequent followup question after seeing a delta.
                ctxs = r.get("contexts") or []
                if ctxs:
                    if len(ctxs) == 1:
                        ctx_payload = ctxs[0]
                    else:
                        ctx_payload = "\n\n---\n\n".join(
                            f"[Context {i}]\n{c}"
                            for i, c in enumerate(ctxs, 1)
                        )
                    qa_rows.append({
                        "field": f"retrieved contexts ({len(ctxs)})",
                        "content": ctx_payload,
                    })
                qa_rows.append({"field": "baseline answer",
                                "content": r.get("baseline_answer", "")})
                qa_rows.append({"field": "optimized answer",
                                "content": r.get("optimized_answer", "")})
                inspector += _table(qa_rows, cols=["field", "content"])
                # Phase 4b: word-level diff between baseline and optimized.
                # Helps the reader spot where the prompt change actually
                # moved the wording vs where DSPy just kept the seed text.
                diff_html = _render_answer_diff(
                    r.get("baseline_answer", "") or "",
                    r.get("optimized_answer", "") or "",
                )
                if diff_html:
                    inspector += diff_html
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
    # Top-level parts: each is (anchor-id, part title, [section renderers]).
    # Empty parts (all sections render "") are skipped, and the TOC only
    # lists parts that actually appear.
    part_specs: list[tuple[str, str, list[str]]] = [
        ("experiment-config", "1 · Experiment Config", [
            _render_meta(bundle),
            _render_questions(bundle),
            _render_objectives(bundle),
            _render_diagnostics(bundle),
        ]),
        # Baseline diagnosis comes BEFORE the optimization sections
        # because it's the *why*: a baseline-driven flow looks at
        # the diagnosis to decide which directions to optimize.
        ("baseline-diagnosis", "2 · Baseline Diagnosis", [
            _render_baseline_diagnosis(bundle),
        ]),
        ("rag-optimize", "3 · RAG Optimization (Bayesian search)", [
            _render_bayes_search(bundle),
        ]),
        ("prompt-optimize", "4 · Prompt Optimization (DSPy)", [
            _render_dspy_a_b(bundle),
        ]),
        # Comparison first ("did it work?"), then the residual analysis
        # ("what's still broken at the final config?").
        ("final-diagnosis", "5 · Final Evaluation & Diagnosis", [
            _render_diagnosis_comparison(bundle),
            _render_optimized_diagnosis(bundle),
        ]),
        ("run-cost", "6 · Run Cost & Extras", [
            _render_run_cost(bundle),
            _render_extras(bundle),
        ]),
    ]
    parts_html: list[str] = []
    toc_items: list[str] = []
    for anchor, part_title, section_list in part_specs:
        content = "\n".join(s for s in section_list if s)
        if not content:
            continue
        toc_items.append(f'<li><a href="#{anchor}">{_esc(part_title)}</a></li>')
        parts_html.append(
            f'<section id="{anchor}">'
            f'<h1 class="part">{_esc(part_title)}</h1>\n{content}</section>'
        )
    toc = (
        '<nav class="toc"><div class="toc-title">Contents</div>'
        f'<ol>{"".join(toc_items)}</ol></nav>'
    )
    sections = toc + "\n" + "\n".join(parts_html)
    return _wrap(
        title or auto_title,
        sections,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


__all__ = ["render_report"]
