"""Streamlit dashboard for the PDF + no-GT optimization demo.

Loads ``.ragdx_pdf_no_gt_demo/result.json`` (produced by
``examples/demo_pdf_no_gt.py``) and presents the full pipeline:

* Run overview  -- PDF metadata, model, objective spec.
* Step 1        -- Synthesised questions (LLM-generated from the corpus).
* Step 2        -- AutoRAG Bayesian search over chunk_size + overlap +
                  top_k. Per-trial composite scores chart + table.
* Step 3        -- BO winner config callout.
* Step 4        -- DSPy before/after using the winner config: ragas
                  baseline vs optimised bar chart, composite delta,
                  MIPROv2 trial chart, optimised prompt + demos,
                  side-by-side sample-answer comparison.

Run::

    streamlit run src/ragdx/ui/pdf_no_gt_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parents[3]
LIVE_RESULT = REPO / ".ragdx_pdf_no_gt_demo" / "result.json"
COMMITTED_RESULT = REPO / "docs" / "examples" / "pdf_no_gt_result.json"
DEFAULT_RESULT = LIVE_RESULT if LIVE_RESULT.exists() else COMMITTED_RESULT


# ---------------------------------------------------------------- helpers
def load_bundle(path: Path) -> dict:
    if not path.exists():
        st.error(
            f"Result file not found: `{path}`.\n\n"
            "Generate it first with:\n\n"
            "```bash\n"
            "ZHIPU_API_KEY=<key> PYTHONPATH=src \\\n"
            "  python examples/demo_pdf_no_gt.py path/to/your.pdf\n"
            "```\n\n"
            "Or load the committed snapshot at "
            "`docs/examples/pdf_no_gt_result.json`."
        )
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, prec: int = 3) -> str:
    if v is None:
        return "N/A"
    try:
        fv = float(v)
        if fv != fv:
            return "NaN"
        return f"{fv:.{prec}f}"
    except (TypeError, ValueError):
        return str(v)


# -------------------------------------------------- run overview
def show_overview(bundle: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Model", bundle.get("model", "?"))
    cols[1].metric("Source PDF", bundle.get("source_pdf", "?"))
    pdf_meta = bundle.get("pdf_meta") or {}
    cols[2].metric("Pages", pdf_meta.get("page_count", "?"))
    cols[3].metric("Total chunks (default)", pdf_meta.get("chunks", "?"))

    obj = bundle.get("objective_spec") or {}
    if obj:
        st.write("**Composite objective in play (no-GT)**")
        st.json(obj, expanded=False)


# -------------------------------------------------- step 1: questions
def show_questions(bundle: dict) -> None:
    qs = bundle.get("questions") or []
    syn = bundle.get("synthesized_meta") or []
    rows = []
    for i, q in enumerate(qs):
        meta = syn[i] if i < len(syn) else {}
        rows.append({
            "#": i + 1,
            "question": q.get("question", ""),
            "source chunks": ", ".join(str(c) for c in (meta.get("source_chunk_ids") or [])),
            "ground_truth": q.get("ground_truth") or "(none — no-GT mode)",
        })
    if not rows:
        st.info("No questions in bundle.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"{len(rows)} questions synthesised by `synthesize_questions()` from "
        "random chunk pairs. Ground-truth is intentionally absent — this is "
        "the cold-start no-GT scenario."
    )


# -------------------------------------------------- step 2: BO search
def show_bo_search(bundle: dict) -> None:
    bo = bundle.get("autorag_bo") or {}
    trials = bo.get("trials") or []
    if not trials:
        st.info("No BO trials in this bundle.")
        return

    st.write("**Search space**")
    st.json(bo.get("search_space") or {}, expanded=False)
    cols = st.columns(3)
    cols[0].metric("n_init (random)", bo.get("n_init", "?"))
    cols[1].metric("max_trials", bo.get("max_trials", "?"))
    cols[2].metric("Total grid size",
                   _grid_size(bo.get("search_space") or {}))

    # Trial composite-score progression bar + running best
    composites = [t.get("composite_score") for t in trials]
    running_best, rb = [], -float("inf")
    for s in composites:
        if s is not None:
            rb = max(rb, float(s))
        running_best.append(rb if rb != -float("inf") else None)

    line_df = pd.DataFrame({
        "trial": [t.get("trial_index", i) + 1 for i, t in enumerate(trials)],
        "composite": composites,
        "running best": running_best,
        "feasible": [t.get("feasible", True) for t in trials],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=line_df["trial"], y=line_df["composite"],
        marker_color=[("#1f77b4" if f else "#aaaaaa") for f in line_df["feasible"]],
        name="trial composite",
        text=[_fmt(s) for s in line_df["composite"]],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=line_df["trial"], y=line_df["running best"],
        mode="lines+markers", name="running best",
        line=dict(color="#d62728", width=2),
    ))
    fig.update_layout(
        title="BO trial-by-trial composite scores",
        xaxis_title="trial", yaxis_title="composite score",
        height=380, margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-trial detail table
    st.write("**Per-trial detail**")
    table_rows = []
    for t in trials:
        params = t.get("params") or {}
        row = {
            "trial": (t.get("trial_index", 0) or 0) + 1,
            **{f"param_{k}": v for k, v in params.items()},
            "n_chunks": t.get("n_chunks", ""),
            "composite": _fmt(t.get("composite_score")),
            "feasible": t.get("feasible", True),
        }
        for k, v in (t.get("scores") or {}).items():
            row[k] = _fmt(v)
        table_rows.append(row)
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with st.expander("Sample generated answers per trial", expanded=False):
        for t in trials:
            params = t.get("params") or {}
            st.markdown(
                f"### Trial {(t.get('trial_index', 0) or 0) + 1} "
                f"-- {params} (composite={_fmt(t.get('composite_score'))})"
            )
            for i, ans in enumerate((t.get("answers_preview") or [])[:5]):
                st.markdown(f"- Q{i + 1}: `{(ans or '')[:240]}`")


def _grid_size(space: dict) -> int:
    n = 1
    for v in space.values():
        n *= max(1, len(v))
    return n


# -------------------------------------------------- step 3: winner
def show_winner(bundle: dict) -> None:
    bo = bundle.get("autorag_bo") or {}
    best_params = bo.get("best_params") or {}
    best_comp = bo.get("best_composite")
    if not best_params:
        return
    cols = st.columns(2)
    with cols[0]:
        st.success(f"**Best config**\n\n```\n{json.dumps(best_params, indent=2)}\n```")
    with cols[1]:
        st.metric("Composite score (winner)", _fmt(best_comp))
        st.caption(
            f"Selected by BO out of {len(bo.get('trials') or [])} trials "
            f"(grid size {_grid_size(bo.get('search_space') or {})})."
        )


# -------------------------------------------------- step 4: DSPy A/B
def show_dspy(bundle: dict) -> None:
    ba = bundle.get("dspy_before_after") or {}
    if not ba:
        st.info("No DSPy before/after in this bundle.")
        return
    bs = ba.get("baseline_scores") or {}
    os_ = ba.get("optimized_scores") or {}
    delta = ba.get("delta") or {}

    # Per-metric grouped bar chart
    metrics = sorted(set(bs) | set(os_))
    rows = []
    for m in metrics:
        rows.append({"metric": m, "phase": "baseline", "score": bs.get(m)})
        rows.append({"metric": m, "phase": "optimized", "score": os_.get(m)})
    if rows:
        df = pd.DataFrame(rows)
        df["score_display"] = df["score"].fillna(0.0)
        fig = px.bar(
            df, x="metric", y="score_display", color="phase", barmode="group",
            text=df["score"].apply(lambda v: _fmt(v)),
            color_discrete_map={"baseline": "#9aa0a6", "optimized": "#1a73e8"},
            title="DSPy: ragas scores baseline vs optimized",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(range=[0, 1.05], title="score"),
            xaxis_tickangle=-20, height=380, margin=dict(t=60, b=70),
            legend_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Composite delta tiles
    comp = ba.get("composite")
    if comp:
        b = comp["baseline"]["score"]
        o = comp["optimized"]["score"]
        d = comp["delta"]
        cols = st.columns(3)
        cols[0].metric("Composite baseline", _fmt(b))
        cols[1].metric("Composite optimized", _fmt(o), delta=f"{d:+.3f}")
        weights = (comp.get("objective_spec") or {}).get("metrics", {})
        cols[2].metric("Weights",
                       ", ".join(f"{k}={v}" for k, v in weights.items()))

    # Per-metric delta table
    if delta:
        rows_d = []
        for m in metrics:
            d = delta.get(m)
            arrow = "→"
            if isinstance(d, (int, float)):
                if d > 0:
                    arrow = "↑"
                elif d < 0:
                    arrow = "↓"
            rows_d.append({
                "metric": m,
                "baseline": _fmt(bs.get(m)),
                "optimized": _fmt(os_.get(m)),
                "delta": f"{arrow} {d:+.3f}" if isinstance(d, (int, float)) else "N/A",
            })
        st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

    # MIPROv2 trial chart (inner-loop scores, NOT ragas)
    trial_scores = ba.get("trial_scores") or []
    if trial_scores:
        running_best, rb = [], -float("inf")
        for s in trial_scores:
            rb = max(rb, s)
            running_best.append(rb)
        line_df = pd.DataFrame({
            "trial": list(range(1, len(trial_scores) + 1)),
            "score": trial_scores,
            "running best": running_best,
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=line_df["trial"], y=line_df["score"], name="trial score",
            marker_color="#1f77b4",
            text=[_fmt(s, 1) for s in line_df["score"]],
            textposition="outside",
        ))
        fig2.add_trace(go.Scatter(
            x=line_df["trial"], y=line_df["running best"],
            mode="lines+markers", name="running best",
            line=dict(color="#d62728", width=2),
        ))
        fig2.update_layout(
            title="MIPROv2 inner-loop trial scores (LLM-as-judge, no-GT mode)",
            xaxis_title="trial", yaxis_title="score",
            height=320, margin=dict(t=40, b=30),
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Optimised instructions + demos
    st.write("### Optimized prompt artifacts")
    instructions = ba.get("instructions") or {}
    if not instructions:
        st.info("No instructions returned by MIPROv2.")
    for name, instr in instructions.items():
        with st.expander(f"Instructions for `{name}`", expanded=False):
            st.code(instr or "(empty)", language="text")
    demos = ba.get("demos") or {}
    for name, ds in demos.items():
        with st.expander(f"Few-shot demos for `{name}` ({len(ds)})", expanded=False):
            for d in ds:
                st.json(d)

    # Sample answer comparison
    ba_ans = ba.get("baseline_sample_answers") or []
    op_ans = ba.get("optimized_sample_answers") or []
    if ba_ans or op_ans:
        st.write("### Sample answer comparison")
        for i in range(max(len(ba_ans), len(op_ans))):
            st.markdown(f"**Q{i + 1}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("_baseline_")
                st.code(ba_ans[i] if i < len(ba_ans) else "(missing)", language="text")
            with c2:
                st.markdown("_optimized_")
                st.code(op_ans[i] if i < len(op_ans) else "(missing)", language="text")


# -------------------------------------------------------------- main page
def main() -> None:
    st.set_page_config(
        page_title="ragdx · PDF no-GT pipeline",
        page_icon=":page_facing_up:",
        layout="wide",
    )

    st.title("ragdx end-to-end PDF (no-GT) pipeline dashboard")
    st.caption(
        "PDF → synthesised questions → Bayesian AutoRAG search → "
        "DSPy before/after. Produced by `examples/demo_pdf_no_gt.py`."
    )

    with st.sidebar:
        st.header("Load result bundle")
        path_str = st.text_input("Path to result.json", value=str(DEFAULT_RESULT))
        path = Path(path_str)
        if st.button("Reload"):
            st.cache_data.clear()
        st.markdown(
            "Re-generate with:\n\n"
            "```bash\n"
            "ZHIPU_API_KEY=<key> PYTHONPATH=src \\\n"
            "  python examples/demo_pdf_no_gt.py path/to/your.pdf\n"
            "```"
        )

    bundle = load_bundle(path)

    st.subheader("Run overview")
    show_overview(bundle)

    st.divider()
    st.subheader("Step 1 — Synthesised questions (cold-start, no GT)")
    st.caption(
        "`synthesize_questions()` randomly sampled chunk pairs from the PDF "
        "corpus and asked GLM-4-Flash for one factual question per pair. "
        "These seed the eval; the LLM never sees a reference answer."
    )
    show_questions(bundle)

    st.divider()
    st.subheader("Step 2 — AutoRAG Bayesian search")
    st.caption(
        "Multi-dim search over `(chunk_size, chunk_overlap, top_k)`. "
        "Each trial re-chunks + re-embeds the PDF (cached per "
        "(chunk_size, chunk_overlap) pair) and scores the resulting RAG "
        "answers with ragas + the production composite objective. "
        "Grey bars are infeasible (constraint violations)."
    )
    show_bo_search(bundle)

    st.divider()
    st.subheader("Step 3 — BO winner")
    show_winner(bundle)

    st.divider()
    st.subheader("Step 4 — DSPy before/after at the BO-winning config")
    st.caption(
        "Baseline = default RAG signature on the winner's retrieval. "
        "Optimized = MIPROv2-tuned program on the same records. Both are "
        "re-evaluated with the same ragas metric set so the comparison is "
        "fair. The trial chart below shows MIPROv2's inner-loop (LLM-as-"
        "judge) scores, NOT the ragas scores above."
    )
    show_dspy(bundle)


if __name__ == "__main__":
    main()
