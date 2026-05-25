"""Streamlit dashboard for the ragdx end-to-end optimization demo.

Loads ``.ragdx_optimize_demo/result.json`` (produced by
``examples/demo_optimize_gt_modes.py``) and renders:

* Data diagnostics (table)
* AutoRAG / DSPy specs in both GT modes (descriptive)
* Metric pre-flight (kept vs skipped with reasons)
* AutoRAG REAL grid search results (bar chart per top_k, best-pick callout)
* DSPy before/after at the AutoRAG winner config (side-by-side bar chart,
  per-metric delta, sample-answer comparison)
* DSPy MIPROv2 trial chart (with-GT vs no-GT contrast)

Run::

    streamlit run src/ragdx/ui/optimization_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parents[3]
LIVE_RESULT = REPO / ".ragdx_optimize_demo" / "result.json"
COMMITTED_RESULT = REPO / "docs" / "examples" / "optimize_gt_modes_result.json"
DEFAULT_RESULT = LIVE_RESULT if LIVE_RESULT.exists() else COMMITTED_RESULT


# ---------------------------------------------------------------- helpers
def load_bundle(path: Path) -> dict:
    if not path.exists():
        st.error(
            f"Result file not found: `{path}`.\n\n"
            "Generate it first with:\n\n"
            "```bash\n"
            "ZHIPU_API_KEY=<key> PYTHONPATH=src \\\n"
            "  python examples/demo_optimize_gt_modes.py\n"
            "```\n\n"
            "Or load the committed snapshot at "
            "`docs/examples/optimize_gt_modes_result.json`."
        )
        st.stop()
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostics_table(d: dict) -> dict[str, list[str]]:
    return {
        "field": ["has_ground_truth", "has_answers", "has_contexts", "gt_mode"],
        "with-GT": [
            str(d["with_gt"]["has_ground_truth"]),
            str(d["with_gt"]["has_answers"]),
            str(d["with_gt"]["has_contexts"]),
            d["with_gt"]["gt_mode"],
        ],
        "no-GT": [
            str(d["no_gt"]["has_ground_truth"]),
            str(d["no_gt"]["has_answers"]),
            str(d["no_gt"]["has_contexts"]),
            d["no_gt"]["gt_mode"],
        ],
    }


def show_spec_panel(panel: dict) -> None:
    cols = st.columns([2, 1])
    cols[0].metric("Objective metric", panel["objective_metric"])
    cols[1].metric("Requires GT", "Yes" if panel["requires_gt"] else "No")
    st.write(f"**Selected metrics ({len(panel['metrics'])})**")
    st.write(", ".join(panel["metrics"]))
    with st.expander("Rendered evaluation block", expanded=False):
        st.json(panel["yaml_template"]["evaluation"])
    st.caption(panel["note"])


def show_filter_panel(panel: dict) -> None:
    st.write("**Requested**")
    st.write(", ".join(panel["requested"]))
    st.write("**Kept**")
    st.write(", ".join(panel["kept"]) if panel["kept"] else "_(none -- pre-flight rejected all)_")
    if panel["skipped"]:
        st.write("**Skipped (with reasons)**")
        for m, reason in panel["skipped"].items():
            st.write(f"- `{m}` -- {reason}")
    else:
        st.success("Every requested metric is supported by the data.")


# ---------------------- AutoRAG grid chart
def show_autorag_grid(grid: dict) -> None:
    runs = grid.get("runs") or []
    if not runs:
        st.info("No AutoRAG grid runs in this bundle.")
        return
    rows = []
    for r in runs:
        scores = r.get("scores") or {}
        for m, v in scores.items():
            rows.append({"top_k": r["top_k"], "metric": m, "score": v})
    if not rows:
        st.warning("Grid ran but produced no scores -- ragas eval may have errored.")
        return
    df = pd.DataFrame(rows)
    objective = grid.get("objective", "faithfulness")
    best_k = grid.get("best_top_k")

    # Highlight best config with a column-marker (text annotation), since
    # plotly's add_vline can't snap to a categorical position reliably.
    df["display_top_k"] = df["top_k"].apply(
        lambda k: f"{k} (best)" if k == best_k else str(k)
    )
    fig = px.bar(
        df, x="display_top_k", y="score", color="metric", barmode="group",
        text=df["score"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"),
        title=f"AutoRAG grid -- per-config ragas scores (objective={objective})",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis=dict(type="category", title="top_k"),
        yaxis=dict(range=[0, 1.05], title="score"),
        height=420, margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"Best config -> **top_k = {best_k}** with "
        f"`{objective} = {grid['best_scores'].get(objective, float('nan')):.3f}`"
    )

    with st.expander("Per-config detail (scores + sample answers)", expanded=False):
        for r in runs:
            st.markdown(f"### top_k = {r['top_k']}")
            st.json(r.get("scores", {}))
            if r.get("answers"):
                st.markdown("**Sample generated answers**")
                for i, ans in enumerate(r["answers"][:3]):
                    st.markdown(f"- Q{i + 1}: `{(ans or '')[:200]}`")


# ---------------------- DSPy before/after chart
def show_dspy_before_after(panel: dict) -> None:
    bs = panel.get("baseline_scores") or {}
    os_ = panel.get("optimized_scores") or {}
    delta = panel.get("delta") or {}

    if not bs and not os_:
        st.info("No before/after scores in this bundle.")
        return

    metrics = sorted(set(bs) | set(os_))
    rows = []
    for m in metrics:
        rows.append({"metric": m, "phase": "baseline", "score": bs.get(m)})
        rows.append({"metric": m, "phase": "optimized", "score": os_.get(m)})
    df = pd.DataFrame(rows)
    df["score_display"] = df["score"].fillna(0.0)

    fig = px.bar(
        df, x="metric", y="score_display", color="phase", barmode="group",
        text=df["score"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"),
        color_discrete_map={"baseline": "#9aa0a6", "optimized": "#1a73e8"},
        title="DSPy: baseline vs optimized (same RAG config, ragas scores)",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="score"),
        xaxis_tickangle=-25, height=420, margin=dict(t=60, b=80),
        legend_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta table
    rows_d = []
    for m in metrics:
        d = delta.get(m)
        arrow = "→"
        if d is not None:
            if d > 0:
                arrow = "↑"
            elif d < 0:
                arrow = "↓"
        rows_d.append({
            "metric": m,
            "baseline": f"{bs.get(m):.3f}" if bs.get(m) is not None else "N/A",
            "optimized": f"{os_.get(m):.3f}" if os_.get(m) is not None else "N/A",
            "delta": f"{arrow} {d:+.3f}" if isinstance(d, (int, float)) else "N/A",
        })
    st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

    # Trial-by-trial chart
    trial_scores = panel.get("trial_scores") or []
    if trial_scores:
        st.write("### MIPROv2 trial-by-trial (training-time metric, not ragas)")
        running_best, rb = [], -float("inf")
        for s in trial_scores:
            rb = max(rb, s)
            running_best.append(rb)
        line_df = pd.DataFrame(
            {
                "trial": list(range(1, len(trial_scores) + 1)),
                "score": trial_scores,
                "running best": running_best,
            }
        )
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(x=line_df["trial"], y=line_df["score"], name="trial score",
                   marker_color="#1f77b4",
                   text=[f"{s:.2f}" for s in line_df["score"]],
                   textposition="outside")
        )
        fig2.add_trace(
            go.Scatter(x=line_df["trial"], y=line_df["running best"],
                       mode="lines+markers", name="running best",
                       line=dict(color="#d62728", width=2))
        )
        fig2.update_layout(xaxis_title="trial", yaxis_title="score",
                           height=320, margin=dict(t=40, b=40),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    # Optimised instructions + demos
    st.write("### Optimized instructions")
    if not panel.get("instructions"):
        st.info("No instructions returned.")
    for name, instr in panel.get("instructions", {}).items():
        st.write(f"**Predictor `{name}`**")
        st.code(instr or "(empty)", language="text")

    st.write("### Bootstrapped few-shot demos")
    if not panel.get("demos"):
        st.info("No demos returned.")
    for name, demos in panel.get("demos", {}).items():
        st.write(f"**Predictor `{name}` -- {len(demos)} demo(s)**")
        for i, d in enumerate(demos):
            with st.expander(f"demo[{i}]", expanded=(i == 0)):
                st.json(d)

    # Sample answers diff
    ba = panel.get("baseline_sample_answers") or []
    oa = panel.get("optimized_sample_answers") or []
    if ba or oa:
        st.write("### Sample answer comparison")
        for i in range(max(len(ba), len(oa))):
            st.markdown(f"**Q{i + 1}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("_baseline_")
                st.code(ba[i] if i < len(ba) else "(missing)", language="text")
            with c2:
                st.markdown("_optimized_")
                st.code(oa[i] if i < len(oa) else "(missing)", language="text")


# ---------------------- DSPy descriptive trial-chart (per GT mode)
def show_dspy_trial_only(panel: dict) -> None:
    if panel.get("error") and not panel.get("trial_scores"):
        st.error(f"MIPROv2 run failed: {panel['error']}")
        st.caption(
            "This often happens when the upstream LLM rate-limits or "
            "drops the connection mid-run. Re-running the demo usually "
            "fixes it."
        )
        return
    cols = st.columns(4)
    cols[0].metric("Optimizer", panel.get("optimizer", "MIPROv2"))
    cols[1].metric("Trainset", panel.get("trainset_size", 0))
    cols[2].metric("GT mode", panel.get("gt_mode", "?"))
    best = max(panel.get("best_score_progression", [0.0]) or [0.0])
    cols[3].metric("Best score", f"{best:.2f}")

    trial_scores = panel.get("trial_scores") or []
    if trial_scores:
        running_best, rb = [], -float("inf")
        for s in trial_scores:
            rb = max(rb, s)
            running_best.append(rb)
        df = pd.DataFrame({
            "trial": list(range(1, len(trial_scores) + 1)),
            "score": trial_scores,
            "running best": running_best,
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["trial"], y=df["score"], name="trial",
                             marker_color="#1f77b4",
                             text=[f"{s:.1f}" for s in df["score"]],
                             textposition="outside"))
        fig.add_trace(go.Scatter(x=df["trial"], y=df["running best"],
                                 mode="lines+markers", name="running best",
                                 line=dict(color="#d62728")))
        fig.update_layout(height=300, margin=dict(t=30, b=30),
                          xaxis_title="trial", yaxis_title="score",
                          legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)

    for name, instr in panel.get("instructions", {}).items():
        with st.expander(f"Optimized instructions for `{name}`", expanded=False):
            st.code(instr or "(empty)", language="text")


# -------------------------------------------------------------- main page
def main() -> None:
    st.set_page_config(
        page_title="ragdx · GT-aware optimization",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("ragdx end-to-end RAG optimization dashboard")
    st.caption(
        "AutoRAG grid search + DSPy before/after, side-by-side with the "
        "with-GT / no-GT contrast. Produced by "
        "`examples/demo_optimize_gt_modes.py`."
    )

    with st.sidebar:
        st.header("Load result bundle")
        path_str = st.text_input("Path to result.json", value=str(DEFAULT_RESULT))
        path = Path(path_str)
        if st.button("Reload"):
            st.cache_data.clear()
        st.markdown(
            "If the file is missing, run the demo first:\n\n"
            "```bash\n"
            "ZHIPU_API_KEY=<key> PYTHONPATH=src \\\n"
            "  python examples/demo_optimize_gt_modes.py\n"
            "```"
        )

    bundle = load_bundle(path)

    st.subheader("Run overview")
    cols = st.columns(4)
    cols[0].metric("Model", bundle.get("model", "?"))
    cols[1].metric("Dataset", bundle.get("dataset", "?"))
    cols[2].metric("Questions", bundle.get("corpus_size", 0))
    cols[3].metric("Corpus chunks", bundle.get("corpus_chunks", 0))

    # -- Step 1: data diagnostics
    st.divider()
    st.subheader("Step 1 — Data diagnostics")
    st.caption("Same records used twice; only `ground_truth` is erased in the no-GT branch.")
    st.table(diagnostics_table(bundle["data_diagnostics"]))

    # -- Step 2: AutoRAG specs (descriptive)
    if "autorag_spec" in bundle:
        st.divider()
        st.subheader("Step 2 — AutoRAG specs (descriptive)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### with-GT")
            show_spec_panel(bundle["autorag_spec"]["with_gt"])
        with c2:
            st.markdown("#### no-GT")
            show_spec_panel(bundle["autorag_spec"]["no_gt"])

    # -- Step 3: pre-flight
    if "metric_filter" in bundle:
        st.divider()
        st.subheader("Step 3 — Metric pre-flight")
        st.caption(
            "Same full ragas-style metric request goes to both branches. "
            "The pre-flight drops what the data can't support."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### with-GT")
            show_filter_panel(bundle["metric_filter"]["with_gt"])
        with c2:
            st.markdown("#### no-GT")
            show_filter_panel(bundle["metric_filter"]["no_gt"])

    # -- Step 4: AutoRAG REAL grid search
    if "autorag_grid" in bundle:
        st.divider()
        st.subheader("Step 4 — AutoRAG REAL grid search")
        st.caption(
            "For each `top_k`, the demo actually built a RAG pipeline "
            "(HuggingFace embeddings + FAISS + GLM-4-Flash), generated "
            "answers, and evaluated them with ragas. The winner becomes "
            "the fixed config for step 5."
        )
        show_autorag_grid(bundle["autorag_grid"])

    # -- Step 5: DSPy before/after
    if "dspy_before_after" in bundle:
        st.divider()
        st.subheader("Step 5 — DSPy before/after at the AutoRAG winner config")
        st.caption(
            "Baseline = default `RAGSignature` running on records "
            "pre-retrieved with the best `top_k`. Optimized = the same "
            "records run through the MIPROv2-tuned program. Both phases "
            "are scored with the SAME ragas metrics so the comparison is fair."
        )
        show_dspy_before_after(bundle["dspy_before_after"])

    # -- Step 6: DSPy descriptive trial charts (GT-mode contrast)
    if "dspy_descriptive" in bundle:
        st.divider()
        st.subheader("Step 6 — DSPy MIPROv2 trial chart (GT-mode contrast)")
        st.caption(
            "MIPROv2 ran in both GT modes purely to demonstrate the "
            "metric_kind difference: with-GT uses token-F1 against "
            "`example.answer` (cheap, deterministic), no-GT uses the "
            "built-in `FaithfulnessJudge` LLM-as-judge."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### with-GT (token-F1 metric)")
            show_dspy_trial_only(bundle["dspy_descriptive"]["with_gt"])
        with c2:
            st.markdown("#### no-GT (LLM-as-judge)")
            show_dspy_trial_only(bundle["dspy_descriptive"]["no_gt"])

    # -- questions appendix
    if bundle.get("questions"):
        st.divider()
        with st.expander("Questions + ground-truth (amnesty_qa)", expanded=False):
            for i, q in enumerate(bundle["questions"]):
                st.markdown(f"**Q{i + 1}.** {q['question']}")
                st.markdown(f"_ground_truth_: {q['ground_truth']}")
                st.markdown("---")


if __name__ == "__main__":
    main()
