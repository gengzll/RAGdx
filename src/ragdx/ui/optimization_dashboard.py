"""Streamlit dashboard for the GT-aware optimization demo.

Loads ``.ragdx_optimize_demo/result.json`` (produced by
``examples/demo_optimize_gt_modes.py``) and presents a side-by-side view of
the with-GT and no-GT pipelines:

* Data diagnostics: did the trainset have GT / answers / contexts?
* AutoRAG specs: which metrics get included, what objective gets chosen,
  what does the rendered evaluation block look like?
* Metric pre-flight: the same set of metrics fed into the GT-aware filter,
  showing what was kept and what was dropped (with reasons).
* Actual evaluation scores: side-by-side bar chart of ragas + embedding
  proxy results, plus a table showing N/A for metrics that the no-GT
  branch had to skip.
* DSPy optimisation: optimized instructions, bootstrapped few-shot demos,
  and a line chart of the MIPROv2 trial-by-trial scores.

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


def show_autorag_panel(panel: dict) -> None:
    cols = st.columns([2, 1])
    cols[0].metric("Objective metric", panel["objective_metric"])
    cols[1].metric("Requires GT", "Yes" if panel["requires_gt"] else "No")
    st.write(f"**Selected metrics ({len(panel['metrics'])})**")
    st.write(", ".join(panel["metrics"]))
    with st.expander("Rendered evaluation block (YAML excerpt)", expanded=False):
        st.json(panel["yaml_template"]["evaluation"])
    st.caption(panel["note"])


def show_filter_panel(panel: dict) -> None:
    st.write("**Requested**")
    st.write(", ".join(panel["requested"]))
    st.write("**Kept**")
    st.write(", ".join(panel["kept"]) if panel["kept"] else "_(none — pre-flight rejected all)_")
    if panel["skipped"]:
        st.write("**Skipped (with reasons)**")
        for m, reason in panel["skipped"].items():
            st.write(f"- `{m}` — {reason}")
    else:
        st.success("Every requested metric is supported by the data.")


# ---------------------------------------------------- evaluation helpers
def _merge_bucket(pack: dict) -> dict[str, float]:
    """Flatten ragdx EvaluationResult-style buckets into a single dict."""
    out: dict[str, float] = {}
    for b in ("retrieval", "generation", "e2e"):
        if isinstance(pack.get(b), dict):
            for k, v in pack[b].items():
                if isinstance(v, (int, float)):
                    out[k] = float(v)
    return out


def evaluation_comparison_dataframe(evaluation: dict, source: str) -> pd.DataFrame:
    """Return a long-form DataFrame with columns metric, mode, score."""
    with_gt_pack = evaluation["with_gt"].get(source, {})
    no_gt_pack = evaluation["no_gt"].get(source, {})

    if "error" in with_gt_pack or "error" in no_gt_pack:
        return pd.DataFrame()

    with_scores = _merge_bucket(with_gt_pack)
    no_scores = _merge_bucket(no_gt_pack)
    metric_names = sorted(set(with_scores) | set(no_scores))

    rows = []
    for m in metric_names:
        rows.append({"metric": m, "mode": "with-GT", "score": with_scores.get(m)})
        rows.append({"metric": m, "mode": "no-GT", "score": no_scores.get(m)})
    return pd.DataFrame(rows)


def show_eval_charts(evaluation: dict) -> None:
    """Side-by-side bar chart of ragas / embedding-proxy scores per GT mode."""
    tabs = st.tabs(["ragas (LLM judge)", "embedding-proxy (no LLM)"])

    for tab, source in zip(tabs, ["ragas", "embedding_proxy"], strict=True):
        with tab:
            df = evaluation_comparison_dataframe(evaluation, source)
            if df.empty:
                st.warning(
                    f"No `{source}` scores available — the evaluator may have errored. "
                    "Check the demo transcript for details."
                )
                continue

            # The bar chart — categorical x (metric), grouped by GT mode.
            chart_df = df.copy()
            chart_df["score_display"] = chart_df["score"].fillna(0.0)
            chart_df["available"] = chart_df["score"].notna()
            fig = px.bar(
                chart_df,
                x="metric",
                y="score_display",
                color="mode",
                barmode="group",
                text=chart_df["score"].apply(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"
                ),
                color_discrete_map={"with-GT": "#1f77b4", "no-GT": "#ff7f0e"},
                title=f"{source} — per-metric score (with-GT vs no-GT)",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_title="score (0..1)",
                yaxis=dict(range=[0, 1.05]),
                xaxis_tickangle=-25,
                margin=dict(t=60, b=80),
                legend_title=None,
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Companion table — surfaces NaN ("metric not computable here").
            pivot = df.pivot(index="metric", columns="mode", values="score")
            pivot.columns.name = None
            pivot = pivot.reindex(columns=["with-GT", "no-GT"])
            pivot = pivot.map(
                lambda v: "N/A" if pd.isna(v) else f"{float(v):.3f}"
            )
            st.dataframe(pivot, use_container_width=True)


# ---------------------------------------------------- DSPy helpers
def show_dspy_panel(panel: dict) -> None:
    if "error" in panel and "trial_scores" not in panel:
        st.error(panel["error"])
        return
    cols = st.columns(4)
    cols[0].metric("Optimizer", panel.get("optimizer", "MIPROv2"))
    cols[1].metric("Trainset", panel.get("trainset_size", 0))
    cols[2].metric("GT mode", panel.get("gt_mode", "?"))
    best = max(panel.get("best_score_progression", [0.0]) or [0.0])
    cols[3].metric("Best score", f"{best:.2f}")

    # Trial-by-trial chart
    trial_scores = panel.get("trial_scores") or []
    if trial_scores:
        st.write("### Trial-by-trial scores")
        bp = panel.get("best_score_progression") or []
        # Compute running best over trial_scores for display alignment.
        running_best = []
        rb = -float("inf")
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
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=line_df["trial"], y=line_df["score"],
                name="trial score", marker_color="#1f77b4",
                text=[f"{s:.2f}" for s in line_df["score"]],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_df["trial"], y=line_df["running best"],
                mode="lines+markers", name="running best",
                line=dict(color="#d62728", width=2),
            )
        )
        fig.update_layout(
            xaxis_title="trial", yaxis_title="score",
            margin=dict(t=40, b=40),
            height=320,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"{len(trial_scores)} trials recorded; best score progression: "
            f"{' -> '.join(f'{b:.2f}' for b in (bp or running_best))}."
        )
    else:
        st.info("No per-trial scores captured. (Older result.json without trial logging.)")

    st.write("### Optimized instructions")
    if not panel.get("instructions"):
        st.info("No instructions returned. (DSPy version may not expose them via named_predictors.)")
    for name, instr in panel.get("instructions", {}).items():
        st.write(f"**Predictor `{name}`**")
        st.code(instr or "(empty)", language="text")

    st.write("### Bootstrapped few-shot demos")
    if not panel.get("demos"):
        st.info("No demos returned.")
    for name, demos in panel.get("demos", {}).items():
        st.write(f"**Predictor `{name}` — {len(demos)} demo(s)**")
        for i, d in enumerate(demos):
            with st.expander(f"demo[{i}]", expanded=(i == 0)):
                st.json(d)


# -------------------------------------------------------------- main page
def main() -> None:
    st.set_page_config(
        page_title="ragdx · GT-aware optimization",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("ragdx GT-aware optimization dashboard")
    st.caption(
        "Side-by-side view of the with-GT and no-GT optimization paths "
        "produced by `examples/demo_optimize_gt_modes.py`."
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
    cols = st.columns(3)
    cols[0].metric("Model", bundle["model"])
    cols[1].metric("Corpus size", bundle["corpus_size"])
    cells = "5 (AutoRAG x 2 + Eval x 2 + DSPy x 2)" if "evaluation" in bundle else "4 (AutoRAG x 2 + DSPy x 2)"
    cols[2].metric("Cells", cells)

    # -- STEP 1: data diagnostics
    st.divider()
    st.subheader("Step 1 — Data diagnostics")
    st.caption(
        "The same 5 questions feed both branches. Only `ground_truth` is "
        "removed in the no-GT branch — that flips `gt_mode` from `with_gt` "
        "to `no_gt` and changes every downstream decision."
    )
    st.table(diagnostics_table(bundle["data_diagnostics"]))

    # -- STEP 2: AutoRAG specs
    st.divider()
    st.subheader("Step 2 — AutoRAG specs")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### with-GT")
        show_autorag_panel(bundle["autorag"]["with_gt"])
    with c2:
        st.markdown("#### no-GT")
        show_autorag_panel(bundle["autorag"]["no_gt"])

    # -- STEP 3: pre-flight filter
    st.divider()
    st.subheader("Step 3 — Metric pre-flight")
    st.caption(
        "We ask both branches for the same full ragas-style suite. The "
        "no-GT branch should reject the reference-required ones, recording "
        "a reason in `skipped`. No silent zero-scores anymore."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### with-GT")
        show_filter_panel(bundle["metric_filter"]["with_gt"])
    with c2:
        st.markdown("#### no-GT")
        show_filter_panel(bundle["metric_filter"]["no_gt"])

    # -- STEP 4: actual evaluation scores (real metric numbers!)
    if "evaluation" in bundle:
        st.divider()
        st.subheader("Step 4 — Actual evaluation scores")
        st.caption(
            "Real numbers from `UnifiedEvaluator`. Two backends: a GLM-as-"
            "judge ragas evaluation, and a dependency-free embedding-proxy. "
            "Metrics dropped by the pre-flight appear as `N/A` in the "
            "no-GT column."
        )
        show_eval_charts(bundle["evaluation"])
    else:
        st.info("This bundle was produced before the evaluation step was added. Re-run the demo to populate scores.")

    # -- STEP 5: DSPy MIPROv2 results
    st.divider()
    st.subheader("Step 5 — DSPy MIPROv2 optimization")
    st.caption(
        "with-GT uses pure token-F1 against `example.answer` (no per-trial "
        "LLM cost for the metric). no-GT uses the built-in "
        "`FaithfulnessJudge` dspy.Signature — every metric call is a real "
        "LLM-as-judge invocation."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### with-GT")
        show_dspy_panel(bundle["dspy"]["with_gt"])
    with c2:
        st.markdown("#### no-GT")
        show_dspy_panel(bundle["dspy"]["no_gt"])

    # -- corpus appendix
    st.divider()
    with st.expander("Source corpus (5 records)", expanded=False):
        for i, row in enumerate(bundle["corpus"]):
            st.markdown(f"**Q{i + 1}.** {row['question']}")
            st.markdown(f"_answer_: {row['answer']}")
            st.markdown(f"_ground_truth_: {row['ground_truth']}")
            st.markdown(f"_contexts_: {row['contexts']}")
            st.markdown("---")


if __name__ == "__main__":
    main()
