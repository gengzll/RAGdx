"""Generic Streamlit dashboard for ``ragdx experiment`` bundles.

This dashboard renders any ``schema_version: 1`` bundle produced by
:func:`ragdx.experiments.run_experiment`. It introspects the bundle's
``meta.modes_run`` and conditionally renders sections for each mode
present — so the same dashboard handles single-mode (no-GT PDF /
synthesized) runs, with-GT only, and side-by-side ``both`` runs without
any per-experiment customisation.

Legacy bundles (from the older ``demo_optimize_gt_modes.py`` /
``demo_pdf_no_gt.py`` outputs) are auto-upgraded via
:func:`ragdx.experiments.migrate_legacy_bundle`.

Run::

    ragdx experiment-dashboard --bundle .ragdx_experiment/result.json
    # or
    streamlit run src/ragdx/ui/experiment_dashboard.py -- \\
        --bundle path/to/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from ragdx.experiments import SCHEMA_VERSION, migrate_legacy_bundle

REPO = Path(__file__).resolve().parents[3]
DEFAULT_LIVE = REPO / ".ragdx_experiment" / "result.json"
DEFAULT_COMMITTED = REPO / "docs" / "examples" / "pdf_no_gt_result.json"


# ---------------------------------------------------------------- args
def _resolve_bundle_path() -> Path:
    """Pick the bundle path from ``--bundle``, env var, or first existing
    fallback (live run dir → committed snapshot)."""
    # streamlit forwards extra args after ``--`` in argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bundle", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.bundle:
        return Path(args.bundle)
    env = os.environ.get("RAGDX_EXPERIMENT_BUNDLE")
    if env:
        return Path(env)
    if DEFAULT_LIVE.exists():
        return DEFAULT_LIVE
    return DEFAULT_COMMITTED


# ---------------------------------------------------------------- load
def load_bundle(path: Path) -> dict:
    if not path.exists():
        st.error(
            f"Bundle not found: `{path}`.\n\n"
            "Run an experiment first:\n\n"
            "```bash\n"
            "ragdx experiment <corpus> --has-gt --mode both\n"
            "```\n\n"
            "Or point the dashboard at an existing bundle:\n\n"
            "```bash\n"
            "ragdx experiment-dashboard --bundle path/to/result.json\n"
            "```"
        )
        st.stop()
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != SCHEMA_VERSION:
        st.info(
            f"Bundle uses schema_version="
            f"{bundle.get('schema_version', 'missing')}; auto-upgrading to "
            f"v{SCHEMA_VERSION}."
        )
        bundle = migrate_legacy_bundle(bundle)
        if bundle.get("schema_version") == 0:
            st.warning(
                "Bundle shape was not recognised by the migration helper. "
                "Some sections may render blank — please re-run the experiment "
                "with the current ragdx version."
            )
    return bundle


# ---------------------------------------------------------------- meta
def render_header(bundle: dict) -> None:
    meta = bundle.get("meta") or {}
    st.title("ragdx experiment dashboard")
    st.caption(
        f"schema v{bundle.get('schema_version', '?')} · "
        f"model: `{meta.get('model', '?')}` · "
        f"endpoint: `{meta.get('model_endpoint', '?')}`"
    )

    cols = st.columns(4)
    cols[0].metric("Experiment mode", meta.get("experiment_mode", "?"))
    cols[1].metric("Modes run", ", ".join(meta.get("modes_run") or []) or "—")
    cols[2].metric("Has GT", "Yes" if meta.get("has_gt") else "No")
    cols[3].metric("Detected GT", meta.get("detected_gt_mode", "—"))

    source = meta.get("source") or {}
    if source:
        with st.expander("Corpus source", expanded=False):
            st.json(source)
    if meta.get("_migrated_from"):
        st.caption(
            f"_migrated from legacy bundle ({meta['_migrated_from']})_"
        )


# ---------------------------------------------------------------- diagnostics
def render_diagnostics(bundle: dict) -> None:
    diag = bundle.get("data_diagnostics") or {}
    if not diag:
        return
    st.subheader("Data diagnostics")
    rows = []
    for m, d in diag.items():
        rows.append({"mode": m, **d})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- questions
def render_questions(bundle: dict) -> None:
    qs = bundle.get("questions") or []
    if not qs:
        return
    with st.expander(f"Questions ({len(qs)})", expanded=False):
        df = pd.DataFrame(qs)
        # Show ground_truth column even if all-null so users see the field.
        if "ground_truth" not in df.columns:
            df["ground_truth"] = None
        st.dataframe(df, use_container_width=True, hide_index=True)
    extras = bundle.get("extras") or {}
    if extras.get("synthesized_questions"):
        with st.expander("Synthesized question provenance", expanded=False):
            st.dataframe(
                pd.DataFrame(extras["synthesized_questions"]),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------- objectives
def render_objectives(bundle: dict) -> None:
    objs = bundle.get("objectives") or {}
    if not objs:
        return
    st.subheader("Composite objectives")
    for m, spec in objs.items():
        with st.expander(f"`{m}` objective", expanded=False):
            metrics = (spec or {}).get("metrics") or {}
            constraints = (spec or {}).get("constraints") or {}
            cols = st.columns(2)
            cols[0].write("**Metric weights**")
            cols[0].dataframe(
                pd.DataFrame(
                    [{"metric": k, "weight": v} for k, v in metrics.items()]
                ),
                use_container_width=True,
                hide_index=True,
            )
            cols[1].write("**Constraints**")
            if constraints:
                cols[1].dataframe(
                    pd.DataFrame(
                        [
                            {
                                "metric": k,
                                "direction": v[0] if isinstance(v, (list, tuple)) else "—",
                                "bound": v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else v,
                            }
                            for k, v in constraints.items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                cols[1].caption("_no hard constraints_")


# ---------------------------------------------------------------- bayes search
def _trial_param_label(params: dict) -> str:
    if not params:
        return "—"
    return " · ".join(f"{k}={v}" for k, v in params.items())


def render_bayes_search(bundle: dict) -> None:
    bs = bundle.get("bayes_search") or {}
    if not bs:
        return
    st.subheader("Bayesian RAG-config search")
    tabs = st.tabs(list(bs.keys()))
    for tab, (mode, payload) in zip(tabs, bs.items(), strict=False):
        with tab:
            payload = payload or {}
            kind = payload.get("_legacy_kind")
            if kind == "grid":
                st.caption("_legacy bundle: grid search (only top_k varied)_")
            trials = payload.get("trials") or []
            best_params = payload.get("best_params") or {}
            best_comp = payload.get("best_composite")
            search_space = payload.get("search_space") or {}

            cols = st.columns(3)
            cols[0].metric("Trials", len(trials))
            cols[1].metric(
                "Best composite",
                f"{best_comp:.3f}" if isinstance(best_comp, (int, float)) else "—",
            )
            cols[2].metric("Best params", _trial_param_label(best_params))

            if search_space:
                with st.expander("Search space", expanded=False):
                    st.json(search_space)

            if not trials:
                st.info("No trials recorded.")
                continue

            # Trial table (long form so any param schema works)
            tbl_rows: list[dict[str, Any]] = []
            for i, t in enumerate(trials):
                row = {
                    "#": t.get("trial_index", i),
                    "params": _trial_param_label(t.get("params") or {}),
                    "composite": t.get("composite_score"),
                    "feasible": t.get("feasible", True),
                }
                for k, v in (t.get("scores") or {}).items():
                    row[k] = v
                tbl_rows.append(row)
            df = pd.DataFrame(tbl_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Trial-progression line (composite per trial + running best)
            comp_series = [t.get("composite_score") for t in trials]
            if any(isinstance(v, (int, float)) for v in comp_series):
                running_best: list[float] = []
                rb = float("-inf")
                for v in comp_series:
                    if isinstance(v, (int, float)):
                        rb = max(rb, v)
                    running_best.append(rb if rb != float("-inf") else None)  # type: ignore[arg-type]
                progress = pd.DataFrame(
                    {
                        "trial": list(range(1, len(trials) + 1)),
                        "composite": comp_series,
                        "running best": running_best,
                    }
                ).melt("trial", value_name="score", var_name="series")
                fig = px.line(
                    progress, x="trial", y="score", color="series",
                    markers=True, title=f"BO progression — {mode}",
                )
                fig.update_layout(height=360, margin=dict(t=60, b=40))
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- dspy a/b
def render_dspy_a_b(bundle: dict) -> None:
    ab = bundle.get("dspy_a_b") or {}
    if not ab:
        return
    st.subheader("DSPy before/after (at BO winner config)")
    tabs = st.tabs(list(ab.keys()))
    for tab, (mode, payload) in zip(tabs, ab.items(), strict=False):
        with tab:
            payload = payload or {}
            bs = payload.get("baseline_scores") or {}
            os_ = payload.get("optimized_scores") or {}
            delta = payload.get("delta") or {}
            comp = payload.get("composite") or {}

            if not bs and not os_:
                st.info("No before/after scores in this bundle.")
                continue

            metrics = sorted(set(bs) | set(os_))
            rows = []
            for m in metrics:
                rows.append({"metric": m, "phase": "baseline", "score": bs.get(m)})
                rows.append({"metric": m, "phase": "optimized", "score": os_.get(m)})
            df = pd.DataFrame(rows)
            df["score_display"] = df["score"].fillna(0.0)
            fig = px.bar(
                df, x="metric", y="score_display", color="phase",
                barmode="group",
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
            st.plotly_chart(fig, use_container_width=True)

            if comp:
                b = (comp.get("baseline") or {}).get("score")
                o = (comp.get("optimized") or {}).get("score")
                d = comp.get("delta")
                cols = st.columns(3)
                cols[0].metric(
                    "Composite baseline",
                    f"{b:.3f}" if isinstance(b, (int, float)) else "—",
                )
                cols[1].metric(
                    "Composite optimized",
                    f"{o:.3f}" if isinstance(o, (int, float)) else "—",
                    delta=f"{d:+.3f}" if isinstance(d, (int, float)) else None,
                )
                weights = (comp.get("objective_spec") or {}).get("metrics", {})
                cols[2].metric(
                    "Weights",
                    ", ".join(f"{k}={v}" for k, v in weights.items()) or "—",
                )

            # Delta table
            rows_d = []
            for m in metrics:
                d = delta.get(m)
                arrow = "→"
                if isinstance(d, (int, float)):
                    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
                rows_d.append({
                    "metric": m,
                    "baseline": f"{bs.get(m):.3f}" if isinstance(bs.get(m), (int, float)) else "N/A",
                    "optimized": f"{os_.get(m):.3f}" if isinstance(os_.get(m), (int, float)) else "N/A",
                    "delta": f"{arrow} {d:+.3f}" if isinstance(d, (int, float)) else "N/A",
                })
            st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

            # MIPROv2 trial-by-trial progression (training metric).
            trial_scores = payload.get("trial_scores") or []
            if trial_scores:
                rb = float("-inf")
                running = []
                for s in trial_scores:
                    rb = max(rb, s)
                    running.append(rb)
                line_df = pd.DataFrame(
                    {
                        "trial": list(range(1, len(trial_scores) + 1)),
                        "score": trial_scores,
                        "running best": running,
                    }
                ).melt("trial", value_name="score", var_name="series")
                fig2 = px.line(
                    line_df, x="trial", y="score", color="series",
                    markers=True, title=f"MIPROv2 trial scores — {mode}",
                )
                fig2.update_layout(height=320, margin=dict(t=60, b=40))
                st.plotly_chart(fig2, use_container_width=True)

            # Sample answers (collapsed)
            base_ans = payload.get("baseline_sample_answers") or []
            opt_ans = payload.get("optimized_sample_answers") or []
            if base_ans or opt_ans:
                with st.expander("Sample answers (first 5)", expanded=False):
                    rows_a = []
                    for i, (b, o) in enumerate(zip(base_ans[:5], opt_ans[:5], strict=False)):
                        rows_a.append({"#": i + 1, "baseline": (b or "")[:400], "optimized": (o or "")[:400]})
                    st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- extras
def render_extras(bundle: dict) -> None:
    extras = bundle.get("extras") or {}
    leftover = {k: v for k, v in extras.items() if k not in {"pdf_meta", "synthesized_questions"}}
    if "pdf_meta" in extras:
        with st.expander("PDF metadata", expanded=False):
            st.json(extras["pdf_meta"])
    if leftover:
        with st.expander("Extras", expanded=False):
            st.json(leftover)


# ---------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(
        page_title="ragdx experiment dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    path = _resolve_bundle_path()
    with st.sidebar:
        st.write("**Bundle**")
        st.code(str(path))
        st.caption(
            "Override with `--bundle path/to/result.json` or the "
            "`RAGDX_EXPERIMENT_BUNDLE` env var."
        )

    bundle = load_bundle(path)
    render_header(bundle)
    render_diagnostics(bundle)
    render_questions(bundle)
    render_objectives(bundle)
    render_bayes_search(bundle)
    render_dspy_a_b(bundle)
    render_extras(bundle)


if __name__ == "__main__":
    main()
