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

    run_config = meta.get("run_config") or {}
    if run_config:
        with st.expander("Run config (reproducibility)", expanded=False):
            st.json(run_config)
            st.caption(
                "These are the values of ``ragdx experiment``'s flags at "
                "the time this bundle was produced. To reproduce this run, "
                "set the same `--llm-max-concurrent` / `--llm-max-retries` "
                "/ `--bo-trials` / `--bo-init` / `--n-questions` / `--seed`."
            )

    source = meta.get("source") or {}
    if source:
        with st.expander("Corpus source", expanded=False):
            # PDFs include `pdf_meta.chunk_size` / `chunk_overlap` reflecting
            # the **initial load** chunking (used for question synthesis),
            # NOT the BO-winner chunking — keep them but relabel honestly
            # so users don't mistake them for the optimized config.
            display = json.loads(json.dumps(source))  # deep copy
            pm = display.get("pdf_meta") if isinstance(display, dict) else None
            if isinstance(pm, dict):
                for k in ("chunk_size", "chunk_overlap"):
                    if k in pm:
                        pm[f"initial_load_{k}"] = pm.pop(k)
            st.json(display)
            st.caption(
                "PDF `initial_load_*` fields reflect the chunking used at "
                "load-time for question synthesis. The Bayesian search below "
                "explores alternate chunk_size / chunk_overlap values; see "
                "**Bayesian RAG-config search → best_params** for the winner."
            )
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
    st.caption(
        "**Metric weights** combine into the composite score "
        "(`sum of weight_i * metric_i`) — what BO and DSPy optimize "
        "against. **Constraints** are hard pass/fail thresholds (e.g. "
        "`faithfulness >= 0.85`): a trial that violates one is still "
        "scored, but marked **infeasible** so the planner can filter "
        "unsafe configs."
    )
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


# ---------------------------------------------------------------- record selectors
def _pick_record_indices(
    n_records: int, key_prefix: str, default_visible: int = 3
) -> list[int]:
    """Return the indices of records the user wants to inspect.

    If ``n_records <= default_visible``, all are shown. Otherwise the
    first ``default_visible`` are shown by default and a multiselect
    widget lets the user pick others.
    """
    if n_records <= default_visible:
        return list(range(n_records))
    options = list(range(n_records))
    default = options[:default_visible]
    picks = st.multiselect(
        f"Records to show (default first {default_visible}, "
        f"{n_records} total)",
        options=options,
        default=default,
        format_func=lambda i: f"record {i + 1}",
        key=f"{key_prefix}_records",
    )
    # Streamlit returns an empty list when the user clears the widget —
    # in that case fall back to the default so the panel never goes blank.
    return picks or default


def _render_qa_record(rec: dict, has_gt: bool, label: str = "") -> None:
    """Render one Q+A record (with optional GT and contexts).

    Used by both the BO trial inspector and the DSPy A/B panel.
    """
    q = (rec.get("question") or "").strip()
    gt = (rec.get("ground_truth") or "").strip() if rec.get("ground_truth") else ""
    contexts = rec.get("contexts") or []

    if label:
        st.markdown(f"#### {label}")
    st.markdown(f"**Question.** {q}")
    if has_gt and gt:
        st.markdown(f"**Ground truth.** {gt}")

    # Answer fields differ between BO trials (single ``answer``) and
    # DSPy A/B (``baseline_answer`` + ``optimized_answer``).
    if "answer" in rec:
        st.markdown(f"**Answer.** {rec['answer']}")
    else:
        cols = st.columns(2)
        cols[0].markdown("**Baseline answer**")
        cols[0].write(rec.get("baseline_answer", "") or "_(empty)_")
        cols[1].markdown("**Optimized answer**")
        cols[1].write(rec.get("optimized_answer", "") or "_(empty)_")

    if contexts:
        with st.expander(f"Retrieved contexts ({len(contexts)})", expanded=False):
            for i, c in enumerate(contexts):
                st.markdown(f"- **#{i + 1}.** {c[:500]}{'...' if len(c) > 500 else ''}")


def _render_trial_inspector(mode: str, trials: list) -> None:
    """Per-trial Q+A+GT inspector for the Bayesian search panel."""
    has_records = any(t.get("records") for t in trials)
    if not has_records:
        # Older bundles only stored ``answers_preview``; show that as
        # the degraded fallback.
        return
    st.markdown("##### Per-trial Q+A inspector")
    trial_options = list(range(len(trials)))
    sel = st.selectbox(
        "Trial",
        options=trial_options,
        index=0,
        format_func=lambda i: (
            f"trial {trials[i].get('trial_index', i)}: "
            f"{_trial_param_label(trials[i].get('params') or {})}"
            f"  (comp={trials[i].get('composite_score', float('nan')):.3f})"
        ),
        key=f"bo_{mode}_trial",
    )
    trial = trials[sel]
    recs = trial.get("records") or []
    has_gt = any((r.get("ground_truth") or "").strip() for r in recs)
    picks = _pick_record_indices(len(recs), key_prefix=f"bo_{mode}_t{sel}")
    for i in picks:
        st.divider()
        _render_qa_record(recs[i], has_gt=has_gt, label=f"Record {i + 1}")


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
                st.caption(
                    "**composite** = this trial's composite score (per the "
                    "objective above). **running best** = highest composite "
                    "seen up to this trial — the monotonic 'best so far' "
                    "curve. Flat tail + low composite-vs-running-best "
                    "delta = BO has converged."
                )

            # Per-trial Q+A+GT inspector (selectable trial; first 3 records
            # shown by default, dropdown lets users pick specific records).
            _render_trial_inspector(mode, trials)


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
                # NB: melt's ``value_name`` must not clash with any existing
                # column, so we keep the long-form value column name distinct
                # from the wide-form column names.
                line_df = pd.DataFrame(
                    {
                        "trial": list(range(1, len(trial_scores) + 1)),
                        "per-trial": trial_scores,
                        "running best": running,
                    }
                ).melt("trial", value_name="score", var_name="series")
                fig2 = px.line(
                    line_df, x="trial", y="score", color="series",
                    markers=True, title=f"MIPROv2 trial scores — {mode}",
                )
                fig2.update_layout(height=320, margin=dict(t=60, b=40))
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(
                    "**per-trial** = MIPROv2's inner-loop training-time "
                    "score for that trial (token-F1 with-GT, LLM-judge "
                    "without-GT — NOT the ragas composite). "
                    "**running best** = highest score seen so far, "
                    "non-decreasing convergence curve."
                )

            # MIPROv2-optimized prompt instructions. The "baseline prompt"
            # in our pipeline is just the default DSPy signature (no
            # bespoke prompt, only field types) -- so we show the
            # optimized side only and label that explicitly.
            instructions = payload.get("instructions") or {}
            if instructions:
                st.markdown("##### Optimized prompt(s) (MIPROv2 output)")
                st.caption(
                    "Baseline is the default DSPy signature "
                    "(no hand-written prompt — just field types). "
                    "Below are the instructions MIPROv2 discovered for "
                    "the optimized program."
                )
                for predictor_name, instr in instructions.items():
                    with st.expander(
                        f"Instruction for `{predictor_name}`", expanded=False
                    ):
                        st.write(instr or "_(empty)_")

            # Q+A inspector (records-based: question / GT / baseline_answer /
            # optimized_answer per record, with the same first-3 + dropdown UX
            # used by the BO trial inspector).
            recs = payload.get("records") or []
            if recs:
                st.markdown("##### Per-record before/after inspector")
                has_gt = any((r.get("ground_truth") or "").strip() for r in recs)
                picks = _pick_record_indices(len(recs), key_prefix=f"dspy_{mode}")
                for i in picks:
                    st.divider()
                    _render_qa_record(recs[i], has_gt=has_gt, label=f"Record {i + 1}")
            else:
                # Older bundles only stored truncated sample arrays.
                base_ans = payload.get("baseline_sample_answers") or []
                opt_ans = payload.get("optimized_sample_answers") or []
                if base_ans or opt_ans:
                    with st.expander("Sample answers (legacy bundle)", expanded=False):
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
