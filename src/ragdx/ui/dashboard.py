"""
Streamlit Dashboard for RAG Diagnosis

Main Idea:
This module provides a comprehensive web-based dashboard for visualizing and interacting with RAG diagnosis results, optimization sessions, and performance metrics using Streamlit.

Functionalities:
- Metrics visualization: Interactive charts for evaluation results across layers
- Diagnosis reports: Display hypotheses, confidence scores, and recommendations
- Optimization tracking: Monitor trial progress, parameter spaces, and convergence
- Trace analysis: Explore query traces with latency, cost, and citation data
- Feedback management: View and analyze user feedback events
- Comparison tools: Side-by-side comparison of evaluation results
- Demo mode: Built-in demo data for testing and demonstration

Dashboard sections:
- Overview: High-level metrics and summary
- Diagnosis: Detailed diagnosis report with causal analysis
- Optimization: Trial results, parameter importance, and convergence plots
- Traces: Query-level performance analysis
- Feedback: User feedback and issue tracking
- Comparison: Baseline vs current performance analysis

Usage:
Run the dashboard:

    streamlit run src/ragdx/ui/dashboard.py

Or integrate into your application:

    from ragdx.ui.dashboard import main
    main()

The dashboard automatically loads data from the RunStore and provides interactive exploration capabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ragdx.core.compare import compare_results
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult, OptimizationSession
from ragdx.storage.run_store import RunStore
from ragdx.utils.reporting import summarize_target_spec


DEMO_RESULT = EvaluationResult(
    retrieval={"context_precision": 0.63, "context_recall": 0.57, "context_entities_recall": 0.54, "hit_rate_at_k": 0.64},
    generation={"faithfulness": 0.79, "response_relevancy": 0.82, "noise_sensitivity": 0.31, "context_utilization": 0.61, "hallucination": 0.19},
    e2e={"answer_correctness": 0.68, "citation_accuracy": 0.71, "user_success_rate": 0.69},
    metadata={"dataset": "demo", "tools": ["ragas", "ragchecker"], "dataset_shift": True},
)


def _flatten(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for layer_name, metrics in [("retrieval", result.retrieval), ("generation", result.generation), ("e2e", result.e2e)]:
        for metric, value in metrics.items():
            rows.append({"layer": layer_name, "metric": metric, "score": value})
    return pd.DataFrame(rows)


def _trials_df(session: OptimizationSession) -> pd.DataFrame:
    rows = []
    for t in session.trials:
        row = {
            "trial_id": t.trial_id,
            "experiment": t.experiment_name,
            "tool": t.tool,
            "strategy": t.strategy,
            "status": t.status,
            "utility": t.utility,
            "feasible": t.feasible,
            "feasibility_penalty": t.feasibility_penalty,
            "pareto_front": t.pareto_front,
            "config_path": t.config_path,
            "output_path": t.output_path,
            "log_path": t.log_path,
        }
        row.update({f"metric::{k}": v for k, v in t.objective_scores.items()})
        row.update({f"param::{k}": v for k, v in t.parameters.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _trace_df(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for tr in result.traces:
        rows.append({
            "trace_id": tr.trace_id,
            "question": tr.question,
            "chunks": len(tr.retrieved_chunks),
            "citations": len(tr.citations),
            "latency_ms": tr.latency_ms,
            "cost_usd": tr.cost_usd,
            "answer_length": len((tr.answer or "").split()),
        })
    return pd.DataFrame(rows)


def _feedback_df(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for f in result.feedback_events:
        rows.append({
            "feedback_id": f.feedback_id,
            "query_id": f.query_id,
            "kind": f.kind,
            "severity": f.severity,
            "rating": f.rating,
            "note": f.note,
            "created_at": f.created_at,
        })
    return pd.DataFrame(rows)


def _plan_metric_rows(exp) -> pd.DataFrame:
    params = exp.parameters or {}
    baseline_metrics = params.get("baseline_metrics", {})
    target_specs = params.get("target_specs", {})
    objective_weights = params.get("objective_weights", exp.objectives or {})
    rows = []
    metrics = list(dict.fromkeys(list(baseline_metrics.keys()) + list(target_specs.keys()) + list(objective_weights.keys())))
    for metric in metrics:
        spec = target_specs.get(metric, {})
        rows.append({
            "metric": metric,
            "direction": spec.get("direction", "monitor"),
            "mode": spec.get("mode", "target"),
            "baseline": baseline_metrics.get(metric),
            "target": spec.get("target_value"),
            "delta": spec.get("delta_from_baseline"),
            "weight": objective_weights.get(metric),
            "summary": summarize_target_spec(metric, spec) if spec else metric,
        })
    return pd.DataFrame(rows)


def _constraint_rows(exp) -> pd.DataFrame:
    params = exp.parameters or {}
    bounds = params.get("constraint_bounds", exp.constraints or {})
    rows = []
    for key, value in bounds.items():
        rows.append({"constraint": key, "bound": value})
    return pd.DataFrame(rows)


def load_result(uploaded_file) -> EvaluationResult:
    if uploaded_file is None:
        return DEMO_RESULT
    payload = json.load(uploaded_file)
    return EvaluationResult(**payload)


def _read_file(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return "File not found."
    return path.read_text(encoding="utf-8")


def main() -> None:
    st.set_page_config(page_title="ragdx dashboard", layout="wide")
    st.title("ragdx: RAG evaluation, diagnosis, and optimization")
    st.caption("Inspect normalized metrics, traces, diagnosis, optimization plans, and production feedback. Refresh sessions while CLI optimization runs are in progress.")

    store = RunStore()
    uploaded = st.sidebar.file_uploader("Upload evaluation JSON", type=["json"])
    use_saved = st.sidebar.checkbox("Use latest saved run", value=False)

    if use_saved and store.latest() is not None:
        current_run = store.latest()
        result = current_run.evaluation
        report = current_run.diagnosis
        plan = current_run.optimization_plan
    else:
        result = load_result(uploaded)
        report = RAGDiagnosisEngine().diagnose(result)
        plan = OptimizationPlanner().build_plan(report, result=result)
        current_run = None

    tabs = st.tabs(["Scores", "Diagnosis", "Optimization Plan", "Optimization Sessions", "Traces", "Feedback & Governance", "Compare", "Raw JSON"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval metrics", len(result.retrieval))
        c2.metric("Generation metrics", len(result.generation))
        c3.metric("E2E metrics", len(result.e2e))
        df = _flatten(result)
        fig = px.bar(df, x="metric", y="score", color="layer", barmode="group", title="Metric overview")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("Summary")
        st.write(report.summary)
        c1, c2 = st.columns(2)
        c1.metric("Diagnosis confidence", f"{report.diagnosis_confidence:.2f}")
        c2.metric("Causal signals", len(report.causal_signals))
        if report.evaluator_agreement:
            st.subheader("Evaluator agreement")
            st.dataframe(pd.DataFrame([{"metric": k, "agreement": v} for k, v in report.evaluator_agreement.items()]), use_container_width=True)
        if report.causal_signals:
            st.subheader("Bayesian causal signals")
            causal_df = pd.DataFrame([s.model_dump() for s in report.causal_signals])
            st.dataframe(causal_df, use_container_width=True)
        if plan.rationale:
            st.subheader("Why this plan was selected")
            for item in plan.rationale:
                st.write(f"- {item}")
        if report.metric_gaps:
            st.subheader("Metric gaps")
            gap_df = pd.DataFrame([{"metric": k, "gap": v} for k, v in report.metric_gaps.items()])
            st.dataframe(gap_df, use_container_width=True)
        st.subheader("Hypotheses")
        for idx, hyp in enumerate(report.hypotheses, start=1):
            with st.expander(f"{idx}. {hyp.root_cause}", expanded=(idx == 1)):
                st.write(f"**Component:** {hyp.component}")
                st.write(f"**Severity:** {hyp.severity}")
                st.write(f"**Confidence:** {hyp.confidence:.2f}")
                st.write("**Evidence**")
                for item in hyp.evidence:
                    st.write(f"- {item}")
                st.write("**Recommended actions**")
                for item in hyp.recommended_actions:
                    st.write(f"- {item}")
        if report.disambiguation_actions:
            st.subheader("Disambiguation actions")
            for item in report.disambiguation_actions:
                st.write(f"- {item}")
        st.subheader("Priority actions")
        for item in report.priority_actions:
            st.write(f"- {item}")

    with tabs[2]:
        if not plan.experiments:
            st.info("No optimization experiment proposed at the current thresholds.")
        else:
            st.caption("Objective weights are trade-off coefficients, not target metric values.")
            for exp in plan.experiments:
                with st.expander(f"{exp.stage} | {exp.name} | {exp.tool} | {exp.search_strategy}", expanded=True):
                    st.write(exp.description)
                    st.write(f"**Depends on:** {', '.join(exp.depends_on) if exp.depends_on else 'none'}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Trial budget", exp.max_trials)
                    c2.metric("Primary baseline", exp.baseline_score if exp.baseline_score is not None else float('nan'))
                    c3.metric("Objectives", len(exp.objectives))
                    metric_df = _plan_metric_rows(exp)
                    if not metric_df.empty:
                        st.subheader("Baseline-relative metric plan")
                        st.dataframe(metric_df, use_container_width=True)
                        if metric_df["baseline"].notna().any() and metric_df["target"].notna().any():
                            plot_df = metric_df.dropna(subset=["baseline", "target"]).copy()
                            plot_df = plot_df.melt(id_vars=["metric"], value_vars=["baseline", "target"], var_name="kind", value_name="value")
                            fig = px.bar(plot_df, x="metric", y="value", color="kind", barmode="group", title="Baseline vs target")
                            st.plotly_chart(fig, use_container_width=True)
                    constraint_df = _constraint_rows(exp)
                    if not constraint_df.empty:
                        st.subheader("Constraint bounds")
                        st.dataframe(constraint_df, use_container_width=True)
                    st.subheader("Search space")
                    st.json(exp.search_space)
                    if exp.notes:
                        st.write(f"**Notes:** {exp.notes}")

    with tabs[3]:
        _ = st.button("Refresh sessions")
        sessions = store.list_sessions()
        if not sessions:
            st.info("No optimization sessions found. Run `ragdx optimize ...` first.")
        else:
            options = {f"{s.session_id} | {s.status} | {s.strategy}": s for s in sessions}
            selected = st.selectbox("Optimization session", list(options.keys()))
            session = options[selected]
            progress = session.completed_trials / max(session.total_trials, 1)
            st.progress(progress)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Status", session.status)
            c2.metric("Strategy", session.strategy)
            c3.metric("Trials", f"{session.completed_trials}/{session.total_trials}")
            c4.metric("Pareto front", len(session.pareto_front_ids))
            c5.metric("Feasible HV", session.feasible_hypervolume)
            if session.best_trial_id:
                st.write(f"**Best trial by scalar utility:** `{session.best_trial_id}`")

            trials_df = _trials_df(session)
            if not trials_df.empty:
                st.dataframe(trials_df, use_container_width=True)
                metric_cols = [c for c in trials_df.columns if c.startswith("metric::")]
                if metric_cols:
                    metric_choice = st.selectbox("Metric to plot", metric_cols)
                    fig = px.line(trials_df.reset_index(), x="index", y=metric_choice, color="experiment", markers=True, title="Objective progression")
                    st.plotly_chart(fig, use_container_width=True)
                if "metric::answer_correctness" in trials_df.columns and "metric::citation_accuracy" in trials_df.columns:
                    fig2 = px.scatter(
                        trials_df,
                        x="metric::answer_correctness",
                        y="metric::citation_accuracy",
                        color="pareto_front",
                        symbol="experiment",
                        hover_data=["trial_id", "utility"],
                        title="Pareto view: answer_correctness vs citation_accuracy",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                selected_trial = st.selectbox("Inspect trial", trials_df["trial_id"].tolist())
                row = trials_df[trials_df["trial_id"] == selected_trial].iloc[0]
                if row.get("config_path"):
                    st.subheader("Config")
                    st.code(_read_file(row["config_path"]), language="yaml")
                if row.get("log_path"):
                    st.subheader("Runner log")
                    st.code(_read_file(row["log_path"]), language="text")
                if row.get("output_path"):
                    st.subheader("Runner output")
                    st.code(_read_file(row["output_path"]), language="json")

    with tabs[4]:
        trace_df = _trace_df(result)
        if trace_df.empty:
            st.info("No traces attached to this evaluation result.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Traces", len(trace_df))
            c2.metric("Avg latency ms", round(trace_df["latency_ms"].dropna().mean(), 2) if trace_df["latency_ms"].notna().any() else 0.0)
            c3.metric("Avg chunks", round(trace_df["chunks"].mean(), 2))
            st.dataframe(trace_df, use_container_width=True)
            if trace_df["latency_ms"].notna().any():
                st.plotly_chart(px.scatter(trace_df, x="chunks", y="latency_ms", hover_data=["trace_id", "question"], title="Trace latency vs retrieved chunks"), use_container_width=True)

    with tabs[5]:
        feedback_df = _feedback_df(result)
        global_feedback = store.feedback_summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("Attached feedback events", len(feedback_df))
        c2.metric("Global feedback total", int(global_feedback.get("total_feedback", 0.0)))
        c3.metric("Global negative rate", global_feedback.get("negative_feedback_rate", 0.0))
        if not feedback_df.empty:
            st.subheader("Attached production feedback")
            st.dataframe(feedback_df, use_container_width=True)
            st.plotly_chart(px.histogram(feedback_df, x="kind", color="severity", title="Feedback distribution"), use_container_width=True)
        st.subheader("Governance view")
        gov_rows = []
        for exp in plan.experiments:
            gov_rows.append({
                "stage": exp.stage,
                "experiment": exp.name,
                "constraints": json.dumps(exp.constraints),
                "depends_on": ", ".join(exp.depends_on),
            })
        st.dataframe(pd.DataFrame(gov_rows), use_container_width=True)

    with tabs[6]:
        saved_runs = store.list_runs()
        if len(saved_runs) < 2 and current_run is None:
            st.info("Save two runs first, or use the CLI compare command for ad hoc comparison.")
        else:
            options = {f"{r.run_id} | {r.name}": r for r in saved_runs}
            selected_current = st.selectbox("Current run", list(options.keys())) if options else None
            selected_base = st.selectbox("Baseline run", list(options.keys()), index=1 if len(options) > 1 else 0) if options else None
            if selected_current and selected_base:
                current = options[selected_current]
                baseline = options[selected_base]
                cmp_df = pd.DataFrame([c.model_dump() for c in compare_results(current.evaluation, baseline.evaluation)])
                st.dataframe(cmp_df, use_container_width=True)

    with tabs[7]:
        st.code(result.model_dump_json(indent=2), language="json")


if __name__ == "__main__":
    main()
