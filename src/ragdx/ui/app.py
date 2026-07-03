"""ragdx studio — upload → configure → run → report Streamlit app.

Layout:

* Sidebar — connection only (model / API base / key).
* 1 · Upload documents & questions — PDFs (pooled), optional Excel/CSV
  ground truth, and a **Questions** subsection (GT column mapping, or in
  no-GT mode: synthesize N questions / type your own). Confirm to lock in.
* 2 · Experiment settings — three groups: RAG optimization parameters,
  prompt optimization parameters, and the baseline system prompt (with
  its default shown). Confirm to lock in.
* 3 · Run — enabled once 1 and 2 are confirmed; live progress + report.

Launch it with ``ragdx ui`` (or ``ragdx-ui`` / ``streamlit run
src/ragdx/ui/app.py``).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ordered milestones shown as a per-stage checklist during a run. Each
# entry is (event-stage-name-that-marks-it-complete, human label).
_STAGE_CHECKLIST: list[tuple[str, str]] = [
    ("corpus_loaded", "Load & chunk documents"),
    ("bo_done", "Bayesian config search"),
    ("dspy_done", "Prompt optimization (DSPy)"),
    ("bundle_written", "Evaluate & build report"),
]

# Persistent per-experiment storage. Each named run lives in
# ``.ragdx_ui_runs/<name>/`` with its uploaded inputs, the engine's
# output dir, and a small meta.json — enough to re-view the report
# later and to resume an interrupted run from its checkpoints.
_RUNS_ROOT = Path(".ragdx_ui_runs")

# Process-wide registry of the single in-flight run. ``st.session_state``
# is per-browser-session: a page refresh resets it to idle while the
# engine daemon thread keeps running — and a user who then clicks
# Resume again stacks a SECOND engine thread onto the same endpoint
# (observed in production: 240 threads fighting over one rate-limited
# endpoint, every ragas job timing out). New sessions adopt the run
# recorded here instead of believing nothing is running.
#
# CRITICAL: streamlit re-executes this script (fresh module namespace)
# on every rerun, so a plain module global here would reset each time.
# Anchor the dict on the ``ragdx.ui`` package, which lives in
# ``sys.modules`` and survives for the whole server process.
import ragdx.ui as _ragdx_ui_pkg  # noqa: E402 — must follow the registry docstring block

if not hasattr(_ragdx_ui_pkg, "_ACTIVE_RUN_REGISTRY"):
    _ragdx_ui_pkg._ACTIVE_RUN_REGISTRY = {}
_ACTIVE_RUN: dict[str, Any] = _ragdx_ui_pkg._ACTIVE_RUN_REGISTRY


def _active_run_alive() -> bool:
    t = _ACTIVE_RUN.get("thread")
    return t is not None and t.is_alive()


def _adopt_active_run(ss) -> None:
    """Re-attach a fresh browser session to the in-flight run, sharing
    the same queue/event/log objects the worker writes into."""
    ss.phase = "running"
    ss.run_name = _ACTIVE_RUN["name"]
    ss.queue = _ACTIVE_RUN["queue"]
    ss.events = _ACTIVE_RUN["events"]
    ss.logs = _ACTIVE_RUN["logs"]
    ss.output_dir = _ACTIVE_RUN["output_dir"]
    ss.thread = _ACTIVE_RUN["thread"]
    ss._checklist_state = None


def _slugify(name: str) -> str:
    """Filesystem- and checkpoint-group-safe experiment name."""
    slug = re.sub(r"[^\w\-.]+", "-", name.strip()).strip("-")
    return slug[:80] or "exp"


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "meta.json"


def _save_run_meta(run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _run_meta_path(run_dir).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_run_meta(run_dir: Path) -> dict | None:
    p = _run_meta_path(run_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_runs() -> list[dict]:
    """All saved experiments, newest first."""
    if not _RUNS_ROOT.exists():
        return []
    metas = []
    for d in _RUNS_ROOT.iterdir():
        if d.is_dir():
            m = _load_run_meta(d)
            if m and m.get("name"):
                metas.append(m)
    return sorted(metas, key=lambda m: m.get("created_at", ""), reverse=True)


# =====================================================================
# Console-script entry point (``ragdx-ui`` / ``ragdx ui``)
# =====================================================================
def main() -> None:
    """Launch this file under ``streamlit run``.

    The console script imports this module (so the app body guarded by
    ``__name__ == "__main__"`` does *not* execute); we then hand the
    file to streamlit, which re-executes it as the main script.
    """
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
    cmd += sys.argv[1:]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


# =====================================================================
# App body (executed by ``streamlit run``)
# =====================================================================
def _run_app() -> None:
    import streamlit as st

    from ragdx.experiments import DEFAULT_SYSTEM_INSTRUCTION, run_experiment
    from ragdx.loaders.tabular import (
        detect_mapping,
        load_gt_table,
        records_from_table,
        write_questions_jsonl,
    )
    from ragdx.optim.objectives import default_objective
    from ragdx.ui.experiment_report import render_report

    st.set_page_config(page_title="ragdx studio", page_icon="🔬", layout="wide")
    st.title("🔬 ragdx studio")
    st.caption(
        "Upload documents, configure the experiment, and run the end-to-end "
        "RAG optimization — then view and download the report."
    )

    ss = st.session_state
    ss.setdefault("phase", "idle")  # idle | running | done | error
    ss.setdefault("events", [])
    ss.setdefault("logs", [])
    ss.setdefault("bundle", None)
    ss.setdefault("output_dir", None)
    ss.setdefault("error", None)
    ss.setdefault("confirm_upload", False)
    ss.setdefault("confirm_settings", False)

    # A page refresh resets session_state but not the engine thread —
    # re-attach to the in-flight run so this session shows its progress
    # (and the Run/Resume buttons stay correctly disabled).
    if ss.phase != "running" and _active_run_alive():
        _adopt_active_run(ss)

    # ------------------------------------------------- sidebar: connection
    with st.sidebar:
        st.header("Connection")
        st.caption("LLM endpoint used for generation, judging, and prompt tuning.")
        model = st.text_input(
            "Model", value="openai/glm-4-flash",
            help="LiteLLM id at the endpoint below. Faster Zhipu options: "
            "openai/glm-4-flashx, openai/glm-4-airx, openai/glm-4.5-flash. "
            "Any OpenAI-compatible endpoint works (e.g. gpt-4o-mini).",
        )
        api_base = st.text_input("API base URL", value="https://open.bigmodel.cn/api/paas/v4")
        api_key = st.text_input(
            "API key",
            value=os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            type="password",
            help="Falls back to ZHIPU_API_KEY / OPENAI_API_KEY if left blank.",
        )
        llm_concurrency = st.number_input(
            "Max concurrent LLM calls", 1, 16, 2,
            help="Biggest speed lever after the model itself. Default 2 "
            "suits strict rate-limited endpoints (free-tier GLM-4-Flash); "
            "raise to 8-16 on paid endpoints (OpenAI / Anthropic / paid "
            "GLM) for a near-linear evaluation speedup.",
        )

        st.header("Experiments")
        saved_runs = _list_runs()
        _NEW = "🆕 New experiment"
        selected_run = st.selectbox(
            "View / resume",
            [_NEW, *[m["name"] for m in saved_runs]],
            index=0,
            help="Pick a saved experiment to view its configuration and "
            "report, or to resume it if it was interrupted.",
        )

    running = ss.phase == "running"

    # =============================================================
    # Viewer mode: a saved experiment is selected in the sidebar.
    # =============================================================
    if selected_run != _NEW:
        _render_run_viewer(
            ss=ss,
            st=st,
            name=selected_run,
            render_report=render_report,
            run_experiment=run_experiment,
            api_key=api_key or None,
        )
        return

    # =============================================================
    # 1 · Upload documents & questions
    # =============================================================
    st.subheader("1 · Upload documents & questions")
    col_pdf, col_gt = st.columns(2)
    with col_pdf:
        pdf_files = st.file_uploader(
            "Documents (PDF, required)",
            type=["pdf"],
            disabled=running,
            accept_multiple_files=True,
            help="Upload one or more PDFs. Multiple files are pooled into a "
            "single corpus for the experiment.",
        )
        if pdf_files:
            st.caption(f"{len(pdf_files)} PDF(s): " + ", ".join(f.name for f in pdf_files))
    with col_gt:
        gt_file = st.file_uploader(
            "Ground truth (Excel/CSV, optional)",
            type=["csv", "tsv", "xlsx", "xls"],
            disabled=running,
            help="Provide to run in with-GT mode. Without it, the run uses "
            "no-GT mode (synthesized or hand-entered questions).",
        )

    st.markdown("#### Questions")
    gt_records = None
    gt_ready = True
    custom_questions: list[str] | None = None
    n_questions = 5  # only used when synthesizing in no-GT mode

    if gt_file is not None:
        st.caption("With-GT mode — map your columns to `question` / `ground_truth` / `contexts`.")
        try:
            tmp_gt = Path(tempfile.gettempdir()) / f"ragdx_gt_{gt_file.name}"
            tmp_gt.write_bytes(gt_file.getvalue())
            df = load_gt_table(tmp_gt)
        except Exception as exc:
            st.error(f"Could not read ground-truth file: {exc}")
            df = None

        if df is not None:
            st.dataframe(df.head(10), use_container_width=True)
            auto = detect_mapping(df)
            cols = list(df.columns)
            none_label = "— none —"

            def _select(field: str, optional: bool) -> str | None:
                options = ([none_label, *cols]) if optional else cols
                default = auto.get(field)
                idx = options.index(default) if default in options else 0
                choice = st.selectbox(
                    f"`{field}` column" + (" (optional)" if optional else ""),
                    options,
                    index=idx,
                    disabled=running,
                    key=f"map_{field}",
                )
                return None if choice == none_label else choice

            m1, m2, m3 = st.columns(3)
            with m1:
                q_col = _select("question", optional=False)
            with m2:
                gt_col = _select("ground_truth", optional=False)
            with m3:
                ctx_col = _select("contexts", optional=True)
            mapping = {"question": q_col, "ground_truth": gt_col, "contexts": ctx_col}

            try:
                gt_records = records_from_table(df, mapping)
                st.success(f"{len(gt_records)} ground-truth question(s) ready (with-GT mode).")
            except Exception as exc:
                gt_ready = False
                st.warning(f"Finish the mapping to enable the run: {exc}")
    else:
        st.caption("No-GT mode — no reference answers.")
        source = st.radio(
            "Where should the evaluation questions come from?",
            ["Synthesize from the documents", "Enter my own questions"],
            disabled=running,
            horizontal=True,
        )
        if source == "Enter my own questions":
            raw = st.text_area(
                "Questions — one per line",
                height=150,
                disabled=running,
                placeholder="What is the company's net-zero target?\nWhat certifications are mentioned?",
                help="These exact questions are used (no synthesis). "
                "No reference answers are needed in no-GT mode.",
            )
            custom_questions = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if custom_questions:
                st.success(f"{len(custom_questions)} question(s) entered.")
            else:
                gt_ready = False
                st.warning("Enter at least one question, or switch to synthesize.")
        else:
            n_questions = st.number_input(
                "Questions to synthesize",
                1, 100, 5,
                disabled=running,
                help="How many questions to synthesize from the documents.",
            )
            st.info("Questions will be synthesized from the uploaded documents.")

    if st.button("✓ Confirm documents & questions", disabled=running):
        if not pdf_files:
            st.warning("Upload at least one PDF first.")
        elif not gt_ready:
            st.warning("Finish the questions setup first.")
        else:
            ss.confirm_upload = True
    if ss.confirm_upload:
        st.success("Documents & questions confirmed.")

    # =============================================================
    # 2 · Experiment settings
    # =============================================================
    st.subheader("2 · Experiment settings")

    scope_label = st.radio(
        "Experiment scope",
        ["Full (RAG + prompt)", "RAG optimization only", "Prompt optimization only"],
        disabled=running,
        horizontal=True,
        help="Full runs the Bayesian RAG-config search, then prompt "
        "optimization at the winner. RAG-only stops after the config "
        "search. Prompt-only skips the search — you fix the RAG config "
        "and only the prompt is optimized.",
    )
    stages = {
        "Full (RAG + prompt)": "all",
        "RAG optimization only": "rag",
        "Prompt optimization only": "prompt",
    }[scope_label]

    seed = st.number_input(
        "Seed", 0, 10_000, 7, disabled=running,
        help="Deterministic seed for the search and question synthesis.",
    )

    # Defaults that apply when a sub-section is hidden by the scope.
    n_bo_trials, n_bo_init = 8, 3
    dspy_optimizer, mipro_auto = "gepa", "light"
    search_space_ok = True

    if stages in ("all", "rag"):
        st.markdown("#### RAG optimization parameters")
        st.caption("Bayesian search over the RAG config (chunk size / overlap / top-k).")
        c1, c2 = st.columns(2)
        with c1:
            n_bo_trials = st.number_input(
                "Bayesian search trials", 2, 64, 8, disabled=running,
                help="Total RAG configurations the optimizer evaluates. Each "
                "trial builds and scores a candidate (chunk size, overlap, "
                "top-k). More = more thorough, more LLM calls.",
            )
        with c2:
            n_bo_init = st.number_input(
                "Bayesian init rounds", 1, 16, 3, disabled=running,
                help="Random configurations sampled before the Bayesian model "
                "starts steering. Seeds the surrogate; keep <= trials (2-3).",
            )

        st.markdown(
            "**Search space** — the candidate values the Bayesian optimizer "
            "explores. Comma-separated; enter any values within the ranges."
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            cs_raw = st.text_input(
                "Chunk sizes (chars, 64-8192)",
                value="256, 512, 1024",
                disabled=running,
                help="Candidate chunk sizes for splitting the documents. "
                "Smaller = more precise retrieval, less context per chunk.",
            )
            chunk_sizes, cs_err = _parse_int_list(cs_raw, 64, 8192)
        with s2:
            co_raw = st.text_input(
                "Chunk overlaps (chars, 0-1024)",
                value="0, 50, 100",
                disabled=running,
                help="Candidate overlap between adjacent chunks. Overlap "
                "avoids cutting facts in half at chunk boundaries. Must stay "
                "smaller than the smallest chunk size.",
            )
            chunk_overlaps, co_err = _parse_int_list(co_raw, 0, 1024)
        with s3:
            tk_raw = st.text_input(
                "Top-k values (1-50)",
                value="1, 3, 5, 7",
                disabled=running,
                help="Candidate numbers of chunks retrieved per question.",
            )
            top_ks, tk_err = _parse_int_list(tk_raw, 1, 50)

        ss_errors = [
            f"{label}: {err}"
            for label, err in (
                ("Chunk sizes", cs_err), ("Chunk overlaps", co_err), ("Top-k", tk_err),
            )
            if err
        ]
        # Cross-axis constraint: the text splitter requires overlap < chunk
        # size for every (size, overlap) combination the search may try.
        if not ss_errors and chunk_sizes and chunk_overlaps and max(chunk_overlaps) >= min(chunk_sizes):
            ss_errors.append(
                f"largest overlap ({max(chunk_overlaps)}) must be smaller than "
                f"the smallest chunk size ({min(chunk_sizes)})"
            )
        search_space_ok = not ss_errors
        if ss_errors:
            st.warning("Fix the search space: " + "; ".join(ss_errors))
    else:
        st.markdown("#### Fixed RAG configuration")
        st.caption("Prompt-only scope: set the RAG config the prompt is optimized at.")
        f1, f2, f3 = st.columns(3)
        with f1:
            fixed_cs = st.number_input("Chunk size (chars)", 64, 8192, 512, disabled=running)
        with f2:
            fixed_co = st.number_input("Chunk overlap (chars)", 0, 1024, 50, disabled=running)
        with f3:
            fixed_tk = st.number_input("Top-k", 1, 50, 5, disabled=running)
        chunk_sizes, chunk_overlaps, top_ks = [int(fixed_cs)], [int(fixed_co)], [int(fixed_tk)]
        if fixed_co >= fixed_cs:
            search_space_ok = False
            st.warning("Chunk overlap must be smaller than the chunk size.")

    if stages in ("all", "prompt"):
        st.markdown("#### Prompt optimization parameters")
        st.caption("DSPy tuning of the generator prompt at the winning RAG config.")
        o1, o2 = st.columns(2)
        with o1:
            dspy_optimizer = st.selectbox(
                "Prompt optimizer (DSPy)", ["gepa", "mipro", "copro"], index=0, disabled=running,
                help="'gepa' (reflective evolution, default), 'mipro' (Bayesian "
                "instruction+demo search), or 'copro' (iterative rewrite). "
                "Note: 'mipro' needs at least 2 questions.",
            )
        with o2:
            mipro_auto = st.selectbox(
                "Optimizer budget", ["light", "medium", "heavy"], index=0, disabled=running,
                help="How hard the optimizer searches. GEPA: ~30 / 100 / 300 LLM "
                "calls. Use 'light' for a quick run.",
            )

    st.markdown("#### Composite objective weights")
    st.caption(
        "The weighted metrics the RAG Bayesian search optimizes (and the "
        "report's headline comparison). Defaults follow groundedness > "
        "correctness > retrieval quality > auxiliary judges; lower-is-better "
        "metrics (hallucination/bias/toxicity) are auto-inverted."
    )
    _gt_mode_now = "with_gt" if gt_file is not None else "no_gt"
    # Always show the FULL metric set (with-GT superset); GT-only
    # metrics are disabled in no-GT mode instead of hidden, so users
    # can see what providing ground truth would unlock.
    _full_defaults = dict(default_objective("with_gt").metrics)
    _mode_defaults = dict(default_objective(_gt_mode_now).metrics)
    _gt_only = set(_full_defaults) - set(default_objective("no_gt").metrics)
    # Which metrics are actually computed PER TRIAL during the RAG
    # search vs only once on the final answers (deepeval supplement).
    _final_only = {"g_eval", "hallucination", "bias", "toxicity"}
    objective_weights: dict[str, float] = {}
    with st.expander("Adjust metric weights", expanded=False):
        def _weight_row(metric: str, wdefault: float) -> None:
            gt_locked = metric in _gt_only and _gt_mode_now == "no_gt"
            val = st.number_input(
                metric + (" (needs GT)" if gt_locked else ""),
                0.0, 10.0, float(wdefault), step=0.1,
                disabled=running or gt_locked,
                key=f"w_{metric}",
                help="Requires a ground-truth file — upload one to enable."
                if gt_locked else None,
            )
            if not gt_locked:
                objective_weights[metric] = val

        st.markdown(
            "**Computed per trial during the RAG search** — these weights "
            "directly steer the Bayesian optimization:"
        )
        live_metrics = [m for m in _full_defaults if m not in _final_only]
        wcols = st.columns(3)
        for i, m in enumerate(live_metrics):
            with wcols[i % 3]:
                _weight_row(m, _full_defaults[m])

        st.markdown(
            "**Computed once on the final answers** (deepeval supplement) — "
            "these weight the report's headline composite but do NOT steer "
            "the per-trial search:"
        )
        fcols = st.columns(4)
        for i, m in enumerate(sorted(_final_only)):
            with fcols[i % 4]:
                _weight_row(m, _full_defaults[m])

    weights_customized = objective_weights != _mode_defaults
    if weights_customized:
        st.info("Custom objective weights will be used for this run.")

    st.markdown("#### Baseline system prompt")
    st.caption("The RAG system prompt used by the baseline (DSPy may evolve it further).")
    system_instruction = st.text_area(
        "System instruction (optional)",
        value="",
        disabled=running,
        help="Inject domain guidance (e.g. 'You are an ESG analyst; answer "
        "only from the context; cite figures precisely'). Leave blank to "
        "use the default shown below.",
    )
    st.caption("Default used when left blank:")
    st.code(DEFAULT_SYSTEM_INSTRUCTION, language="text")

    if st.button("✓ Confirm experiment settings", disabled=running):
        if not search_space_ok:
            st.warning("Fix the search space first (every axis needs a value).")
        else:
            ss.confirm_settings = True
    if ss.confirm_settings:
        st.success("Experiment settings confirmed.")

    # =============================================================
    # 3 · Run
    # =============================================================
    st.subheader("3 · Run")

    # Workload estimate: LLM-call counts are derived from the pipeline's
    # actual call structure; wall-time assumes ~10 s/call at concurrency 2
    # (GLM-4-Flash-like). Faster endpoints finish sooner.
    est_q = len(custom_questions) if custom_questions else (
        len(gt_records) if gt_records is not None else int(n_questions)
    )
    est_rows, est_calls, est_time = _estimate_workload(
        n_questions=est_q,
        n_bo_trials=int(n_bo_trials),
        dspy_optimizer=dspy_optimizer,
        budget=mipro_auto,
        synthesize=gt_records is None and not custom_questions,
        stages=stages,
    )
    st.markdown("**Estimated workload** (current settings)")
    st.table(est_rows)
    st.caption(
        f"Total ≈ {est_calls} LLM calls, {est_time}. Assumes ~10 s/call at "
        "concurrency 2 (typical for GLM-4-Flash); faster models/endpoints "
        "cut this roughly proportionally. Call counts are structural "
        "estimates — judge retries can add a few percent."
    )

    ss.setdefault("default_run_name", f"exp-{datetime.now():%Y%m%d-%H%M%S}")
    run_name_raw = st.text_input(
        "Experiment name",
        value=ss.default_run_name,
        key="run_name_input",
        disabled=running,
        help="The run is saved under this name — pick it later in the "
        "sidebar to view the report or resume after an interruption.",
    )
    run_name = _slugify(run_name_raw)
    name_taken = (_RUNS_ROOT / run_name).exists()
    if name_taken:
        st.warning(f"Experiment `{run_name}` already exists — pick another name "
                   "(or select it in the sidebar to view / resume it).")

    can_run = (
        bool(pdf_files) and gt_ready and ss.confirm_upload and ss.confirm_settings
        and not running and not name_taken
    )
    if not (ss.confirm_upload and ss.confirm_settings):
        st.caption("Confirm sections 1 and 2 above to enable the run.")
    if st.button("▶ Run experiment", type="primary", disabled=not can_run):
        _start_run(
            ss=ss,
            run_name=run_name,
            pdf_files=pdf_files,
            gt_records=gt_records,
            custom_questions=custom_questions,
            write_questions_jsonl=write_questions_jsonl,
            run_experiment=run_experiment,
            api_key=api_key or None,
            settings=dict(
                model=model,
                api_base=api_base,
                n_questions=int(n_questions),
                n_bo_trials=int(n_bo_trials),
                n_bo_init=int(n_bo_init),
                seed=int(seed),
                llm_max_concurrent=int(llm_concurrency),
                system_instruction=system_instruction or None,
                dspy_optimizer=dspy_optimizer,
                mipro_auto=mipro_auto,
                stages=stages,
                chunk_sizes=sorted(chunk_sizes),
                chunk_overlaps=sorted(chunk_overlaps),
                top_ks=sorted(top_ks),
                # Stored as a plain dict (JSON-safe for meta.json); the
                # worker converts it into objective_overrides.
                objective_weights=objective_weights,
            ),
        )
        st.rerun()

    # ------------------------------------------------- live progress view
    if ss.phase == "running":
        _drain_and_render_progress(ss, st)

    if ss.phase == "error":
        st.error(f"Experiment failed: {ss.error}")

    # --------------------------------------------------------- report view
    if ss.phase == "done" and ss.bundle is not None:
        st.subheader("4 · Report")
        _render_report_and_downloads(
            st, render_report, ss.bundle,
            Path(ss.output_dir) if ss.output_dir else None,
        )


# --------------------------------------------------------------- helpers
def _parse_int_list(raw: str, lo: int, hi: int) -> tuple[list[int], str]:
    """Parse a comma/space-separated list of ints within ``[lo, hi]``.

    Returns ``(sorted unique values, error_message)`` — error is ""
    when the input is valid. Tolerates Chinese/Western commas, spaces,
    and semicolons as separators.
    """
    vals: list[int] = []
    for tok in re.split(r"[,;\s，、；]+", raw.strip()):  # noqa: RUF001 — fullwidth separators are intentional
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            return [], f"'{tok}' is not a whole number"
        if not (lo <= v <= hi):
            return [], f"{v} is outside the allowed range [{lo}, {hi}]"
        vals.append(v)
    if not vals:
        return [], "enter at least one value"
    return sorted(set(vals)), ""


def _fmt_minutes(calls: int, sec_per_call: float = 10.0, concurrency: int = 2) -> str:
    """Rough wall-time range for ``calls`` LLM calls (±~40%)."""
    mid = calls * sec_per_call / concurrency
    lo, hi = mid * 0.6, mid * 1.4
    if hi < 90:
        return "< 2 min"
    return f"~{max(1, round(lo / 60))}-{max(2, round(hi / 60))} min"


def _estimate_workload(
    *, n_questions: int, n_bo_trials: int, dspy_optimizer: str,
    budget: str, synthesize: bool, stages: str = "all",
) -> tuple[list[dict], int, str]:
    """Static per-stage LLM-call estimate for the current settings.

    Counts mirror the pipeline's call structure: each evaluated answer
    costs 1 generation + ~4 judge calls (ragas metrics); the prompt
    optimizer's inner loop uses its budget preset; the deepeval
    supplement scores both prompt variants on 4 extra metrics.
    """
    q = max(1, int(n_questions))
    per_q_eval = 5  # 1 generation + ~4 ragas judge calls
    run_rag = stages in ("all", "rag")
    run_prompt = stages in ("all", "prompt")

    synth_calls = q if synthesize else 0
    bo_calls = int(n_bo_trials) * q * per_q_eval if run_rag else 0
    optimizer_body = {
        "gepa": {"light": 30, "medium": 100, "heavy": 300},
        "mipro": {"light": 30, "medium": 70, "heavy": 150},
        "copro": {"light": 30, "medium": 30, "heavy": 30},
    }.get(dspy_optimizer, {}).get(budget, 30)
    # baseline + re-eval + search; deepeval supplement only runs on the
    # prompt stage's answers.
    dspy_calls = (q * per_q_eval * 2 + optimizer_body) if run_prompt else 0
    supplement_calls = 2 * q * 4 if run_prompt else 0

    rows = [
        {
            "Stage": "Load & chunk documents" + (" + synthesize questions" if synthesize else ""),
            "LLM calls": f"~{synth_calls}" if synth_calls else "0",
            "Est. time": _fmt_minutes(synth_calls) if synth_calls else "< 1 min",
        },
    ]
    if run_rag:
        rows.append({
            "Stage": f"Bayesian RAG search ({n_bo_trials} trials x {q} questions)",
            "LLM calls": f"~{bo_calls}",
            "Est. time": _fmt_minutes(bo_calls),
        })
    if run_prompt:
        rows.append({
            "Stage": f"Prompt optimization ({dspy_optimizer}/{budget} + before/after eval)",
            "LLM calls": f"~{dspy_calls}",
            "Est. time": _fmt_minutes(dspy_calls),
        })
        rows.append({
            "Stage": "Final evaluation (deepeval supplement) & report",
            "LLM calls": f"~{supplement_calls}",
            "Est. time": _fmt_minutes(supplement_calls),
        })
    total = synth_calls + bo_calls + dspy_calls + supplement_calls
    return rows, total, _fmt_minutes(total)


def _start_run(
    *, ss, run_name, pdf_files, gt_records, custom_questions,
    write_questions_jsonl, run_experiment, api_key, settings,
) -> None:
    """Persist uploads into the named run dir and launch the experiment."""
    from ragdx.schemas.models import DatasetRecord

    run_dir = _RUNS_ROOT / run_name
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths: list[str] = []
    for f in pdf_files:
        p = inputs_dir / f.name
        p.write_bytes(f.getvalue())
        pdf_paths.append(str(p))

    has_gt = gt_records is not None
    questions_path = None
    if has_gt:
        questions_path = str(write_questions_jsonl(gt_records, inputs_dir / "questions.jsonl"))
    elif custom_questions:
        records = [DatasetRecord(question=q, ground_truth=None, contexts=[]) for q in custom_questions]
        questions_path = str(write_questions_jsonl(records, inputs_dir / "questions.jsonl"))

    # meta.json makes the run re-viewable and resumable later. The API
    # key deliberately never lands on disk — resume takes the current
    # sidebar value instead.
    meta = {
        "name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "corpus": pdf_paths,
        "questions_path": questions_path,
        "has_gt": has_gt,
        "mode": "with_gt" if has_gt else "no_gt",
        "settings": settings,
    }
    _save_run_meta(run_dir, meta)

    _launch_worker(
        ss=ss,
        run_experiment=run_experiment,
        run_dir=run_dir,
        meta=meta,
        api_key=api_key,
        resume="",
    )


def _launch_worker(*, ss, run_experiment, run_dir, meta, api_key, resume) -> None:
    """Start (or resume) the engine in a daemon thread.

    Everything the worker needs is captured as a local before the thread
    starts — Streamlit's session_state is not accessible from spawned
    threads. The worker also keeps ``meta.json`` up to date so the run
    can be found, viewed, and resumed across app restarts.
    """
    # Hard guard against stacking a second engine thread (e.g. Resume
    # clicked in a fresh session while an adopted-run check was missed).
    if _active_run_alive():
        ss.error = (
            f"Experiment `{_ACTIVE_RUN.get('name')}` is still running — "
            "wait for it to finish before starting another."
        )
        ss.phase = "error"
        return

    ss.events = []
    ss.logs = []
    ss.bundle = None
    ss.error = None
    ss.output_dir = str(run_dir / "out")
    ss.run_name = meta["name"]
    ss.phase = "running"
    ss._checklist_state = None
    ss.queue = queue.Queue()

    q = ss.queue
    corpus_list = list(meta["corpus"])
    corpus = corpus_list if len(corpus_list) > 1 else corpus_list[0]
    questions_path = meta.get("questions_path")
    has_gt = bool(meta.get("has_gt"))
    mode = meta.get("mode") or ("with_gt" if has_gt else "no_gt")
    settings = dict(meta.get("settings") or {})
    output_dir = str(run_dir / "out")
    run_name = meta["name"]
    meta_snapshot = dict(meta)

    # Objective weights travel through meta.json as a plain dict; turn
    # them into the engine's objective_overrides here (only when they
    # differ from the defaults, so default runs keep default behavior).
    weights = settings.pop("objective_weights", None)
    objective_overrides = None
    if weights:
        from ragdx.optim.objectives import default_objective

        base_obj = default_objective(mode if mode in ("with_gt", "no_gt") else "no_gt")
        if dict(weights) != dict(base_obj.metrics):
            objective_overrides = {
                mode: base_obj.with_overrides(metrics={k: float(v) for k, v in weights.items()})
            }

    def _progress(event: dict[str, Any]) -> None:
        q.put(("event", event))

    def _update_meta(**fields) -> None:
        meta_snapshot.update(fields)
        try:
            _save_run_meta(run_dir, meta_snapshot)
        except Exception:  # pragma: no cover - meta is best-effort
            pass

    # Persist "running" (and drop any error from a previous failed
    # attempt) to disk immediately — the viewer re-reads meta.json on
    # every rerun and would otherwise keep showing the stale error
    # banner for the whole resumed run.
    meta_snapshot.pop("error", None)
    _update_meta(status="running")

    def _worker() -> None:
        handler = _QueueLogHandler(q)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger()
        # The ragdx logger has propagate=False (it owns a console
        # handler), so a root-attached handler never sees its records —
        # attach directly to it as well.
        ragdx_logger = logging.getLogger("ragdx")
        # Root defaults to WARNING, which drops third-party INFO records
        # (ragas / datasets / dspy sub-loggers) at the source. Open it to
        # INFO for the run, but pin the per-HTTP-call chatterboxes to
        # WARNING so the feed stays readable.
        noisy = ["httpx", "httpcore", "urllib3", "LiteLLM", "LiteLLM Router", "openai", "filelock"]
        prior_root_level = root.level
        prior_noisy = {n: logging.getLogger(n).level for n in noisy}
        root.addHandler(handler)
        ragdx_logger.addHandler(handler)
        root.setLevel(logging.INFO)
        for n in noisy:
            logging.getLogger(n).setLevel(logging.WARNING)
        try:
            result = run_experiment(
                corpus=corpus,
                has_gt=has_gt,
                mode=mode,
                questions_path=questions_path,
                output_dir=output_dir,
                progress_callback=_progress,
                api_key=api_key,
                resume=resume,
                experiment_group=run_name,
                objective_overrides=objective_overrides,
                **settings,
            )
            _update_meta(
                status="done",
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
            q.put(("done", result.bundle))
        except Exception as exc:
            _update_meta(status="error", error=f"{type(exc).__name__}: {exc}")
            q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            root.removeHandler(handler)
            ragdx_logger.removeHandler(handler)
            root.setLevel(prior_root_level)
            for n, lvl in prior_noisy.items():
                logging.getLogger(n).setLevel(lvl)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ss.thread = t
    # Register process-wide so fresh browser sessions (page refresh)
    # adopt this run instead of stacking another engine thread.
    _ACTIVE_RUN.clear()
    _ACTIVE_RUN.update(
        name=meta["name"],
        queue=q,
        thread=t,
        events=ss.events,
        logs=ss.logs,
        output_dir=str(run_dir / "out"),
    )


class _QueueLogHandler(logging.Handler):
    """Push formatted log records into the progress queue."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put(("log", self.format(record)))
        except Exception:  # pragma: no cover - never let logging break the run
            pass


_CHECKLIST_CSS = """<style>
@keyframes ragdx-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.ragdx-stage { margin: 4px 0; font-size: 15px; }
.ragdx-stage.done { color: #15803d; }
.ragdx-stage.pending { color: #9ca3af; }
.ragdx-stage.running { color: #1a73e8; font-weight: 600; }
.ragdx-spinner { display: inline-block; width: 13px; height: 13px;
  border: 2px solid #1a73e8; border-top-color: transparent;
  border-radius: 50%; animation: ragdx-spin 0.8s linear infinite;
  vertical-align: -2px; margin-right: 7px; }
</style>"""


def _render_stage_checklist(ss, container) -> None:
    """Per-stage checklist: done / running (animated spinner) / pending.

    Only re-renders when the stage state actually changes — the drain
    loop ticks every 0.4 s, and replacing the HTML each tick would
    restart the CSS spin animation, making it stutter.
    """
    seen = {e.get("stage") for e in ss.events}
    done_flags = tuple(marker in seen for marker, _ in _STAGE_CHECKLIST)
    state_key = (done_flags, ss.phase)
    if ss.get("_checklist_state") == state_key:
        return
    ss._checklist_state = state_key

    items = []
    for i, (_marker, label) in enumerate(_STAGE_CHECKLIST):
        if done_flags[i]:
            items.append(f'<div class="ragdx-stage done">✅ {label}</div>')
        elif (i == 0 or done_flags[i - 1]) and ss.phase == "running":
            items.append(
                '<div class="ragdx-stage running">'
                f'<span class="ragdx-spinner"></span>{label}…</div>'
            )
        else:
            items.append(f'<div class="ragdx-stage pending">⬜ {label}</div>')
    container.markdown(_CHECKLIST_CSS + "".join(items), unsafe_allow_html=True)


def _drain_and_render_progress(ss, st) -> None:
    """Block this script run, live-updating a progress bar, per-stage
    checklist, and detailed log feed until the worker thread finishes."""
    bar = st.progress(0.0, text="Starting…")
    status = st.empty()
    heartbeat = st.empty()
    checklist = st.container()
    checklist_slot = checklist.empty()
    log_box = st.expander("Detailed log", expanded=True)
    log_area = log_box.empty()

    q = ss.queue
    pct = 0.0
    run_started = time.time()
    last_event_at = time.time()
    last_hb = 0.0
    while True:
        drained = False
        try:
            while True:
                kind, payload = q.get_nowait()
                drained = True
                if kind == "event":
                    pct = float(payload.get("pct", pct))
                    detail = payload.get("detail") or payload.get("stage", "")
                    mode = payload.get("mode")
                    label = f"[{mode}] {detail}" if mode else detail
                    ss.events.append(payload)
                    last_event_at = time.time()
                    bar.progress(min(1.0, pct), text=label)
                    if payload.get("stage") in {m for m, _ in _STAGE_CHECKLIST}:
                        status.success(f"✓ {label}  ·  {int(pct * 100)}%")
                    else:
                        status.markdown(f"**{label}**  ·  {int(pct * 100)}%")
                elif kind == "log":
                    ss.logs.append(payload)
                    last_event_at = time.time()
                elif kind == "done":
                    ss.bundle = payload
                    ss.phase = "done"
                    bar.progress(1.0, text="Complete")
                elif kind == "error":
                    ss.error = payload
                    ss.phase = "error"
        except queue.Empty:
            pass

        # Heartbeat: long evaluation batches (ragas judging a trial's
        # answers) emit neither events nor log records — show elapsed
        # time so a quiet stretch doesn't read as a hang.
        now = time.time()
        if now - last_hb >= 1.0:
            last_hb = now
            total_m, total_s = divmod(int(now - run_started), 60)
            quiet = int(now - last_event_at)
            quiet_txt = ""
            if quiet >= 30:
                q_m, q_s = divmod(quiet, 60)
                quiet_txt = (
                    f" · current step running {q_m} min {q_s:02d} s "
                    "(evaluation batches log nothing until they finish — "
                    "still working)"
                )
            heartbeat.caption(f"⏱ elapsed {total_m} min {total_s:02d} s{quiet_txt}")

        _render_stage_checklist(ss, checklist_slot)
        if ss.logs:
            log_area.code("\n".join(ss.logs[-300:]), language="log")

        if ss.phase in ("done", "error"):
            break

        thread = ss.get("thread")
        if thread is not None and not thread.is_alive() and not drained and q.empty():
            ss.error = "Experiment thread exited unexpectedly."
            ss.phase = "error"
            break
        time.sleep(0.4)

    st.rerun()


def _render_run_viewer(*, ss, st, name, render_report, run_experiment, api_key) -> None:
    """Sidebar-selected saved experiment: config, report, and resume."""
    run_dir = _RUNS_ROOT / name
    meta = _load_run_meta(run_dir)
    st.subheader(f"Experiment · {name}")
    if meta is None:
        st.error("This experiment's meta.json is missing or unreadable.")
        return

    status = meta.get("status", "?")
    badge = {"done": "✅ done", "running": "⏸ interrupted / running", "error": "❌ error"}.get(status, status)
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", badge)
    c2.metric("Created", meta.get("created_at", "—"))
    c3.metric("Finished", meta.get("finished_at", "—"))
    # Only surface the stored error while the run is actually IN the
    # error state — a resumed (running) or completed run must not keep
    # showing a stale banner from an earlier failed attempt.
    if meta.get("error") and status == "error":
        st.error(meta["error"])

    with st.expander("Configuration", expanded=(status != "done")):
        st.json({
            "mode": meta.get("mode"),
            "corpus": meta.get("corpus"),
            "questions_path": meta.get("questions_path"),
            **(meta.get("settings") or {}),
        })

    # Live progress for the run currently executing in this session.
    if ss.phase == "running":
        if ss.get("run_name") == name:
            _drain_and_render_progress(ss, st)
        else:
            st.info(f"Another experiment (`{ss.get('run_name')}`) is running — "
                    "wait for it to finish before resuming this one.")
        return

    result_json = run_dir / "out" / "result.json"
    if result_json.exists():
        try:
            bundle = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception as exc:
            st.error(f"Could not read result.json: {exc}")
            bundle = None
        if bundle is not None:
            st.subheader("Report")
            _render_report_and_downloads(st, render_report, bundle, run_dir / "out")

    # Resume: anything not marked done can pick up from its checkpoints.
    if status != "done":
        st.divider()
        st.markdown(
            "**Resume** — completed stages replay from checkpoints without "
            "LLM calls; the interrupted stage continues from its last saved "
            "trial/phase."
        )
        if st.button("⟳ Resume this experiment", type="primary"):
            resume_meta = {**meta, "status": "running"}
            # Reuse the questions the interrupted run synthesized (the
            # engine persists them next to the bundle) — skips the
            # re-synthesis LLM calls and keeps checkpointed trial scores
            # comparable.
            if not resume_meta.get("questions_path"):
                synth_q = run_dir / "out" / "questions_synthesized.jsonl"
                if synth_q.exists():
                    resume_meta["questions_path"] = str(synth_q)
            _launch_worker(
                ss=ss,
                run_experiment=run_experiment,
                run_dir=run_dir,
                meta=resume_meta,
                api_key=api_key,
                resume=name,
            )
            st.rerun()


def _render_report_and_downloads(st, render_report, bundle, out_dir: Path | None) -> None:
    """Render the HTML report inline and offer download buttons."""
    import streamlit.components.v1 as components

    try:
        html = render_report(bundle)
    except Exception as exc:
        st.error(f"Could not render report: {exc}")
        html = None

    cols = st.columns(3)
    if html is not None:
        cols[0].download_button(
            "⬇ Report (HTML)", data=html, file_name="ragdx_report.html", mime="text/html"
        )
    result_json = out_dir / "result.json" if out_dir else None
    if result_json and result_json.exists():
        cols[1].download_button(
            "⬇ Bundle (result.json)",
            data=result_json.read_bytes(),
            file_name="result.json",
            mime="application/json",
        )
    final_dir = out_dir / "final" if out_dir else None
    if final_dir and final_dir.exists() and any(final_dir.iterdir()):
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(final_dir.iterdir()):
                if p.is_file():
                    zf.write(p, arcname=p.name)
        cols[2].download_button(
            "⬇ Final config + prompt (.zip)",
            data=buf.getvalue(),
            file_name="ragdx_final.zip",
            mime="application/zip",
        )

    if html is not None:
        components.html(html, height=1400, scrolling=True)


if __name__ == "__main__":
    _run_app()
