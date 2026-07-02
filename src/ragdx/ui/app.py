"""ragdx studio — upload → run → report Streamlit app.

A single-page studio around :func:`ragdx.experiments.run_experiment`:

1. Upload one or more PDFs (required, pooled into one corpus) and,
   optionally, an Excel/CSV ground-truth file. A GT file switches the
   run into **with-GT** mode; without one it runs in **no-GT** mode,
   where you either synthesize questions from the documents or type
   your own.
2. For with-GT, map the GT columns to ``question`` / ``ground_truth`` /
   ``contexts`` (auto-detected, with a manual fallback).
3. Tune the experiment settings, then run with a live progress bar,
   per-stage status, and a detailed log feed.
4. View the rendered report inline and download the HTML report, the
   raw ``result.json`` bundle, and the ship-ready ``final/`` artifacts.

Launch it with ``ragdx ui`` (or ``ragdx-ui`` / ``streamlit run
src/ragdx/ui/app.py``).
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
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

    from ragdx.experiments import run_experiment
    from ragdx.loaders.tabular import (
        detect_mapping,
        load_gt_table,
        records_from_table,
        write_questions_jsonl,
    )
    from ragdx.ui.experiment_report import render_report

    st.set_page_config(page_title="ragdx studio", page_icon="🔬", layout="wide")
    st.title("🔬 ragdx studio")
    st.caption(
        "Upload documents, optionally add ground truth, and run the "
        "end-to-end RAG optimization experiment — then view and download the report."
    )

    ss = st.session_state
    ss.setdefault("phase", "idle")  # idle | running | done | error
    ss.setdefault("events", [])
    ss.setdefault("logs", [])
    ss.setdefault("bundle", None)
    ss.setdefault("output_dir", None)
    ss.setdefault("error", None)

    # ------------------------------------------------- sidebar: connection
    with st.sidebar:
        st.header("Connection")
        st.caption("LLM endpoint used for generation, judging, and prompt tuning.")
        model = st.text_input("Model", value="openai/glm-4-flash")
        api_base = st.text_input("API base URL", value="https://open.bigmodel.cn/api/paas/v4")
        api_key = st.text_input(
            "API key",
            value=os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            type="password",
            help="Falls back to ZHIPU_API_KEY / OPENAI_API_KEY if left blank.",
        )

    running = ss.phase == "running"

    # ------------------------------------------------------- 1 · upload
    st.subheader("1 · Upload documents")
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

    # ------------------------------ 2 · GT mapping OR no-GT question source
    gt_records = None
    gt_ready = True
    custom_questions: list[str] | None = None

    if gt_file is not None:
        st.subheader("2 · Map ground-truth columns")
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
        st.subheader("2 · Questions (no-GT mode)")
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
            st.info("Questions will be synthesized from the uploaded documents.")

    # --------------------------------------------- 3 · experiment settings
    st.subheader("3 · Experiment settings")
    synth_mode = gt_file is None and not custom_questions
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_questions = st.number_input(
            "Questions to synthesize",
            1, 100, 5,
            disabled=running or not synth_mode,
            help="How many questions to synthesize from the documents. "
            "Only used in no-GT mode when you don't supply your own "
            "questions (ignored otherwise).",
        )
    with c2:
        n_bo_trials = st.number_input(
            "Bayesian search trials",
            2, 64, 8,
            disabled=running,
            help="Total number of RAG configurations the Bayesian optimizer "
            "evaluates. Each trial builds the pipeline with a candidate "
            "(chunk size, overlap, top-k) and scores it. More trials search "
            "more thoroughly but cost more time and LLM calls.",
        )
    with c3:
        n_bo_init = st.number_input(
            "Bayesian init rounds",
            1, 16, 3,
            disabled=running,
            help="Number of initial configurations sampled at random before "
            "the Bayesian model starts steering the search. These seed the "
            "surrogate model; keep it <= trials (typically 2-3).",
        )
    with c4:
        seed = st.number_input(
            "Seed", 0, 10_000, 7, disabled=running,
            help="Deterministic seed for the search and question synthesis.",
        )
    system_instruction = st.text_area(
        "System instruction (optional)",
        value="",
        disabled=running,
        help="RAG system prompt shared by the generation path and the DSPy "
        "baseline. Leave blank for the generic default; DSPy may evolve it further.",
    )

    # -------------------------------------------------------------- 4 · run
    st.subheader("4 · Run")
    can_run = bool(pdf_files) and gt_ready and not running
    if st.button("▶ Run experiment", type="primary", disabled=not can_run):
        _start_run(
            ss=ss,
            pdf_files=pdf_files,
            gt_records=gt_records,
            custom_questions=custom_questions,
            write_questions_jsonl=write_questions_jsonl,
            run_experiment=run_experiment,
            settings=dict(
                model=model,
                api_base=api_base,
                api_key=api_key or None,
                n_questions=int(n_questions),
                n_bo_trials=int(n_bo_trials),
                n_bo_init=int(n_bo_init),
                seed=int(seed),
                system_instruction=system_instruction or None,
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
        st.subheader("5 · Report")
        _render_report_and_downloads(ss, st, render_report)


# --------------------------------------------------------------- helpers
def _start_run(
    *, ss, pdf_files, gt_records, custom_questions, write_questions_jsonl, run_experiment, settings
) -> None:
    """Persist uploads to a temp workdir and launch the run in a thread.

    Everything the worker thread needs is captured as a local *before*
    the thread starts — Streamlit's ``st.session_state`` is not
    accessible from spawned threads, so the worker must never touch it.
    """
    from ragdx.schemas.models import DatasetRecord

    workdir = Path(tempfile.mkdtemp(prefix="ragdx_run_"))
    # Multiple PDFs are pooled into a single corpus by the engine, which
    # accepts a list of paths. A single file is passed as a 1-item list.
    pdf_paths: list[str] = []
    for f in pdf_files:
        p = workdir / f.name
        p.write_bytes(f.getvalue())
        pdf_paths.append(str(p))

    has_gt = gt_records is not None
    questions_path = None
    if has_gt:
        # with-GT: reference answers drive answer-correctness metrics.
        questions_path = str(write_questions_jsonl(gt_records, workdir / "questions.jsonl"))
    elif custom_questions:
        # no-GT with user-supplied questions: same JSONL shape, GT = null.
        records = [DatasetRecord(question=q, ground_truth=None, contexts=[]) for q in custom_questions]
        questions_path = str(write_questions_jsonl(records, workdir / "questions.jsonl"))

    output_dir = str(workdir / "out")

    ss.events = []
    ss.logs = []
    ss.bundle = None
    ss.error = None
    ss.output_dir = output_dir
    ss.phase = "running"
    ss.queue = queue.Queue()

    # Locals captured by the closure — no session_state access in the thread.
    q = ss.queue
    corpus = pdf_paths if len(pdf_paths) > 1 else pdf_paths[0]

    def _progress(event: dict[str, Any]) -> None:
        q.put(("event", event))

    def _worker() -> None:
        # Attach to the ROOT logger so the feed captures ragdx *and* the
        # library internals (dspy / ragas / langchain) at INFO — the
        # "detailed log" the user sees during a run.
        handler = _QueueLogHandler(q)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            result = run_experiment(
                corpus=corpus,
                has_gt=has_gt,
                mode="with_gt" if has_gt else "no_gt",
                questions_path=questions_path,
                output_dir=output_dir,
                progress_callback=_progress,
                **settings,
            )
            q.put(("done", result.bundle))
        except Exception as exc:
            q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            root.removeHandler(handler)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ss.thread = t


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


def _render_stage_checklist(ss, container) -> None:
    """Render the per-stage checklist (✅ done / 🔄 running / ⬜ pending)."""
    seen = {e.get("stage") for e in ss.events}
    done_flags = [marker in seen for marker, _ in _STAGE_CHECKLIST]
    lines = []
    for i, (_marker, label) in enumerate(_STAGE_CHECKLIST):
        if done_flags[i]:
            icon = "✅"
        elif i == 0 or done_flags[i - 1]:
            icon = "🔄" if ss.phase == "running" else "⬜"
        else:
            icon = "⬜"
        lines.append(f"{icon} {label}")
    container.markdown("\n\n".join(lines))


def _drain_and_render_progress(ss, st) -> None:
    """Block this script run, live-updating a progress bar, per-stage
    checklist, and detailed log feed until the worker thread finishes."""
    bar = st.progress(0.0, text="Starting…")
    status = st.empty()
    checklist = st.container()
    checklist_slot = checklist.empty()
    log_box = st.expander("Detailed log", expanded=True)
    log_area = log_box.empty()

    q = ss.queue
    pct = 0.0
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
                    bar.progress(min(1.0, pct), text=label)
                    # Announce each completed milestone.
                    if payload.get("stage") in {m for m, _ in _STAGE_CHECKLIST}:
                        status.success(f"✓ {label}  ·  {int(pct * 100)}%")
                    else:
                        status.markdown(f"**{label}**  ·  {int(pct * 100)}%")
                elif kind == "log":
                    ss.logs.append(payload)
                elif kind == "done":
                    ss.bundle = payload
                    ss.phase = "done"
                    bar.progress(1.0, text="Complete")
                elif kind == "error":
                    ss.error = payload
                    ss.phase = "error"
        except queue.Empty:
            pass

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


def _render_report_and_downloads(ss, st, render_report) -> None:
    """Render the HTML report inline and offer download buttons."""
    import streamlit.components.v1 as components

    bundle = ss.bundle
    try:
        html = render_report(bundle)
    except Exception as exc:
        st.error(f"Could not render report: {exc}")
        html = None

    out_dir = Path(ss.output_dir) if ss.output_dir else None

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
