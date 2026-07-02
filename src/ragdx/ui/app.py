"""ragdx studio — upload → run → report Streamlit app.

A single-page studio around :func:`ragdx.experiments.run_experiment`:

1. Upload a PDF (required) and, optionally, an Excel/CSV ground-truth
   file. A GT file switches the run into **with-GT** mode; without one
   the questions are synthesized from the PDF (**no-GT** mode).
2. Map the GT columns to ``question`` / ``ground_truth`` / ``contexts``
   (auto-detected, with a manual fallback).
3. Run the end-to-end experiment with a live progress bar + log feed.
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
        "Upload a document, optionally add ground truth, and run the "
        "end-to-end RAG optimization experiment — then view and download the report."
    )

    ss = st.session_state
    ss.setdefault("phase", "idle")  # idle | running | done | error
    ss.setdefault("events", [])
    ss.setdefault("logs", [])
    ss.setdefault("bundle", None)
    ss.setdefault("output_dir", None)
    ss.setdefault("error", None)

    # ---------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Run settings")
        model = st.text_input("Model", value="openai/glm-4-flash")
        api_base = st.text_input("API base URL", value="https://open.bigmodel.cn/api/paas/v4")
        api_key = st.text_input(
            "API key",
            value=os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY") or "",
            type="password",
            help="Falls back to ZHIPU_API_KEY / OPENAI_API_KEY if left blank.",
        )
        n_questions = st.number_input("Questions", 1, 100, 5)
        n_bo_trials = st.number_input("Bayesian search trials", 2, 64, 8)
        n_bo_init = st.number_input("Bayesian init rounds", 1, 16, 3)
        seed = st.number_input("Seed", 0, 10_000, 7)
        system_instruction = st.text_area(
            "System instruction (optional)",
            value="",
            help="RAG system prompt shared by the BO generation path and the DSPy baseline. "
            "Leave blank for the generic default.",
        )

    running = ss.phase == "running"

    # ------------------------------------------------------------ upload
    st.subheader("1 · Upload")
    col_pdf, col_gt = st.columns(2)
    with col_pdf:
        pdf_file = st.file_uploader(
            "Document (PDF, required)", type=["pdf"], disabled=running
        )
    with col_gt:
        gt_file = st.file_uploader(
            "Ground truth (Excel/CSV, optional)",
            type=["csv", "tsv", "xlsx", "xls"],
            disabled=running,
            help="Provide to run in with-GT mode. Without it, questions are "
            "synthesized from the document (no-GT mode).",
        )

    # ------------------------------------------- GT preview + col mapping
    mapping: dict[str, str | None] | None = None
    gt_records = None
    gt_ready = True
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
                if default in options:
                    idx = options.index(default)
                elif optional:
                    idx = 0
                else:
                    idx = 0
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
        st.info("No ground-truth file — the run will use **no-GT** mode (synthesized questions).")

    # -------------------------------------------------------------- run
    st.subheader("3 · Run")
    can_run = (pdf_file is not None) and gt_ready and not running
    if st.button("▶ Run experiment", type="primary", disabled=not can_run):
        _start_run(
            ss=ss,
            pdf_file=pdf_file,
            gt_records=gt_records,
            run_experiment=run_experiment,
            write_questions_jsonl=write_questions_jsonl,
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
        st.subheader("4 · Report")
        _render_report_and_downloads(ss, st, render_report)


# --------------------------------------------------------------- helpers
def _start_run(*, ss, pdf_file, gt_records, run_experiment, write_questions_jsonl, settings) -> None:
    """Persist uploads to a temp workdir and launch the run in a thread."""
    workdir = Path(tempfile.mkdtemp(prefix="ragdx_run_"))
    pdf_path = workdir / pdf_file.name
    pdf_path.write_bytes(pdf_file.getvalue())

    has_gt = gt_records is not None
    questions_path = None
    if has_gt:
        questions_path = write_questions_jsonl(gt_records, workdir / "questions.jsonl")

    ss.events = []
    ss.logs = []
    ss.bundle = None
    ss.error = None
    ss.output_dir = str(workdir / "out")
    ss.phase = "running"
    ss.queue = queue.Queue()

    q = ss.queue

    def _progress(event: dict[str, Any]) -> None:
        q.put(("event", event))

    def _worker() -> None:
        handler = _QueueLogHandler(q)
        handler.setLevel(logging.INFO)
        ragdx_logger = logging.getLogger("ragdx")
        ragdx_logger.addHandler(handler)
        try:
            result = run_experiment(
                corpus=str(pdf_path),
                has_gt=has_gt,
                mode="with_gt" if has_gt else "no_gt",
                questions_path=str(questions_path) if questions_path else None,
                output_dir=ss.output_dir,
                progress_callback=_progress,
                **settings,
            )
            q.put(("done", result.bundle))
        except Exception as exc:
            q.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            ragdx_logger.removeHandler(handler)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    ss.thread = t


class _QueueLogHandler(logging.Handler):
    """Push formatted ``ragdx`` log records into the progress queue."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put(("log", self.format(record)))
        except Exception:  # pragma: no cover - never let logging break the run
            pass


def _drain_and_render_progress(ss, st) -> None:
    """Block this script run, live-updating a progress bar + log feed
    until the worker thread finishes."""
    bar = st.progress(0.0, text="Starting…")
    status = st.empty()
    log_box = st.expander("Live log", expanded=True)
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

        if ss.logs:
            log_area.code("\n".join(ss.logs[-200:]), language="log")

        if ss.phase in ("done", "error"):
            break

        thread = ss.get("thread")
        if thread is not None and not thread.is_alive() and not drained and q.empty():
            # Thread died without a terminal message — surface it.
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

    # Download buttons row.
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
    # Bundle the final/ deliverables (optimized config + prompt) as a zip.
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
