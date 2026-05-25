"""Streamlit dashboard for the GT-aware optimization demo.

Loads ``.ragdx_optimize_demo/result.json`` (produced by
``examples/demo_optimize_gt_modes.py``) and presents a side-by-side view of
the with-GT and no-GT pipelines:

* Data diagnostics: did the trainset have GT / answers / contexts?
* AutoRAG specs: which metrics get included, what objective gets chosen,
  what does the rendered evaluation block look like?
* Metric pre-flight: the same set of metrics fed into the GT-aware filter,
  showing what was kept and what was dropped (with reasons).
* DSPy optimisation: the optimized instructions and the bootstrapped
  few-shot demos for each (predictor, GT mode) cell.

Run::

    streamlit run src/ragdx/ui/optimization_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[3]
LIVE_RESULT = REPO / ".ragdx_optimize_demo" / "result.json"
COMMITTED_RESULT = REPO / "docs" / "examples" / "optimize_gt_modes_result.json"
# Prefer the fresh local run; fall back to the committed snapshot so this
# dashboard works after a clean clone before any LLM calls have been made.
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


def show_dspy_panel(panel: dict) -> None:
    if "error" in panel:
        st.error(panel["error"])
        return
    cols = st.columns(3)
    cols[0].metric("Optimizer", panel["optimizer"])
    cols[1].metric("Trainset", panel["trainset_size"])
    cols[2].metric("GT mode", panel["gt_mode"])

    st.write("### Optimized instructions")
    if not panel["instructions"]:
        st.info("No instructions returned. (DSPy version may not expose them via named_predictors.)")
    for name, instr in panel["instructions"].items():
        st.write(f"**Predictor `{name}`**")
        st.code(instr or "(empty)", language="text")

    st.write("### Bootstrapped few-shot demos")
    if not panel["demos"]:
        st.info("No demos returned.")
    for name, demos in panel["demos"].items():
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
    cols[2].metric("Cells", "4 (AutoRAG x 2  +  DSPy x 2)")

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

    # -- STEP 4: DSPy MIPROv2 results
    st.divider()
    st.subheader("Step 4 — DSPy MIPROv2 optimization")
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
