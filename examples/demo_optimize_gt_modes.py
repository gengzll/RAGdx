"""End-to-end RAG optimization demo with real LLM, AutoRAG grid + DSPy A/B.

Drives the full ragdx optimization story on the amnesty_qa[:5] dataset:

  1. Load corpus + records (real questions, real GT, ~15 corpus chunks).
  2. Show AutoRAG / DSPy specs in both GT modes (descriptive).
  3. Metric pre-flight (descriptive — filtered metric set per GT mode).
  4. AutoRAG grid search (REAL runs): for each top_k in {1, 3, 6}, build a
     RAG pipeline, generate answers, evaluate with ragas. Pick the best
     ``top_k`` by ``faithfulness``.
  5. DSPy before/after (REAL runs): using the best top_k from step 4,
     - baseline:  default RAG Signature -> generate -> ragas eval
     - optimized: MIPROv2 (with-GT path) -> rebuilt program -> generate -> ragas eval
     - compute per-metric delta
  6. DSPy MIPROv2 trial chart for both GT modes (descriptive — uses the
     existing trainset + adapter to show metric_kind contrast).

All cells call the LLM (GLM-4-Flash by default). There is no offline
path; if you don't have a key the demo errors immediately at startup.

Outputs are written to ``.ragdx_optimize_demo/result.json`` for the
Streamlit dashboard (``streamlit run src/ragdx/ui/optimization_dashboard.py``)
and also echoed to stdout.

Usage::

    ZHIPU_API_KEY=<key> PYTHONPATH=src python examples/demo_optimize_gt_modes.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUTPUT_ROOT = REPO / ".ragdx_optimize_demo"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_ROOT / "result.json"

# Pick up GLM credentials in the same style as the other real-LLM demos.
KEY = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not KEY:
    raise SystemExit(
        "ZHIPU_API_KEY (or OPENAI_API_KEY) is required. Run with:\n"
        "    ZHIPU_API_KEY=<key> PYTHONPATH=src python examples/demo_optimize_gt_modes.py"
    )

# Configure OpenAI-compatible env vars so LiteLLM / DSPy / ragas route to Zhipu.
os.environ["OPENAI_API_KEY"] = KEY
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# GLM-4-Flash rejects temperature < 0.01 ("API parameter error"). DSPy
# / ragas / litellm sometimes default to 1e-5 internally, so clamp.
import litellm  # noqa: E402

_orig = litellm.batch_completion
_orig_completion = litellm.completion


def _clamp(kwargs):
    t = kwargs.get("temperature")
    if t is not None and 0 < t < 0.01:
        kwargs["temperature"] = 0.01
    return kwargs


def _batch(*a, **kw):
    return _orig(*a, **_clamp(kw))


def _comp(*a, **kw):
    return _orig_completion(*a, **_clamp(kw))


litellm.batch_completion = _batch
litellm.completion = _comp

import logging  # noqa: E402

from ragdx import UnifiedEvaluator  # noqa: E402
from ragdx.optim._gt_helpers import (  # noqa: E402
    filter_metrics_by_data,
    gt_mode,
    has_answers,
    has_contexts,
    has_ground_truth,
)
from ragdx.optim.autorag_adapter import AutoRAGAdapter  # noqa: E402
from ragdx.optim.dspy_adapter import DSPyAdapter  # noqa: E402
from ragdx.schemas.models import DatasetRecord, OptimizationExperiment  # noqa: E402


# ----------------------------------------- DSPy trial-score capture helper
class _MIPROTrialScoreCapture(logging.Handler):
    """Attach to ``dspy.teleprompt.mipro_optimizer_v2`` to record the per-trial
    score progression and the running-best so we can plot them later.

    MIPROv2 logs lines like ``Scores so far: [21.65, 21.32, ...]`` and
    ``Best score so far: 28.6`` -- we parse those out instead of patching
    the teleprompter internals.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scores_so_far: list[float] = []
        self.best_scores: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Scores so far:" in msg:
            tail = msg.split("Scores so far:", 1)[1].strip()
            try:
                self.scores_so_far = json.loads(tail)
            except Exception:
                pass
        elif "Best score so far:" in msg:
            tail = msg.split("Best score so far:", 1)[1].strip()
            try:
                self.best_scores.append(float(tail))
            except Exception:
                pass

    def reset(self) -> None:
        self.scores_so_far = []
        self.best_scores = []


# ----------------------------------------------------------------- corpus
def load_amnesty_corpus():
    """Load ``explodinggradients/amnesty_qa[:5]``: real Q + GT + chunks.

    Returns ``(corpus_chunks, records)`` where ``corpus_chunks`` is a list
    of strings (about 15 chunks total) and ``records`` is a list of
    ``DatasetRecord`` with question + ground_truth populated (contexts
    are left empty -- they get filled per-retrieval-config in step 4/5).
    """
    from datasets import load_dataset

    ds = load_dataset("explodinggradients/amnesty_qa", "english_v3", split="eval[:5]")
    corpus_chunks: list[str] = []
    records: list[DatasetRecord] = []
    for row in ds:
        for passage in (row.get("retrieved_contexts") or []):
            text = (passage or "").strip()
            if text:
                corpus_chunks.append(text)
        records.append(
            DatasetRecord(
                question=row["user_input"],
                ground_truth=row.get("reference") or "",
                contexts=[],
            )
        )
    return corpus_chunks, records


def build_records(records: list[DatasetRecord], *, with_gt: bool) -> list[DatasetRecord]:
    """Clone records with GT either preserved or erased -- for GT-mode contrast."""
    return [
        DatasetRecord(
            question=r.question,
            answer=r.answer,
            ground_truth=r.ground_truth if with_gt else None,
            contexts=list(r.contexts),
        )
        for r in records
    ]


# -------------------------------------------------- RAG pipeline runners
def build_embeddings():
    """Local sentence-transformers, no API call. Shared across configs."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks, embeddings):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_texts(chunks, embeddings)


def retrieve(vstore, question: str, top_k: int) -> list[str]:
    docs = vstore.similarity_search(question, k=top_k)
    return [d.page_content for d in docs]


def generate_answer(question: str, contexts: list[str], lm_callable) -> str:
    """Single-shot answer generation with a basic RAG prompt template."""
    ctx_str = "\n---\n".join(contexts) if contexts else "(no context)"
    prompt = (
        "Answer the question using only the provided context.\n"
        "Be concise. Do not invent facts not in the context.\n\n"
        f"Context:\n{ctx_str}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        return lm_callable(prompt)
    except Exception as e:  # pragma: no cover -- live LLM
        return f"<generation error: {e}>"


def build_litellm_callable():
    """Returns a simple ``prompt -> answer_text`` callable backed by GLM."""

    def _call(prompt: str) -> str:
        resp = litellm.completion(
            model="openai/glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            api_key=KEY,
            api_base="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.01,
            max_tokens=350,
            timeout=60,
        )
        return resp.choices[0].message.content or ""

    return _call


def build_dspy_lm():
    import dspy

    return dspy.LM(
        "openai/glm-4-flash",
        api_key=KEY,
        api_base="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.01,
        max_tokens=400,
        cache=False,
    )


def build_ragas_judge():
    """ragas-compatible LLM wrapper around GLM-4-Flash."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model="glm-4-flash",
        api_key=KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.01,
        timeout=60,
        max_retries=2,
    )
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings():
    """ragas-compatible local HuggingFace embeddings (no API calls).

    Some ragas metrics (e.g. ``answer_relevancy``) need embeddings in
    addition to the LLM. Without this they fall back to OpenAI's
    text-embedding-ada-002 and hit ``api.openai.com``, which fails on
    networks without OpenAI access (and is also the wrong provider for
    GLM keys).
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def build_ref_free_ragas_metrics():
    """The reference-free metric subset (works in both GT modes)."""
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    return [faithfulness, answer_relevancy, context_precision]


def evaluate_records_with_ragas(records, judge, embeddings, metrics) -> dict:
    """Run UnifiedEvaluator + ragas on the records, return flat metric dict."""
    evaluator = UnifiedEvaluator()
    try:
        result = evaluator.evaluate(
            records,
            use_ragas=True, run_ragas=True,
            use_ragchecker=False, use_embedding=False,
            ragas_kwargs={
                "llm": judge,
                "embeddings": embeddings,
                "metrics": metrics,
            },
        )
        flat = {**dict(result.retrieval), **dict(result.generation), **dict(result.e2e)}
        return {
            "scores": flat,
            "skipped": result.metadata.get("skipped_metrics", {}),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "scores": {}}


# --------------------------------------------------------------- helpers
def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kv(label: str, value: object, width: int = 22) -> None:
    print(f"  {label:<{width}s} {value}")


# ------------------------------------ STEP 2 — AutoRAG spec (descriptive)
def show_autorag(records, label: str) -> dict:
    adapter = AutoRAGAdapter()
    exp = OptimizationExperiment(
        name=f"autorag-{label}",
        tool="autorag",
        target_component="pipeline",
        description=f"AutoRAG search in {label} mode",
        parameters={},
        objectives={"faithfulness": 1.0},
        search_space={"top_k": [1, 3, 6]},
        max_trials=3,
    )
    result = adapter.run(exp, {"top_k": 3}, records=records)
    p = result.payload
    print(f"\n  AutoRAG spec ({label}):")
    kv("  gt_mode", p["gt_mode"])
    kv("  objective_metric", p["objective_metric"])
    kv("  metrics", ", ".join(p["metrics"]))
    kv("  requires_gt", p["yaml_template"]["evaluation"]["requires_ground_truth"])
    return {
        "label": label,
        "gt_mode": p["gt_mode"],
        "objective_metric": p["objective_metric"],
        "metrics": list(p["metrics"]),
        "requires_gt": p["yaml_template"]["evaluation"]["requires_ground_truth"],
        "note": result.note,
        "yaml_template": p["yaml_template"],
    }


# ------------------------------ STEP 3 — pre-flight (descriptive)
def show_metric_filter(records, label: str) -> dict:
    requested = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_relevancy",
    ]
    kept, skipped = filter_metrics_by_data(requested, records)
    print(f"\n  Asking for the full ragas-style suite ({label}):")
    kv("  requested", requested)
    kv("  kept", kept)
    if skipped:
        kv("  skipped (reasons)", skipped)
    else:
        kv("  skipped", "none -- data supports every metric")
    return {"label": label, "requested": requested, "kept": kept, "skipped": skipped}


# --------------------- STEP 4 — AutoRAG REAL grid search (with eval)
def run_autorag_grid_search(corpus_chunks, base_records, top_ks, judge, ragas_embeddings, metrics) -> dict:
    """For each top_k: build RAG, generate answers, eval with ragas."""
    print("\n  Building shared embeddings + vector store ...")
    embeddings = build_embeddings()
    vstore = build_vector_store(corpus_chunks, embeddings)
    lm = build_litellm_callable()

    runs = []
    for top_k in top_ks:
        print(f"\n  [AutoRAG] top_k={top_k}")
        answered = []
        for i, r in enumerate(base_records):
            ctxs = retrieve(vstore, r.question, top_k)
            ans = generate_answer(r.question, ctxs, lm)
            answered.append(
                DatasetRecord(
                    question=r.question,
                    ground_truth=r.ground_truth,
                    contexts=ctxs,
                    answer=ans,
                )
            )
            print(f"    Q{i + 1}: retrieved {len(ctxs)} ctxs, answer={ans[:80]!r}")
        print("    evaluating with ragas ...")
        ev = evaluate_records_with_ragas(answered, judge, ragas_embeddings, metrics)
        if "error" in ev:
            print(f"      eval error: {ev['error']}")
        else:
            print(f"      scores: {ev['scores']}")
        runs.append(
            {
                "top_k": top_k,
                "scores": ev.get("scores", {}),
                "skipped": ev.get("skipped", {}),
                "answers": [a.answer for a in answered],
                "contexts": [a.contexts for a in answered],
            }
        )

    # Pick best by faithfulness (the universal ref-free target).
    best = max(runs, key=lambda r: r["scores"].get("faithfulness", -1.0))
    print(f"\n  -> Best config: top_k={best['top_k']} (faithfulness={best['scores'].get('faithfulness'):.3f})")
    return {
        "objective": "faithfulness",
        "runs": runs,
        "best_top_k": best["top_k"],
        "best_scores": best["scores"],
    }


# ------------------- STEP 5 — DSPy before/after using best top_k
def run_dspy_before_after(records_with_ctxs, judge, ragas_embeddings, metrics) -> dict:
    """Compare baseline RAG Signature vs MIPROv2-optimized variant on the
    SAME records (already retrieved with the best top_k from step 4).

    Runs three sub-passes:
      a) baseline:  default Signature -> generate -> ragas eval
      b) MIPROv2:   adapter.optimize(...) -- captures trial scores
      c) optimized: re-run with optimised program -> generate -> ragas eval
    """
    import dspy

    student = build_dspy_lm()
    dspy.configure(lm=student)

    adapter = DSPyAdapter()
    baseline_program = adapter.build_program()

    def _run_program(program):
        out = []
        for r in records_with_ctxs:
            ctx_str = "\n".join(r.contexts) if r.contexts else ""
            try:
                with dspy.context(lm=student):
                    pred = program(question=r.question, context=ctx_str)
                ans = str(getattr(pred, "answer", "") or "")
            except Exception as e:  # pragma: no cover
                ans = f"<error: {e}>"
            out.append(
                DatasetRecord(
                    question=r.question,
                    ground_truth=r.ground_truth,
                    contexts=r.contexts,
                    answer=ans,
                )
            )
        return out

    # (a) baseline
    print("\n  (a) Running BASELINE (default RAG Signature) ...")
    baseline_answered = _run_program(baseline_program)
    print("       evaluating baseline answers with ragas ...")
    baseline_eval = evaluate_records_with_ragas(baseline_answered, judge, ragas_embeddings, metrics)
    print(f"       baseline scores: {baseline_eval.get('scores')}")

    # (b) MIPROv2 optimisation, with trial-score capture
    print("\n  (b) Running MIPROv2 optimisation (with-GT path) ...")
    capture = _MIPROTrialScoreCapture()
    capture.setLevel(logging.INFO)
    mipro_logger = logging.getLogger("dspy.teleprompt.mipro_optimizer_v2")
    mipro_logger.addHandler(capture)
    prior_level = mipro_logger.level
    mipro_logger.setLevel(logging.INFO)
    try:
        opt_result = adapter.optimize(
            records_with_ctxs,
            student_lm=student,
            judge_lm=student,
            optimizer="MIPROv2",
            optimizer_kwargs={"auto": "light"},
        )
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior_level)

    optimised_program = opt_result["optimized_program"]
    print(f"       MIPROv2 done: {len(capture.scores_so_far)} trials, best={max(capture.best_scores or [0]):.2f}")

    # (c) re-run with optimised program
    print("\n  (c) Running OPTIMIZED program on same records ...")
    optimised_answered = _run_program(optimised_program)
    print("       evaluating optimised answers with ragas ...")
    opt_eval = evaluate_records_with_ragas(optimised_answered, judge, ragas_embeddings, metrics)
    print(f"       optimised scores: {opt_eval.get('scores')}")

    baseline_scores = baseline_eval.get("scores", {}) or {}
    optimised_scores = opt_eval.get("scores", {}) or {}
    all_metrics = sorted(set(baseline_scores) | set(optimised_scores))
    delta = {
        m: (optimised_scores.get(m, float("nan")) - baseline_scores.get(m, float("nan")))
        for m in all_metrics
    }
    print(f"\n  Delta (optimised - baseline): {delta}")

    serial_demos = {
        name: [dict(d) for d in demos]
        for name, demos in opt_result["demos"].items()
    }
    return {
        "baseline_scores": baseline_scores,
        "optimized_scores": optimised_scores,
        "delta": delta,
        "baseline_sample_answers": [a.answer for a in baseline_answered],
        "optimized_sample_answers": [a.answer for a in optimised_answered],
        "instructions": dict(opt_result["instructions"]),
        "demos": serial_demos,
        "trial_scores": list(capture.scores_so_far),
        "best_score_progression": list(capture.best_scores),
        "gt_mode": opt_result["gt_mode"],
        "optimizer": opt_result["optimizer"],
        "trainset_size": opt_result["trainset_size"],
    }


# ----------- STEP 6 — DSPy MIPROv2 only (descriptive, for GT-mode contrast)
def run_dspy_optimize_descriptive(records, label: str) -> dict:
    """Like the original demo cell -- runs MIPROv2 and captures trial
    scores but does NOT re-run the optimised program for ragas eval.
    Used so the dashboard can show the GT-mode metric_kind contrast
    (token-F1 vs LLM-judge) on the same trial chart layout as before."""
    print(f"\n  Running MIPROv2 ({label}) for trial-score contrast ...")
    import dspy

    student = build_dspy_lm()
    dspy.configure(lm=student)

    capture = _MIPROTrialScoreCapture()
    capture.setLevel(logging.INFO)
    mipro_logger = logging.getLogger("dspy.teleprompt.mipro_optimizer_v2")
    mipro_logger.addHandler(capture)
    prior_level = mipro_logger.level
    mipro_logger.setLevel(logging.INFO)

    adapter = DSPyAdapter()
    try:
        result = adapter.optimize(
            records,
            student_lm=student,
            judge_lm=student,
            optimizer="MIPROv2",
            optimizer_kwargs={"auto": "light"},
        )
    except Exception as exc:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior_level)
        return {
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
            "trial_scores": list(capture.scores_so_far),
            "best_score_progression": list(capture.best_scores),
        }
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior_level)

    serial_demos = {
        name: [dict(d) for d in demos]
        for name, demos in result["demos"].items()
    }
    return {
        "label": label,
        "gt_mode": result["gt_mode"],
        "optimizer": result["optimizer"],
        "trainset_size": result["trainset_size"],
        "instructions": dict(result["instructions"]),
        "demos": serial_demos,
        "trial_scores": list(capture.scores_so_far),
        "best_score_progression": list(capture.best_scores),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    print("Loading amnesty_qa[:5] corpus + records (may download on first run) ...")
    corpus_chunks, base_records = load_amnesty_corpus()
    print(f"  corpus chunks: {len(corpus_chunks)}; records: {len(base_records)}")

    records_gt = build_records(base_records, with_gt=True)
    records_no = build_records(base_records, with_gt=False)

    section("STEP 1 / 6 -- Corpus + data diagnostics")
    print("  Corpus: explodinggradients/amnesty_qa (english_v3, eval[:5])")
    print(f"  {len(base_records)} questions, {len(corpus_chunks)} corpus chunks total.")
    print("  Data diagnostics (with-GT):")
    kv("  has_ground_truth", has_ground_truth(records_gt))
    kv("  has_answers", has_answers(records_gt))
    kv("  has_contexts", has_contexts(records_gt))
    kv("  gt_mode", gt_mode(records_gt))
    print("\n  Data diagnostics (no-GT):")
    kv("  has_ground_truth", has_ground_truth(records_no))
    kv("  has_answers", has_answers(records_no))
    kv("  has_contexts", has_contexts(records_no))
    kv("  gt_mode", gt_mode(records_no))

    section("STEP 2 / 6 -- AutoRAG specs (descriptive, GT-mode contrast)")
    autorag_with = show_autorag(records_gt, "with-GT")
    autorag_no = show_autorag(records_no, "no-GT")

    section("STEP 3 / 6 -- Metric pre-flight (descriptive)")
    filter_with = show_metric_filter(records_gt, "with-GT")
    filter_no = show_metric_filter(records_no, "no-GT")

    judge = build_ragas_judge()
    ragas_embeddings = build_ragas_embeddings()
    metrics = build_ref_free_ragas_metrics()
    print(f"\n  Reference-free ragas metrics in play: {[m.name for m in metrics]}")

    section("STEP 4 / 6 -- AutoRAG REAL grid search (top_k in {1, 3, 6})")
    autorag_grid = run_autorag_grid_search(
        corpus_chunks, records_gt, top_ks=[1, 3, 6],
        judge=judge, ragas_embeddings=ragas_embeddings, metrics=metrics,
    )

    section("STEP 5 / 6 -- DSPy before/after at best top_k")
    best_top_k = autorag_grid["best_top_k"]
    print(f"  Using best_top_k={best_top_k} from step 4 ...")

    # Pre-retrieve contexts so DSPy program just sees (question, context).
    embeddings = build_embeddings()
    vstore = build_vector_store(corpus_chunks, embeddings)
    records_for_dspy = []
    for r in records_gt:
        ctxs = retrieve(vstore, r.question, best_top_k)
        records_for_dspy.append(
            DatasetRecord(
                question=r.question,
                ground_truth=r.ground_truth,
                contexts=ctxs,
            )
        )

    dspy_before_after = run_dspy_before_after(records_for_dspy, judge, ragas_embeddings, metrics)

    section("STEP 6 / 6 -- DSPy MIPROv2 trial chart (with-GT vs no-GT)")
    print("  Re-running MIPROv2 in both GT modes just to expose the")
    print("  metric_kind contrast (token-F1 vs LLM-as-judge).")
    dspy_with = run_dspy_optimize_descriptive(records_gt, "with-GT")
    dspy_no = run_dspy_optimize_descriptive(records_no, "no-GT")

    # Persist the structured artefact.
    bundle = {
        "model": "openai/glm-4-flash",
        "model_endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "dataset": "explodinggradients/amnesty_qa[:5]",
        "corpus_size": len(base_records),
        "corpus_chunks": len(corpus_chunks),
        "data_diagnostics": {
            "with_gt": {
                "has_ground_truth": has_ground_truth(records_gt),
                "has_answers": has_answers(records_gt),
                "has_contexts": has_contexts(records_gt),
                "gt_mode": gt_mode(records_gt),
            },
            "no_gt": {
                "has_ground_truth": has_ground_truth(records_no),
                "has_answers": has_answers(records_no),
                "has_contexts": has_contexts(records_no),
                "gt_mode": gt_mode(records_no),
            },
        },
        "autorag_spec": {"with_gt": autorag_with, "no_gt": autorag_no},
        "metric_filter": {"with_gt": filter_with, "no_gt": filter_no},
        "autorag_grid": autorag_grid,
        "dspy_before_after": dspy_before_after,
        "dspy_descriptive": {"with_gt": dspy_with, "no_gt": dspy_no},
        "questions": [
            {"question": r.question, "ground_truth": r.ground_truth}
            for r in base_records
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Structured result saved -> {OUTPUT_JSON}")

    section("SUMMARY")
    bs = dspy_before_after["baseline_scores"]
    os_ = dspy_before_after["optimized_scores"]
    print("  AutoRAG winner    :", f"top_k={best_top_k} (faithfulness={autorag_grid['best_scores'].get('faithfulness'):.3f})")
    print("  DSPy baseline     :", bs)
    print("  DSPy optimized    :", os_)
    print("  Delta             :", dspy_before_after["delta"])


if __name__ == "__main__":
    main()
