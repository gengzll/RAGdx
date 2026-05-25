"""End-to-end demo: same corpus, two GT modes, real LLM calls.

Walks through the four cells of the optimization matrix (tool x GT mode):

                       AutoRAG (RAG flow)      DSPy (prompt)
    with-GT     -->   reference-based         token-F1 metric
    no-GT       -->   reference-free          LLM-as-judge metric

What this demo actually does:

1. Loads a 5-record corpus with ground-truths.
2. Runs a real ragas evaluation in BOTH branches (with-GT and the same
   records with GT erased) to demonstrate that the adapter now drops
   reference-required metrics instead of silently emitting 0s.
3. Runs DSPyAdapter.optimize() in BOTH branches with MIPROv2 -- with-GT
   uses token-F1 against example.answer; no-GT uses the built-in
   FaithfulnessJudge LLM-as-judge. The optimised instructions + demos
   are printed at the end.

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

# Configure OpenAI-compatible env vars so LiteLLM / DSPy route to Zhipu.
os.environ["OPENAI_API_KEY"] = KEY
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    ``Best score so far: 28.6`` — we parse those out instead of patching
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

# ---------------------------------------------------------------- corpus
RAW_QA = [
    {
        "question": "What is RAG?",
        "answer": "Retrieval-augmented generation combines retrieval with generation.",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) augments a language model by "
            "retrieving relevant documents and conditioning the generation on them."
        ),
        "contexts": [
            "RAG augments LLMs by retrieving documents before generation.",
            "Generation is conditioned on the retrieved passages.",
        ],
    },
    {
        "question": "Why use chunking in RAG?",
        "answer": "Chunking splits documents so retrieval can return precise passages.",
        "ground_truth": (
            "Documents are split into smaller chunks so the retriever can return "
            "tightly-scoped passages instead of whole documents."
        ),
        "contexts": [
            "Chunking trades off chunk size against recall.",
            "Smaller chunks improve precision but may fragment context.",
        ],
    },
    {
        "question": "What does an embedding model do?",
        "answer": "It encodes text into a vector space for similarity search.",
        "ground_truth": (
            "An embedding model maps text into a dense vector representation so "
            "that semantically similar text has nearby vectors."
        ),
        "contexts": [
            "Sentence embeddings enable nearest-neighbour search.",
            "Cosine similarity in embedding space approximates semantic similarity.",
        ],
    },
    {
        "question": "How is a reranker different from a retriever?",
        "answer": "A reranker re-scores the retriever's top-k results.",
        "ground_truth": (
            "Retrievers do a fast first-pass over the corpus; rerankers do a "
            "slower but more accurate re-scoring of the retriever's top-k."
        ),
        "contexts": [
            "Cross-encoder rerankers are expensive but accurate.",
            "Bi-encoder retrievers are cheap and approximate.",
        ],
    },
    {
        "question": "Name one cause of hallucination in RAG.",
        "answer": "Retrieved context not actually supporting the answer.",
        "ground_truth": (
            "Common causes include irrelevant retrieved chunks, missing key "
            "information, and the model preferring its parametric knowledge."
        ),
        "contexts": [
            "Hallucination rises when retrieval recall is low.",
            "Models may ignore context and rely on their pretraining.",
        ],
    },
]


def build_records(*, with_gt: bool) -> list[DatasetRecord]:
    return [
        DatasetRecord(
            question=row["question"],
            answer=row["answer"],
            ground_truth=row["ground_truth"] if with_gt else None,
            contexts=list(row["contexts"]),
        )
        for row in RAW_QA
    ]


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kv(label: str, value: object, width: int = 22) -> None:
    print(f"  {label:<{width}s} {value}")


# ------------------------------------------------------- LLM construction
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


# ----------------------------------------------------------- AutoRAG cell
def show_autorag(records, label: str) -> dict:
    adapter = AutoRAGAdapter()
    exp = OptimizationExperiment(
        name=f"autorag-{label}",
        tool="autorag",
        target_component="pipeline",
        description=f"AutoRAG search in {label} mode",
        parameters={},
        objectives={"faithfulness": 1.0},
        search_space={
            "chunk_size": [256, 512, 1024],
            "top_k": [1, 3, 6],
            "reranker": ["none", "bge"],
        },
        max_trials=10,
    )
    result = adapter.run(
        exp,
        {"chunk_size": 512, "top_k": 3, "reranker": "bge"},
        records=records,
    )
    p = result.payload
    print(f"\n  AutoRAG spec ({label}):")
    kv("  gt_mode", p["gt_mode"])
    kv("  objective_metric", p["objective_metric"])
    kv("  metrics", ", ".join(p["metrics"]))
    kv("  requires_gt", p["yaml_template"]["evaluation"]["requires_ground_truth"])
    print(f"  Note: {result.note}")
    return {
        "label": label,
        "gt_mode": p["gt_mode"],
        "objective_metric": p["objective_metric"],
        "metrics": list(p["metrics"]),
        "requires_gt": p["yaml_template"]["evaluation"]["requires_ground_truth"],
        "note": result.note,
        "yaml_template": p["yaml_template"],
    }


# ----------------------------------------------------- Pre-flight cell
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


# --------------------------------------------- DSPy MIPROv2 compile cell
def run_dspy_optimize(records, label: str) -> dict:
    """Compile a tiny RAG predictor with MIPROv2 and print the result."""
    print(f"\n  Running DSPyAdapter.optimize({label}) -- this calls GLM-4-Flash...")
    import dspy

    student = build_dspy_lm()
    dspy.configure(lm=student)

    # Capture the per-trial score progression so the dashboard can plot it.
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
            judge_lm=student,  # same LM serves as judge in no-GT mode
            optimizer="MIPROv2",
            optimizer_kwargs={"auto": "light"},
        )
    except Exception as exc:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior_level)
        print(f"  optimize() raised: {type(exc).__name__}: {exc}")
        return {
            "label": label,
            "error": f"{type(exc).__name__}: {exc}",
            "trial_scores": list(capture.scores_so_far),
            "best_score_progression": list(capture.best_scores),
        }
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior_level)

    print("\n  Optimisation done.")
    kv("  gt_mode", result["gt_mode"])
    kv("  optimizer", result["optimizer"])
    kv("  trainset_size", result["trainset_size"])
    kv("  total trials", len(capture.scores_so_far))
    if capture.best_scores:
        kv("  best score", capture.best_scores[-1])
    for name, instr in result["instructions"].items():
        print(f"\n  Instructions for predictor {name!r}:")
        for line in (instr or "").splitlines() or [""]:
            print(f"    {line}")
    for name, demos in result["demos"].items():
        print(f"\n  Selected {len(demos)} few-shot demo(s) for predictor {name!r}.")
        for i, d in enumerate(demos[:2]):  # cap printing to first 2
            print(f"    demo[{i}]: {dict(d)!s:.220s}")

    # Serialise demos so they survive JSON dump (drop the dspy.Example
    # wrapper, keep the underlying input/output dict).
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


# ----------------------------------------- Actual evaluation step
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


def run_evaluation(records, label: str) -> dict:
    """Run UnifiedEvaluator twice on the same records: once with the
    dependency-free embedding-proxy (fast deterministic baseline), once
    with real ragas (GLM-as-judge). Returns per-bucket scores for the
    dashboard."""
    print(f"\n  Running UnifiedEvaluator({label}) ...")
    evaluator = UnifiedEvaluator()

    # Cell A: embedding-proxy (instant, no API)
    print("    [A] embedding-proxy ...")
    proxy = evaluator.evaluate(
        records,
        use_ragas=False, use_ragchecker=False, use_embedding=True,
    )
    proxy_pack = {
        "retrieval": dict(proxy.retrieval),
        "generation": dict(proxy.generation),
        "e2e": dict(proxy.e2e),
        "metadata": dict(proxy.metadata),
    }
    print(f"        retrieval={proxy_pack['retrieval']}")
    print(f"        generation={proxy_pack['generation']}")
    print(f"        e2e={proxy_pack['e2e']}")

    # Cell B: ragas with GLM judge (real eval calls)
    print("    [B] ragas (GLM-4-Flash judge) ...")
    ragas_judge = build_ragas_judge()
    try:
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        requested_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ]
        real = evaluator.evaluate(
            records,
            use_ragas=True, run_ragas=True,
            use_ragchecker=False, use_embedding=False,
            ragas_kwargs={
                "llm": ragas_judge,
                "metrics": requested_metrics,
            },
        )
        real_pack = {
            "retrieval": dict(real.retrieval),
            "generation": dict(real.generation),
            "e2e": dict(real.e2e),
            "metadata": {
                k: v for k, v in real.metadata.items()
                if k in ("ragas_metrics", "skipped_metrics", "data_diagnostics", "mode")
            },
        }
        print(f"        retrieval={real_pack['retrieval']}")
        print(f"        generation={real_pack['generation']}")
        print(f"        e2e={real_pack['e2e']}")
        print(f"        skipped={real_pack['metadata'].get('skipped_metrics', {})}")
    except Exception as exc:
        print(f"    ragas eval failed: {type(exc).__name__}: {exc}")
        real_pack = {"error": f"{type(exc).__name__}: {exc}"}

    return {"label": label, "embedding_proxy": proxy_pack, "ragas": real_pack}


# ---------------------------------------------------------------- main
def main() -> None:
    records_gt = build_records(with_gt=True)
    records_no = build_records(with_gt=False)

    section("STEP 1 / 4 -- Corpus")
    print(f"  {len(RAW_QA)} questions, identical content in both branches.")
    print("  Data diagnostics with GT:")
    kv("  has_ground_truth", has_ground_truth(records_gt))
    kv("  has_answers", has_answers(records_gt))
    kv("  has_contexts", has_contexts(records_gt))
    kv("  gt_mode", gt_mode(records_gt))
    print("\n  Data diagnostics without GT:")
    kv("  has_ground_truth", has_ground_truth(records_no))
    kv("  has_answers", has_answers(records_no))
    kv("  has_contexts", has_contexts(records_no))
    kv("  gt_mode", gt_mode(records_no))

    section("STEP 2 / 4 -- AutoRAG specs (with-GT vs no-GT)")
    autorag_with = show_autorag(records_gt, "with-GT")
    autorag_no = show_autorag(records_no, "no-GT")

    section("STEP 3 / 4 -- Metric pre-flight on each branch")
    filter_with = show_metric_filter(records_gt, "with-GT")
    filter_no = show_metric_filter(records_no, "no-GT")

    section("STEP 4 / 5 -- Actual evaluation (UnifiedEvaluator: embedding-proxy + ragas)")
    eval_with = run_evaluation(records_gt, "with-GT")
    eval_no = run_evaluation(records_no, "no-GT")

    section("STEP 5 / 5 -- DSPy MIPROv2 optimize (real LLM calls)")
    print("  LLM: GLM-4-Flash via Zhipu (openai-compatible endpoint)")
    print("  Optimizer: MIPROv2 (auto='light')")
    dspy_with = run_dspy_optimize(records_gt, "with-GT")
    dspy_no = run_dspy_optimize(records_no, "no-GT")

    # Persist a structured artefact for the Streamlit dashboard.
    bundle = {
        "model": "openai/glm-4-flash",
        "model_endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "corpus": RAW_QA,
        "corpus_size": len(RAW_QA),
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
        "autorag": {"with_gt": autorag_with, "no_gt": autorag_no},
        "metric_filter": {"with_gt": filter_with, "no_gt": filter_no},
        "evaluation": {"with_gt": eval_with, "no_gt": eval_no},
        "dspy": {"with_gt": dspy_with, "no_gt": dspy_no},
    }
    OUTPUT_JSON.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Structured result saved -> {OUTPUT_JSON}")

    section("SUMMARY")
    print("""
  AutoRAG:
    * with-GT  -> objective=answer_correctness; reference-based metrics
                  (context_recall, context_entity_recall, recall,
                  claim_recall, context_utilization, answer_correctness,
                  answer_accuracy) are included
    * no-GT    -> objective=faithfulness; reference-required metrics
                  removed; AutoRAG runtime must provide an LLM-as-judge
                  for the surviving reference-free ones

  DSPy:
    * with-GT  -> metric = token-F1 over example.answer (offline, no
                  per-trial LLM cost for the metric itself); demos are
                  bootstrapped from the labelled trainset
    * no-GT    -> metric = built-in FaithfulnessJudge dspy.Signature;
                  every metric call is an LLM-as-judge invocation;
                  trainset has no answer field

  Adapter data gating (now enforced):
    * ragas       -> _evaluate_in_process drops reference-required metrics
                     and records skipped_metrics in result.metadata
    * ragchecker  -> raises RuntimeError when answers or contexts are
                     missing; filters metric_names against data
    * embedding   -> already well-behaved (each metric only computed
                     when its inputs exist)
""")


if __name__ == "__main__":
    main()
