"""End-to-end demo: same corpus, two GT modes, both optimization paths.

Walks through the four cells of the optimization matrix that ragdx now
supports out of the box:

                       AutoRAG (RAG flow)      DSPy (prompt)
    with-GT     -->   reference-based         token-F1 metric
    no-GT       -->   reference-free          LLM-as-judge metric

The demo:

1. Builds a tiny corpus + 5 records (the same ones, with and without GT).
2. For each (tool x GT-mode) cell prints the rendered spec and what the
   adapter decided about metrics / objective / metric_kind.
3. (Optional) If ``dspy`` and an ``OPENAI_API_KEY`` are available, runs a
   light MIPROv2 compile on the no-GT path so you can see the optimised
   instruction and few-shot demos emerge -- purely informational, not
   required for the demo to finish.

This demo is intentionally offline-friendly: the parts that always run do
not call any LLM. Set ``RAGDX_RUN_DSPY=1`` to enable the DSPy compile cell.

Usage::

    PYTHONPATH=src python examples/demo_optimize_gt_modes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ragdx.optim._gt_helpers import (  # noqa: E402
    filter_metrics_by_data,
    gt_mode,
    has_answers,
    has_contexts,
    has_ground_truth,
)
from ragdx.optim.autorag_adapter import AutoRAGAdapter  # noqa: E402
from ragdx.optim.dspy_adapter import DSPyAdapter  # noqa: E402
from ragdx.schemas.models import (  # noqa: E402
    DatasetRecord,
    OptimizationExperiment,
)

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


def print_data_diagnostics(records):
    print("\n  Data diagnostics:")
    kv("  has_ground_truth", has_ground_truth(records))
    kv("  has_answers", has_answers(records))
    kv("  has_contexts", has_contexts(records))
    kv("  gt_mode (auto)", gt_mode(records))


# ------------------------------------------------------------- cells
def show_autorag(records, label: str) -> None:
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
    payload = result.payload
    print(f"\n  AutoRAG spec ({label}):")
    kv("  gt_mode", payload["gt_mode"])
    kv("  objective_metric", payload["objective_metric"])
    kv("  metrics", ", ".join(payload["metrics"]))
    kv("  requires_gt", payload["yaml_template"]["evaluation"]["requires_ground_truth"])
    print("  Adapter note:")
    print(f"    {result.note}")


def show_dspy_spec(records, label: str) -> None:
    adapter = DSPyAdapter()
    exp = OptimizationExperiment(
        name=f"dspy-{label}",
        tool="dspy",
        target_component="generation",
        description=f"DSPy prompt optimization in {label} mode",
        parameters={},
        objectives={"faithfulness": 1.0},
        search_space={},
        max_trials=10,
    )
    spec = adapter.build_optimizer_spec(
        exp,
        {"optimizer": "MIPROv2", "fewshot_count": 4},
        records=records,
    )
    print(f"\n  DSPy spec ({label}):")
    kv("  gt_mode", spec["gt_mode"])
    kv("  objective_metric", spec["objective_metric"])
    kv("  metric_kind", spec["compile_hints"]["metric_kind"])
    kv("  fewshot_enabled", spec["compile_hints"]["fewshot_enabled"])


def show_metric_filter_demo(records, label: str) -> None:
    """Concrete demonstration that bad metric requests get filtered out."""
    print(f"\n  Asking for the full ragas-style suite ({label}):")
    requested = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_relevancy",
    ]
    kept, skipped = filter_metrics_by_data(requested, records)
    kv("  requested", requested)
    kv("  kept", kept)
    if skipped:
        kv("  skipped (reasons)", skipped)
    else:
        kv("  skipped", "none -- data supports every metric")


def maybe_run_dspy_compile(records) -> None:
    """Optional cell: actually call MIPROv2 on the no-GT path.

    Only runs when ``RAGDX_RUN_DSPY=1`` and a usable LM is configured. Kept
    separate so the rest of the demo stays offline-friendly.
    """
    if os.environ.get("RAGDX_RUN_DSPY", "0") != "1":
        print("\n  [skipped] Set RAGDX_RUN_DSPY=1 to compile MIPROv2 on the no-GT path")
        return

    try:
        import dspy  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"\n  [skipped] dspy not importable: {exc}")
        return

    # Configure both the program LM and the judge LM. Prefer GLM-4-Flash if
    # ZHIPU_API_KEY is set (matches the rest of the project's examples).
    zhipu = os.environ.get("ZHIPU_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if zhipu:
        student = dspy.LM(
            "openai/glm-4-flash",
            api_key=zhipu,
            api_base="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.01,
        )
        judge = student
    elif openai_key:
        student = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        judge = student
    else:
        print("\n  [skipped] No ZHIPU_API_KEY / OPENAI_API_KEY in env")
        return

    print("\n  Running DSPyAdapter.optimize(no_gt) -- this calls the LM...")
    adapter = DSPyAdapter()
    try:
        result = adapter.optimize(
            records,
            student_lm=student,
            judge_lm=judge,
            optimizer="MIPROv2",
            optimizer_kwargs={"auto": "light"},
        )
    except Exception as exc:
        print(f"  optimize() raised: {exc!r}")
        return

    print("\n  Optimisation done.")
    kv("  gt_mode", result["gt_mode"])
    kv("  optimizer", result["optimizer"])
    kv("  trainset_size", result["trainset_size"])
    for name, instr in result["instructions"].items():
        print(f"\n  Instructions for predictor {name!r}:")
        print(f"    {instr}")
    for name, demos in result["demos"].items():
        print(f"\n  Selected {len(demos)} few-shot demo(s) for predictor {name!r}.")


# ---------------------------------------------------------------- main
def main() -> None:
    records_gt = build_records(with_gt=True)
    records_no = build_records(with_gt=False)

    section("STEP 1 / 4 -- Corpus")
    print(f"  {len(RAW_QA)} questions; same content under both branches.")
    print(f"  Each record has: question, answer, contexts ({len(RAW_QA[0]['contexts'])} per).")
    print("  ground_truth is populated only in the with-GT branch.")
    print_data_diagnostics(records_gt)
    print_data_diagnostics(records_no)

    section("STEP 2 / 4 -- AutoRAG specs (with-GT vs no-GT)")
    show_autorag(records_gt, "with-GT")
    show_autorag(records_no, "no-GT")

    section("STEP 3 / 4 -- DSPy specs (with-GT vs no-GT)")
    show_dspy_spec(records_gt, "with-GT")
    show_dspy_spec(records_no, "no-GT")

    section("STEP 4 / 4 -- Metric pre-flight on each branch")
    show_metric_filter_demo(records_gt, "with-GT")
    show_metric_filter_demo(records_no, "no-GT")

    section("BONUS -- DSPy compile on no-GT path (optional)")
    maybe_run_dspy_compile(records_no)

    section("SUMMARY")
    print("""
  AutoRAG:
    * with-GT   --> objective_metric = answer_correctness; metrics include
                 context_recall, context_entity_recall, recall, claim_recall
    * no-GT     --> objective_metric = faithfulness; reference-required metrics
                 are dropped; AutoRAG runtime must be wired up with an
                 LLM-as-judge to compute the reference-free ones

  DSPy:
    * with-GT   --> metric_kind = reference_based; default token-F1 over
                 example.answer; demos are real labelled examples
    * no-GT     --> metric_kind = reference_free_llm_judge; built-in
                 FaithfulnessJudge scores predictions against contexts;
                 trainset omits answer fields

  Evaluator data gating (now enforced):
    * ragas    -- ``_evaluate_in_process`` drops reference-required metrics
                 when GT is missing; result.metadata records skipped reasons
    * ragchecker -- same filtering, plus refuses to run at all without an
                 answer or contexts (would otherwise emit silent 0s)
    * embedding-proxy -- already well-behaved (no change)
""")


if __name__ == "__main__":
    main()
