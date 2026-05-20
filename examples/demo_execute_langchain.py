"""Real end-to-end ragdx `execute` mode demo via LangChain runner.

Unlike `demo_real_data.py`, this script does NOT call `simulate`. It:

    1. Loads the real amnesty_qa dataset.
    2. Hand-builds an OptimizationPlan with one LangChain experiment
       (so the executor must invoke the external subprocess runner).
    3. Writes the dataset as JSONL (the format the runner expects).
    4. Points ``RAGDX_LANGCHAIN_RUNNER_CMD`` at ``run_langchain_trial.py``.
    5. Calls ``executor.execute_plan(..., mode='execute')`` — which spawns
       the subprocess, hands it a YAML config + dataset, runs the demo
       LangChain RAG pipeline, scores with embedding-proxy, and writes
       metrics back into a JSON file the executor reads.
    6. Verifies trials are NOT marked ``simulated=True``.
    7. Persists run + session, prints summary.

No API keys required. The LangChain pipeline in
``examples/langchain_pipeline.py`` uses a hardcoded 2-document demo
corpus, so this is a *toy* runtime but a *real* subprocess flow.

Usage::

    PYTHONPATH=src python examples/demo_execute_langchain.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

# Isolate storage.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEMO_ROOT = REPO / ".ragdx_execute"
os.environ["RAGDX_ROOT"] = str(DEMO_ROOT)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# So the runner subprocess can find examples.langchain_pipeline.
os.environ["PYTHONPATH"] = (
    str(REPO / "src") + os.pathsep + str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from datasets import load_dataset  # noqa: E402

from ragdx import (  # noqa: E402
    OptimizationExecutor,
    RAGDiagnosisEngine,
    RunStore,
    UnifiedEvaluator,
)
from ragdx.schemas.models import (  # noqa: E402
    DatasetRecord,
    OptimizationExperiment,
    OptimizationPlan,
)


def section(t: str) -> None:
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    section("STEP 1 / 7  --  Load real amnesty_qa + build retrieval corpus")
    ds = load_dataset(
        "explodinggradients/amnesty_qa", "english_v3", split="eval[:5]"
    )
    records = [
        DatasetRecord(
            question=row["user_input"],
            ground_truth=row.get("reference") or "",
            answer=row.get("response") or "",
            contexts=list(row.get("retrieved_contexts") or []),
            # amnesty_qa doesn't ship gold retrieval contexts; use the
            # gold answer text as a reference passage for recall scoring.
            reference_contexts=[row.get("reference") or ""] if row.get("reference") else [],
            metadata={"source": "amnesty_qa"},
        )
        for row in ds
    ]
    print(f"Records          : {len(records)}")

    # Build a real retrieval corpus from all retrieved_contexts across
    # the 5 records (5 x 3 = ~15 chunks). The pipeline's FAISS index
    # will search this corpus at runtime.
    corpus_chunks = []
    for i, row in enumerate(ds):
        for j, passage in enumerate(row.get("retrieved_contexts") or []):
            text = (passage or "").strip()
            if not text:
                continue
            corpus_chunks.append({
                "text": text,
                "source": f"amnesty_qa#{i}-{j}",
            })
    corpus_path = DEMO_ROOT / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for c in corpus_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Corpus chunks    : {len(corpus_chunks)}")
    print(f"Wrote corpus     : {corpus_path}")

    # Write dataset JSONL for the runner. IMPORTANT: we deliberately do
    # NOT include a "contexts" key, so the runner uses the contexts the
    # pipeline actually retrieved at runtime (see run_langchain_trial.py:57).
    dataset_path = DEMO_ROOT / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "question": r.question,
                "ground_truth": r.ground_truth,
                "reference_contexts": r.reference_contexts,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote dataset    : {dataset_path}")

    # --------------------------------------------------------------------- #
    section("STEP 2 / 7  --  Baseline evaluation (embedding-proxy)")
    baseline = UnifiedEvaluator().evaluate(
        records, use_embedding=True, use_ragas=False, use_ragchecker=False
    )
    print("Baseline retrieval :", baseline.retrieval)
    print("Baseline generation:", baseline.generation)
    print("Baseline e2e       :", baseline.e2e)

    # --------------------------------------------------------------------- #
    section("STEP 3 / 7  --  Diagnose baseline")
    report = RAGDiagnosisEngine().diagnose(baseline)
    print(f"Summary: {report.summary}")
    print(f"Confidence: {report.diagnosis_confidence:.2f}")

    # --------------------------------------------------------------------- #
    section("STEP 4 / 7  --  Hand-build OptimizationPlan w/ LangChain runner")
    # Build a single retrieval-stage experiment that targets the langchain tool.
    # The Bayesian executor will explore retriever_k.
    # NOTE on parameter names:
    #   - The langchain adapter writes `runtime.retriever_k = parameters["top_k"]`
    #     (see langchain_adapter.py:63). So we must search on `top_k`, NOT
    #     `retriever_k`, for the value to actually reach the pipeline.
    #   - `corpus_path` is a single-value space; it just rides along in
    #     `search_parameters` so the pipeline factory can find it.
    corpus_path_str = str(corpus_path).replace("\\", "/")
    experiment = OptimizationExperiment(
        name="langchain-faiss-topk-search",
        tool="langchain",
        target_component="retrieval",
        description="Real subprocess trial via LangChain+FAISS — vary top_k.",
        parameters={"top_k": 3},
        objectives={
            "answer_correctness": 1.0,
            "faithfulness": 0.7,
            "context_precision": 0.5,
        },
        search_space={
            "top_k": [1, 3, 6],
            "corpus_path": [corpus_path_str],
        },
        search_strategy="bayesian",
        max_trials=3,
        stage="retrieval",
    )
    plan = OptimizationPlan(
        objective_metric="answer_correctness",
        experiments=[experiment],
        rationale=["Demo: exercise the real subprocess runner path with no API keys."],
    )
    print(f"Plan experiments : {len(plan.experiments)}")
    print(f"Tool             : {experiment.tool}")
    print(f"Max trials       : {experiment.max_trials}")
    print(f"Search space     : {experiment.search_space}")

    # --------------------------------------------------------------------- #
    section("STEP 5 / 7  --  Wire program_contract + RAGDX_LANGCHAIN_RUNNER_CMD")
    # The LangChain adapter reads ``dataset_path`` and ``pipeline_module``
    # directly from ``experiment.parameters`` (see langchain_adapter.py:54)
    # to populate the trial YAML's ``program_contract`` block. So we must
    # set them at the top level, not nested.
    experiment.parameters["dataset_path"] = str(dataset_path).replace("\\", "/")
    experiment.parameters["pipeline_module"] = "examples.langchain_pipeline_real:create_pipeline"

    # Point the executor at our runner script.
    # NOTE: shlex.split() (used by the executor) treats backslashes as
    # escape characters in POSIX mode, which corrupts Windows paths. We
    # convert to forward slashes — Windows Python tolerates them in
    # all stdlib path APIs.
    py = sys.executable.replace("\\", "/")
    runner_script = str(HERE / "run_langchain_trial.py").replace("\\", "/")
    runner_cmd = f'"{py}" "{runner_script}" --config {{config}} --output {{output}}'
    os.environ["RAGDX_LANGCHAIN_RUNNER_CMD"] = runner_cmd
    # Strict execute: refuse to silently simulate if runner is missing.
    os.environ["RAGDX_STRICT_EXECUTE"] = "1"
    print(f"Runner cmd       : {runner_cmd}")

    # --------------------------------------------------------------------- #
    section("STEP 6 / 7  --  EXECUTE plan (real subprocess, no simulation)")
    executor = OptimizationExecutor(strict_execute=True)
    session = executor.execute_plan(
        plan, baseline=baseline, strategy="bayesian", mode="execute"
    )
    print(f"Session ID       : {session.session_id}")
    print(f"Status           : {session.status}")
    print(f"Trials completed : {session.completed_trials} / {session.total_trials}")
    print(f"Best trial id    : {session.best_trial_id}")

    if not session.trials:
        print("NO TRIALS — investigate logs.")
        return

    print("\n-- per-trial detail --")
    for t in session.trials:
        print(
            f"  trial={t.trial_id}  status={t.status}  simulated={t.simulated}  "
            f"rc={t.return_code}  utility={t.utility}"
        )
        if t.runner_command:
            print(f"    cmd:    {t.runner_command}")
        if t.output_path and Path(t.output_path).exists():
            print(f"    output: {t.output_path}")
        if t.objective_scores:
            scored = {k: round(v, 4) for k, v in t.objective_scores.items()}
            print(f"    scores: {scored}")

    real_count = sum(1 for t in session.trials if not t.simulated and t.status == "done")
    print(f"\nReal (non-simulated) successful trials: {real_count} / {len(session.trials)}")

    # --------------------------------------------------------------------- #
    section("STEP 7 / 7  --  Persist & inspect artifacts")
    store = RunStore(root=str(DEMO_ROOT))
    saved = store.save_run(
        baseline, report, plan, name="execute-langchain-real", tags=["execute", "real-runner"]
    )
    store.save_session(session)
    store.update_run_latest_session(saved.run_id, session.session_id)
    print(f"Saved run      : {saved.run_id}")
    print(f"Saved session  : {session.session_id}")
    print(f"Trial outputs  : {DEMO_ROOT}/optimization/{session.session_id}/outputs/")
    print(f"Storage root   : {DEMO_ROOT}")


if __name__ == "__main__":
    main()
