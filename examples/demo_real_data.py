"""End-to-end ragdx demo against a real public RAG dataset.

Loads ``explodinggradients/amnesty_qa`` (the canonical Ragas demo
dataset: 20 questions about USA Supreme Court / global abortion rulings,
with retrieved contexts, generated responses, and reference answers),
maps it into ``DatasetRecord`` objects, and runs the full ragdx loop:

    1. Load real RAG eval data from HuggingFace.
    2. Evaluate with the embedding-proxy evaluator (no API key needed).
       — If ``OPENAI_API_KEY`` is set, Ragas is also invoked for cross-check.
    3. Diagnose with the rule-based + causal Bayes engine.
    4. Plan optimization experiments.
    5. Execute the plan in ``simulate`` mode.
    6. Persist run + session to an isolated RunStore at ``.ragdx_real/``.
    7. Attach a synthetic FeedbackEvent.
    8. Print a summary report.

Usage::

    PYTHONPATH=src python examples/demo_real_data.py

Optional env vars::

    OPENAI_API_KEY    enable Ragas evaluator alongside embedding-proxy
    RAGDX_DATASET_N   override the number of records to load (default 20)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Isolate storage so this demo does not collide with your real RunStore.
DEMO_ROOT = Path(__file__).resolve().parent.parent / ".ragdx_real"
os.environ.setdefault("RAGDX_ROOT", str(DEMO_ROOT))

# Quieter HF download UX on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from datasets import load_dataset  # noqa: E402

from ragdx import (  # noqa: E402
    OptimizationExecutor,
    OptimizationPlanner,
    RAGDiagnosisEngine,
    RunStore,
    UnifiedEvaluator,
)
from ragdx.schemas.models import DatasetRecord, FeedbackEvent


N_RECORDS = int(os.environ.get("RAGDX_DATASET_N", "20"))


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def load_real_dataset() -> list[DatasetRecord]:
    """Load amnesty_qa from HuggingFace and convert to DatasetRecord."""
    ds = load_dataset(
        "explodinggradients/amnesty_qa", "english_v3", split=f"eval[:{N_RECORDS}]"
    )
    records: list[DatasetRecord] = []
    for row in ds:
        contexts = row.get("retrieved_contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        records.append(
            DatasetRecord(
                question=row["user_input"],
                ground_truth=row.get("reference") or "",
                answer=row.get("response") or "",
                contexts=list(contexts),
                reference_contexts=[row.get("reference") or ""] if row.get("reference") else [],
                metadata={"source": "explodinggradients/amnesty_qa"},
            )
        )
    return records


def main() -> None:
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)

    # --------------------------------------------------------------------- #
    section("STEP 1 / 7  --  Load real public RAG dataset")
    records = load_real_dataset()
    print(f"Dataset          : explodinggradients/amnesty_qa (english_v3, eval split)")
    print(f"Records          : {len(records)}")
    print(f"Sample question  : {records[0].question[:80]}...")
    print(f"Sample contexts  : {len(records[0].contexts)} retrieved passages")
    print(f"Avg answer len   : {sum(len(r.answer) for r in records) // len(records)} chars")

    # --------------------------------------------------------------------- #
    section("STEP 2 / 7  --  Evaluate with embedding-proxy")
    evaluator = UnifiedEvaluator()
    # Try Ragas if API key is present; otherwise just embedding-proxy.
    use_ragas = bool(os.environ.get("OPENAI_API_KEY"))
    result = evaluator.evaluate(
        records,
        use_embedding=True,
        use_ragas=use_ragas,
        use_ragchecker=False,  # ragchecker also needs LLM keys; skip by default
        run_ragas=use_ragas,
    )
    print(f"Evaluator mode   : {result.metadata.get('mode')}")
    if use_ragas:
        print("Ragas            : ENABLED (OPENAI_API_KEY detected)")
    print("Retrieval        :", result.retrieval)
    print("Generation       :", result.generation)
    print("E2E              :", result.e2e)

    # --------------------------------------------------------------------- #
    section("STEP 3 / 7  --  Diagnose (rule-based + causal Bayes)")
    report = RAGDiagnosisEngine().diagnose(result)
    print(f"Summary          : {report.summary}")
    print(f"Confidence       : {report.diagnosis_confidence:.2f}")
    print(f"# hypotheses     : {len(report.hypotheses)}")
    for h in report.hypotheses[:5]:
        print(f"  - [{h.severity:8s}] {h.component:10s} -> {h.root_cause}")
    if report.causal_signals:
        print("Top causal nodes :")
        for s in report.causal_signals[:3]:
            print(f"  - {s.node:35s} posterior={s.posterior:.2f}")
    print("Priority actions :")
    for a in report.priority_actions[:4]:
        print(f"  * {a}")

    # --------------------------------------------------------------------- #
    section("STEP 4 / 7  --  Plan optimization experiments")
    plan = OptimizationPlanner().build_plan(
        report, result=result, strategy="bayesian", budget=12
    )
    print(f"Objective metric : {plan.objective_metric}")
    print(f"Experiments      : {len(plan.experiments)}")
    for exp in plan.experiments:
        print(f"  - [{exp.stage:13s}] {exp.name}  ->  tool={exp.tool}")
        print(f"      {exp.description}")
    print("Rationale        :")
    for r in plan.rationale[:5]:
        print(f"  * {r}")

    # --------------------------------------------------------------------- #
    section("STEP 5 / 7  --  Execute plan in `simulate` mode")
    executor = OptimizationExecutor(strict_execute=False)
    session = executor.execute_plan(
        plan, baseline=result, strategy="bayesian", mode="simulate"
    )
    print(f"Session ID       : {session.session_id}")
    print(f"Status           : {session.status}")
    print(f"Trials           : {session.completed_trials} / {session.total_trials}")
    print(f"Best trial id    : {session.best_trial_id}")
    if session.trials:
        best = next((t for t in session.trials if t.trial_id == session.best_trial_id), None)
        if best:
            print(f"Best objectives  : {best.objective_scores}")
            print(f"Utility          : {best.utility:.4f}" if best.utility is not None else "Utility          : -")
            print(f"Simulated flag   : {best.simulated}")

    # --------------------------------------------------------------------- #
    section("STEP 6 / 7  --  Persist run + session")
    store = RunStore(root=str(DEMO_ROOT))
    saved = store.save_run(
        result,
        report,
        plan,
        name="amnesty_qa-real-data-demo",
        tags=["real-data", "amnesty_qa", "huggingface"],
        notes=f"Source: explodinggradients/amnesty_qa, {len(records)} records, eval split.",
    )
    store.save_session(session)
    store.update_run_latest_session(saved.run_id, session.session_id)
    print(f"Saved run        : {saved.run_id}   schema_version={saved.schema_version}")
    print(f"Saved session    : {session.session_id}")
    print(f"Storage root     : {DEMO_ROOT}")

    # --------------------------------------------------------------------- #
    section("STEP 7 / 7  --  Attach feedback (closing the loop)")
    fb = FeedbackEvent(
        feedback_id="fb-real-001",
        kind="thumbs_down",
        severity="medium",
        rating=0.2,
        note=(
            "Generated answer was on-topic but missed several globally-relevant "
            "consequences cited by reference. Suggests retrieval recall issue."
        ),
        metadata={"dataset": "amnesty_qa", "sample_index": 0},
    )
    updated = store.attach_feedback_to_run(saved.run_id, [fb])
    print(f"Feedback events  : {len(updated.evaluation.feedback_events)}")
    print(f"Aggregate summary: {store.feedback_summary()}")

    # --------------------------------------------------------------------- #
    section("DONE")
    print(
        "Inspect artifacts at:\n"
        f"  {DEMO_ROOT / 'runs' / (saved.run_id + '.json')}\n"
        f"  {DEMO_ROOT / 'optimization' / 'sessions' / (session.session_id + '.json')}\n"
    )
    print(
        "View in Streamlit dashboard:\n"
        f"  set RAGDX_ROOT={DEMO_ROOT}\n"
        f"  PYTHONPATH=src python -m streamlit run src/ragdx/ui/dashboard.py"
    )


if __name__ == "__main__":
    main()
