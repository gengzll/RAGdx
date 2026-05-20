"""End-to-end ragdx pipeline demo.

Runs the full ragdx flow against a small synthetic dataset, with no
external services required:

    1. Build a DatasetRecord corpus (5 Q/A pairs with retrieved contexts).
    2. Evaluate with the built-in embedding-proxy (TF-IDF + cosine).
    3. Diagnose with the rule-based + causal engine.
    4. Plan optimization experiments (Bayesian strategy, 8-trial budget).
    5. Execute the plan in `simulate` mode (no external runner needed).
    6. Persist the run + session to a temporary .ragdx/ root.
    7. Attach two FeedbackEvent records to the saved run.
    8. Print a compact summary so the result is human-readable.

Run with::

    PYTHONPATH=src python examples/demo_full_pipeline.py

The script writes everything into ``./.ragdx_demo/`` so it does not
collide with your real RunStore. Delete that folder when you're done.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 on Windows consoles so the demo can print ASCII-only output
# safely; some Windows shells default to GBK which can choke on stray
# non-ASCII bytes from underlying libraries.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Use an isolated RunStore so this demo doesn't pollute the user's .ragdx/.
DEMO_ROOT = Path(__file__).resolve().parent.parent / ".ragdx_demo"
os.environ["RAGDX_ROOT"] = str(DEMO_ROOT)

from ragdx import (  # noqa: E402  — env var must be set before import
    OptimizationExecutor,
    OptimizationPlanner,
    RAGDiagnosisEngine,
    RunStore,
    UnifiedEvaluator,
)
from ragdx.schemas.models import DatasetRecord, FeedbackEvent


# --------------------------------------------------------------------------- #
# 1. Dataset                                                                  #
# --------------------------------------------------------------------------- #
# A deliberately mixed-quality dataset: half the answers are well-grounded,
# half are partially off-topic or under-cited, so the diagnosis surfaces
# real retrieval/generation gaps rather than a single uniform symptom.
DATASET = [
    DatasetRecord(
        question="What is the capital of France?",
        ground_truth="Paris is the capital of France.",
        answer="Paris is the capital of France.",
        contexts=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
        reference_contexts=["Paris is the capital of France."],
    ),
    DatasetRecord(
        question="Who wrote the novel '1984'?",
        ground_truth="George Orwell wrote 1984.",
        answer="The novel 1984 was written by George Orwell in 1949.",
        contexts=[
            "1984 is a dystopian novel written by George Orwell, published in 1949.",
            "Orwell is also known for Animal Farm.",
        ],
        reference_contexts=["George Orwell wrote 1984 in 1949."],
    ),
    DatasetRecord(
        question="What is the boiling point of water at sea level?",
        ground_truth="Water boils at 100 degrees Celsius at sea level.",
        # Slightly wrong answer + weak context => should hurt faithfulness.
        answer="Water boils at around 90 degrees at high altitude.",
        contexts=[
            "Atmospheric pressure decreases with altitude.",
            "Cooking instructions often vary by elevation.",
        ],
        reference_contexts=["At sea level water boils at 100 C."],
    ),
    DatasetRecord(
        question="When did humans first land on the Moon?",
        ground_truth="Humans first landed on the Moon on July 20, 1969.",
        answer="Apollo 11 landed on the Moon on July 20, 1969.",
        contexts=[
            "The Apollo 11 mission successfully landed astronauts on the Moon on July 20, 1969.",
            "Neil Armstrong was the first person to walk on the lunar surface.",
        ],
        reference_contexts=["Apollo 11 Moon landing happened on July 20, 1969."],
    ),
    DatasetRecord(
        question="Explain how photosynthesis works.",
        ground_truth=(
            "Photosynthesis is the process by which plants use sunlight, water, "
            "and carbon dioxide to produce oxygen and glucose."
        ),
        # Off-topic context + over-confident answer => low context_precision.
        answer="Plants make food using sunlight, and they release oxygen as a byproduct.",
        contexts=[
            "Plants grow best in well-drained soil.",
            "Many crops require regular watering during the growing season.",
        ],
        reference_contexts=[
            "Photosynthesis converts CO2 and water into glucose and O2 using light energy.",
        ],
    ),
]


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    # Start clean so reruns are deterministic.
    if DEMO_ROOT.exists():
        shutil.rmtree(DEMO_ROOT)

    # --------------------------------------------------------------------- #
    # 2. Evaluate                                                           #
    # --------------------------------------------------------------------- #
    section("STEP 1 / 6  --  Evaluate dataset (embedding-proxy)")
    evaluator = UnifiedEvaluator()
    result = evaluator.evaluate(
        DATASET, use_embedding=True, use_ragas=False, use_ragchecker=False
    )
    print(f"Records evaluated : {len(DATASET)}")
    print(f"Retrieval metrics : {result.retrieval}")
    print(f"Generation metrics: {result.generation}")
    print(f"E2E metrics       : {result.e2e}")
    print(f"Mode              : {result.metadata.get('mode')}")

    # --------------------------------------------------------------------- #
    # 3. Diagnose                                                           #
    # --------------------------------------------------------------------- #
    section("STEP 2 / 6  --  Diagnose (rule-based + causal Bayes)")
    engine = RAGDiagnosisEngine()
    report = engine.diagnose(result)
    print(f"Summary          : {report.summary}")
    print(f"Confidence       : {report.diagnosis_confidence:.2f}")
    print(f"# hypotheses     : {len(report.hypotheses)}")
    for h in report.hypotheses[:3]:
        print(f"  - [{h.severity}] {h.component}: {h.root_cause}")
    if report.causal_signals:
        top = report.causal_signals[0]
        print(f"Top causal node  : {top.node}  posterior={top.posterior:.2f}")
    print("Priority actions :")
    for action in report.priority_actions[:4]:
        print(f"  * {action}")

    # --------------------------------------------------------------------- #
    # 4. Plan                                                               #
    # --------------------------------------------------------------------- #
    section("STEP 3 / 6  --  Plan optimization experiments")
    plan = OptimizationPlanner().build_plan(
        report, result=result, strategy="bayesian", budget=8
    )
    print(f"Objective metric : {plan.objective_metric}")
    print(f"Experiments      : {len(plan.experiments)}")
    for exp in plan.experiments:
        print(f"  - [{exp.stage:13s}] {exp.name}  ->  tool={exp.tool}")
        print(f"      {exp.description}")

    # --------------------------------------------------------------------- #
    # 5. Execute (simulate)                                                 #
    # --------------------------------------------------------------------- #
    section("STEP 4 / 6  --  Execute plan in `simulate` mode")
    executor = OptimizationExecutor(strict_execute=False)
    session = executor.execute_plan(
        plan, baseline=result, strategy="bayesian", mode="simulate"
    )
    print(f"Session ID       : {session.session_id}")
    print(f"Status           : {session.status}")
    print(f"Trials completed : {session.completed_trials} / {session.total_trials}")
    print(f"Best trial id    : {session.best_trial_id}")
    if session.trials:
        sample = session.trials[0]
        print(
            f"Example trial    : id={sample.trial_id}  "
            f"status={sample.status}  simulated={sample.simulated}"
        )

    # --------------------------------------------------------------------- #
    # 6. Persist                                                            #
    # --------------------------------------------------------------------- #
    section("STEP 5 / 6  --  Persist run + session to RunStore")
    store = RunStore(root=str(DEMO_ROOT))
    saved = store.save_run(
        result, report, plan, name="full-pipeline-demo", tags=["demo", "end-to-end"]
    )
    store.save_session(session)
    store.update_run_latest_session(saved.run_id, session.session_id)
    print(f"Saved run        : {saved.run_id}   schema_version={saved.schema_version}")
    print(f"Saved session    : {session.session_id}")
    print(f"Storage root     : {DEMO_ROOT}")

    # --------------------------------------------------------------------- #
    # 7. Feedback loop                                                      #
    # --------------------------------------------------------------------- #
    section("STEP 6 / 6  --  Attach FeedbackEvent records")
    events = [
        FeedbackEvent(
            feedback_id="fb-001",
            kind="thumbs_down",
            severity="high",
            rating=0.0,
            note="Photosynthesis answer ignored the retrieved soil/watering text — context off-topic.",
            metadata={"question_index": 4},
        ),
        FeedbackEvent(
            feedback_id="fb-002",
            kind="hallucination",
            severity="critical",
            rating=0.0,
            note="Stated water boils at 90 C — incorrect; ground truth is 100 C at sea level.",
            metadata={"question_index": 2},
        ),
    ]
    updated = store.attach_feedback_to_run(saved.run_id, events)
    print(f"Attached         : {len(events)}")
    print(f"Total on run     : {len(updated.evaluation.feedback_events)}")

    summary = store.feedback_summary()
    print(f"Negative rate    : {summary.get('negative_feedback_rate'):.2f}")

    # --------------------------------------------------------------------- #
    # Wrap-up                                                               #
    # --------------------------------------------------------------------- #
    section("DONE")
    print(
        f"Open the dashboard against this store with:\n"
        f"    RAGDX_ROOT={DEMO_ROOT} \\\n"
        f"    PYTHONPATH=src python -m streamlit run src/ragdx/ui/dashboard.py\n"
    )
    print(
        "Or inspect the JSON directly:\n"
        f"    {DEMO_ROOT / 'runs' / (saved.run_id + '.json')}\n"
        f"    {DEMO_ROOT / 'optimization' / 'sessions' / (session.session_id + '.json')}"
    )


if __name__ == "__main__":
    main()
