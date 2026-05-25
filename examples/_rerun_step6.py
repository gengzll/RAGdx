"""Re-run STEP 6 only (DSPy MIPROv2 descriptive contrast) and merge into
the existing ``result.json``. Useful when the main demo hit a connection
error on step 6 after a long step 5 — re-running just this part avoids
spending the LLM budget for steps 1-5 again.

Usage::

    ZHIPU_API_KEY=<key> PYTHONPATH=src python examples/_rerun_step6.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# Reuse the same env setup, monkey-patches and helpers from the main demo.
from examples.demo_optimize_gt_modes import (  # noqa: E402
    OUTPUT_JSON,
    build_records,
    load_amnesty_corpus,
    run_dspy_optimize_descriptive,
    section,
)


def main() -> None:
    if not OUTPUT_JSON.exists():
        raise SystemExit(
            f"{OUTPUT_JSON} not found -- run the full demo first to "
            "produce steps 1-5, then this helper can fill in step 6."
        )

    bundle = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    print(f"Loaded existing bundle: {len(bundle)} top-level keys")

    print("Re-loading amnesty_qa[:5] for the descriptive step-6 runs ...")
    _chunks, base_records = load_amnesty_corpus()
    records_gt = build_records(base_records, with_gt=True)
    records_no = build_records(base_records, with_gt=False)

    section("STEP 6 / 6 RETRY -- DSPy MIPROv2 (with-GT vs no-GT)")
    dspy_with = run_dspy_optimize_descriptive(records_gt, "with-GT")
    dspy_no = run_dspy_optimize_descriptive(records_no, "no-GT")

    bundle["dspy_descriptive"] = {"with_gt": dspy_with, "no_gt": dspy_no}
    OUTPUT_JSON.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Merged step 6 into -> {OUTPUT_JSON}")

    print("\nSummary:")
    for mode in ("with_gt", "no_gt"):
        p = bundle["dspy_descriptive"][mode]
        if p.get("error"):
            print(f"  [{mode}] error: {p['error']}")
        else:
            trials = len(p.get("trial_scores") or [])
            best = max(p.get("best_score_progression") or [0.0])
            print(f"  [{mode}] {trials} trials, best={best:.2f}")


if __name__ == "__main__":
    main()
