"""Re-score an existing ``result.json`` with the new CompositeObjective
without re-running any LLM calls. Reads each AutoRAG grid run and DSPy
before/after pack, annotates them with composite score + feasibility
using :func:`ragdx.optim.objectives.default_objective` (or a user-defined
override), and writes the bundle back in place.

Usage::

    PYTHONPATH=src python examples/_rescore_with_composite.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ragdx.optim.objectives import default_objective  # noqa: E402


def rescore(path: Path) -> None:
    bundle = json.loads(path.read_text(encoding="utf-8"))

    objectives = {
        "with_gt": default_objective("with_gt"),
        "no_gt": default_objective("no_gt"),
    }

    # --- AutoRAG grid: annotate each run + bump best by composite ---
    grid_section = bundle.get("autorag_grid", {})
    for mode, key in (("with_gt", "with_gt"), ("no_gt", "no_gt")):
        if key not in grid_section:
            continue
        grid = grid_section[key]
        obj = objectives[mode]
        for r in grid.get("runs", []):
            ev = obj.evaluate(r.get("scores", {}))
            r["composite_score"] = ev["score"]
            r["feasible"] = ev["feasible"]
            r["violations"] = ev["violations"]
        best = obj.best_among(grid.get("runs", []))
        if best is not None:
            grid["objective_spec"] = obj.to_dict()
            grid["best_top_k"] = best["top_k"]
            grid["best_scores"] = best.get("scores", {})
            grid["best_composite"] = best["_composite"]
            grid["best_feasible"] = best["_feasible"]

    # --- DSPy before/after: annotate each phase with composite ---
    ba_section = bundle.get("dspy_before_after", {})
    for mode, key in (("with_gt", "with_gt"), ("no_gt", "no_gt")):
        if key not in ba_section:
            continue
        ba = ba_section[key]
        obj = objectives[mode]
        baseline_ev = obj.evaluate(ba.get("baseline_scores", {}))
        optimised_ev = obj.evaluate(ba.get("optimized_scores", {}))
        ba["composite"] = {
            "objective_spec": obj.to_dict(),
            "baseline": baseline_ev,
            "optimized": optimised_ev,
            "delta": optimised_ev["score"] - baseline_ev["score"],
        }

    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Re-scored bundle written -> {path}")

    # Quick summary
    print("\nAutoRAG winners (by composite):")
    for mode in ("with_gt", "no_gt"):
        if mode in grid_section:
            g = grid_section[mode]
            print(
                f"  [{mode}] top_k={g.get('best_top_k')} "
                f"composite={g.get('best_composite'):.3f} "
                f"feasible={g.get('best_feasible')}"
            )

    print("\nDSPy before/after (composite Δ):")
    for mode in ("with_gt", "no_gt"):
        if mode in ba_section:
            c = ba_section[mode]["composite"]
            print(
                f"  [{mode}] baseline={c['baseline']['score']:.3f} "
                f"-> optimized={c['optimized']['score']:.3f} "
                f"(Δ={c['delta']:+.3f})"
            )


def main() -> None:
    targets = [
        REPO / ".ragdx_optimize_demo" / "result.json",
        REPO / "docs" / "examples" / "optimize_gt_modes_result.json",
    ]
    for t in targets:
        if t.exists():
            print(f"\n=== rescoring {t} ===")
            rescore(t)


if __name__ == "__main__":
    main()
