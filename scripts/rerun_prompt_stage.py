"""Re-run ONLY the prompt (generation) stage of an existing experiment.

Reuses the BO winner params and the already-retrieved per-question
contexts stored in the bundle, so the Bayesian-search stage is not
repeated. Splices the fresh ``dspy_a_b`` into the bundle, regenerates
the pre/post diagnosis, writes the ``final/`` deliverables, and
re-renders the HTML report.

Why this exists: the first GEPA demo run produced truncated proposals
(reflection LM capped at the student LM's 600 max_tokens). The code
fix gives the reflection LM 4000 tokens; this script refreshes the
demo artifacts without paying for the BO stage again.

Usage (from the repo root, env vars set)::

    python scripts/rerun_prompt_stage.py workspaces/end2end_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ragdx.experiments import (
    _build_ragas_metrics_for_mode,
    _diagnose_per_mode,
    _supplement_deepeval_metrics,
    _write_final_deliverables,
)
from ragdx.loaders.pdf import load_pdf_chunks
from ragdx.optim.objectives import default_objective
from ragdx.optim.stages import GenerationOptimizer, StageContext
from ragdx.runtime.factories import build_runtime
from ragdx.schemas.models import DatasetRecord
from ragdx.schemas.rag_config import RAGConfig


def main(out_dir: str) -> None:
    out = Path(out_dir)
    bundle_path = out / "result.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    mode = (bundle["meta"].get("modes_run") or ["no_gt"])[0]
    bo = bundle["bayes_search"][mode]
    best = bo.get("best_params") or {}
    old_ab = bundle["dspy_a_b"][mode]
    print(f"[rerun] mode={mode} BO winner={best}")

    # --- Winner config: demo base YAML + BO winner params -------------
    cfg = RAGConfig.from_yaml("rag_config.demo.yaml")
    import os
    cfg.generator.api_key = (
        os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    if best.get("chunk_size"):
        cfg.chunker.chunk_size = int(best["chunk_size"])
    if best.get("chunk_overlap") is not None:
        cfg.chunker.chunk_overlap = int(best["chunk_overlap"])
    if best.get("top_k"):
        cfg.retriever.top_k = int(best["top_k"])

    # --- Records: contexts were already retrieved at the winner config
    records = [
        DatasetRecord(
            question=r["question"],
            ground_truth=r.get("ground_truth"),
            contexts=list(r.get("contexts") or []),
        )
        for r in (old_ab.get("records") or [])
    ]
    if not records:
        raise SystemExit("bundle has no dspy_a_b records to reuse")
    print(f"[rerun] reusing {len(records)} records with retrieved contexts")

    runtime = build_runtime(cfg)
    chunks = load_pdf_chunks(
        bundle["meta"]["source"]["corpus"],
        chunk_size=cfg.chunker.chunk_size,
        chunk_overlap=cfg.chunker.chunk_overlap,
    ).chunks

    ctx = StageContext(
        base_config=cfg,
        chunks_master=chunks,
        records=records,
        objective=default_objective(mode),
        metrics=_build_ragas_metrics_for_mode(mode),
        runtime=runtime,
        label=mode,
        # defaults: dspy_optimizer=gepa, dspy_metric=auto->embed_rubric,
        # mipro_auto=light (30 GEPA metric calls)
    )
    print("[rerun] running GenerationOptimizer (GEPA, 4000-token reflection)...")
    result = GenerationOptimizer().optimize(ctx)

    # --- deepeval supplement on the fresh optimized answers -----------
    try:
        extra = _supplement_deepeval_metrics(
            result.extras.get("records") or [], runtime,
        )
        if extra:
            opt = dict(result.extras.get("optimized_scores") or {})
            opt.update(extra)
            result.extras["optimized_scores"] = opt
            result.extras["deepeval_supplement"] = extra
            print(f"[rerun] deepeval supplement: {extra}")
    except Exception as exc:
        print(f"[rerun] deepeval supplement skipped: {exc}")

    # --- Splice into the bundle + regenerate diagnosis ----------------
    bundle["dspy_a_b"][mode] = result.extras
    bundle["diagnosis"] = _diagnose_per_mode({
        m: {"dspy_a_b": bundle["dspy_a_b"][m],
            "bayes_search": bundle["bayes_search"].get(m, {})}
        for m in bundle["dspy_a_b"]
    })
    bundle_path.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[rerun] bundle updated: {bundle_path}")

    # --- final/ deliverables -------------------------------------------
    final_cfg = result.best_config or cfg
    written = _write_final_deliverables(out, {mode: final_cfg})
    for p in written:
        print(f"[rerun] deliverable: {p}")

    # sanity: winning prompt must not be truncated mid-word
    instr = (result.extras.get("instructions") or {}).get("predict", "")
    print(f"[rerun] winning instruction: {len(instr)} chars, "
          f"ends: {instr[-60:]!r}")
    print("[rerun] DONE")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "workspaces/end2end_demo")
