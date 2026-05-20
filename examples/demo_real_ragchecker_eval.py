"""Real RAGChecker evaluation with GLM-4-Flash (Zhipu) as extractor + checker.

Mirrors ``demo_real_ragas_eval.py`` but uses the RAGChecker engine instead of
ragas. RAGChecker decomposes answers and reference into atomic claims, then
checks each claim against the retrieved contexts — yielding metrics like
``precision`` (claim_precision), ``recall`` (claim_recall), ``faithfulness``,
``relevant_noise_sensitivity``, ``hallucination``, etc.

Configures RAGChecker to call GLM-4-Flash via the OpenAI-compatible endpoint
(``base_url=https://open.bigmodel.cn/api/paas/v4``). Local
sentence-transformers is used for any semantic distance internal to RAGChecker.

Usage::

    ZHIPU_API_KEY=<key> PYTHONPATH=src python examples/demo_real_ragchecker_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEMO_ROOT = REPO / ".ragdx_ragchecker_eval"
os.environ.setdefault("RAGDX_ROOT", str(DEMO_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LiteLLM (used inside RAGChecker) reads these for OpenAI-compatible providers.
key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
if key:
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"

os.environ["PYTHONPATH"] = (
    str(REPO / "src") + os.pathsep + str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from datasets import load_dataset  # noqa: E402

# ----------------------------------------------------------------------
# Patch: RAGChecker calls litellm.batch_completion with temperature=1e-5
# which GLM-4-Flash rejects ("API 调用参数有误"). Clamp to >= 0.01.
# ----------------------------------------------------------------------
import litellm  # noqa: E402
_orig_batch_completion = litellm.batch_completion


def _clamped_batch_completion(*args, **kwargs):
    t = kwargs.get("temperature")
    if t is not None and 0 < t < 0.01:
        kwargs["temperature"] = 0.01
    return _orig_batch_completion(*args, **kwargs)


litellm.batch_completion = _clamped_batch_completion
# refchecker.utils imports batch_completion at module load — repatch there too.
try:
    from refchecker import utils as _rc_utils  # type: ignore
    _rc_utils.batch_completion = _clamped_batch_completion
except Exception:
    pass

from ragdx import UnifiedEvaluator  # noqa: E402
from ragdx.schemas.models import DatasetRecord  # noqa: E402


def section(t: str) -> None:
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def build_ragchecker():
    if not key:
        raise RuntimeError(
            "Set ZHIPU_API_KEY (or OPENAI_API_KEY) before running this demo."
        )
    from ragchecker import RAGChecker
    return RAGChecker(
        extractor_name="openai/glm-4-flash",
        checker_name="openai/glm-4-flash",
        extractor_api_base="https://open.bigmodel.cn/api/paas/v4",
        checker_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=key,
        batch_size_extractor=4,   # be gentle on free-tier rate limits
        batch_size_checker=4,
        joint_check=True,
    )


def build_corpus_and_questions():
    ds = load_dataset("explodinggradients/amnesty_qa", "english_v3", split="eval[:5]")
    corpus_chunks = []
    questions = []
    for i, row in enumerate(ds):
        for j, passage in enumerate(row.get("retrieved_contexts") or []):
            text = (passage or "").strip()
            if not text:
                continue
            corpus_chunks.append({"text": text, "source": f"amnesty_qa#{i}-{j}"})
        questions.append({
            "question": row["user_input"],
            "ground_truth": row.get("reference") or "",
            "reference_contexts": [row.get("reference") or ""] if row.get("reference") else [],
        })

    DEMO_ROOT.mkdir(parents=True, exist_ok=True)
    corpus_path = DEMO_ROOT / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for c in corpus_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return corpus_path, questions


def run_pipeline_for_topk(corpus_path: Path, top_k: int, questions: list):
    from examples.langchain_pipeline_real import create_pipeline

    cfg = {
        "search_parameters": {"corpus_path": str(corpus_path).replace("\\", "/")},
        "runtime": {"retriever_k": top_k},
    }
    chain = create_pipeline(cfg)

    records: list[DatasetRecord] = []
    for q in questions:
        out = chain.invoke({"input": q["question"]})
        inner = out.get("answer", "")
        if isinstance(inner, dict):
            answer_text = str(inner.get("answer", ""))
            ctxs = list(inner.get("contexts") or [])
        else:
            answer_text = str(inner)
            docs = out.get("context") or []
            ctxs = [getattr(d, "page_content", str(d)) for d in docs]
        records.append(DatasetRecord(
            question=q["question"],
            ground_truth=q["ground_truth"],
            answer=answer_text,
            contexts=ctxs,
            reference_contexts=q["reference_contexts"],
        ))
    return records


def main():
    section("STEP 1  --  Build corpus + load amnesty_qa")
    corpus_path, questions = build_corpus_and_questions()
    print(f"Corpus path : {corpus_path}")
    print(f"Questions   : {len(questions)}")

    section("STEP 2  --  Construct RAGChecker (GLM-4-Flash as extractor+checker)")
    checker = build_ragchecker()
    print("Extractor   : openai/glm-4-flash @ open.bigmodel.cn")
    print("Checker     : openai/glm-4-flash @ open.bigmodel.cn")

    evaluator = UnifiedEvaluator()
    summary = []
    # RAGChecker's lightweight + most informative metrics.
    metric_names = ["precision", "recall", "f1", "faithfulness", "hallucination"]

    for top_k in [1, 3, 6]:
        section(f"STEP 3.{top_k}  --  top_k={top_k}: pipeline -> RAGChecker")
        records = run_pipeline_for_topk(corpus_path, top_k, questions)
        print(f"  generated {len(records)} answers")

        # Drive RAGChecker through ragdx's adapter — exercises the real
        # UnifiedEvaluator code path now that the query_id bug is fixed.
        result = evaluator.evaluate(
            records,
            use_ragas=False, use_ragchecker=True, use_embedding=False,
            run_ragchecker=True,
            ragchecker_kwargs={
                "evaluator": checker,
                "metric_names": metric_names,
            },
        )
        # The adapter writes the original ragchecker keys into raw_tool_outputs
        # and normalizes a subset into the standard buckets. We surface both.
        raw = dict(result.raw_tool_outputs.get("ragchecker") or {})
        normalized = {**result.retrieval, **result.generation, **result.e2e}

        print("\n  Raw RAGChecker metrics (avg over 5 records):")
        for k, v in sorted(raw.items()):
            print(f"    {k:<26s} {v:.4f}")
        print("\n  Normalized into ragdx buckets:")
        for k, v in sorted(normalized.items()):
            print(f"    {k:<26s} {v:.4f}")

        summary.append({
            "top_k": top_k,
            "raw": raw,
            "normalized": normalized,
        })

    section("STEP 4  --  SIDE-BY-SIDE SUMMARY (raw RAGChecker scores)")
    all_keys = sorted({k for s in summary for k in s["raw"]})
    header = f"  {'metric':<26s} | " + " | ".join(f"k={s['top_k']:<2d}" for s in summary)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in all_keys:
        row = [f"{key:<26s}"]
        for s in summary:
            v = s["raw"].get(key)
            row.append(f"{v:.3f}" if v is not None else "  -  ")
        print("  " + " | ".join(row))

    out_json = DEMO_ROOT / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary -> {out_json}")


if __name__ == "__main__":
    main()
