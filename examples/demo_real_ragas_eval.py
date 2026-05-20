"""Real RAG evaluation with ragas + GLM-4-Flash (Zhipu) as judge LLM.

This demo:
1. Loads amnesty_qa (5 records).
2. Builds the real FAISS LangChain pipeline once for each top_k in [1,3,6].
3. Collects (question, answer, retrieved_contexts) per trial.
4. Evaluates each trial with BOTH:
   - embedding-proxy (TF-IDF cosine, the cheap signal)
   - real ragas with GLM-4-Flash as judge + local sentence-transformers
5. Prints a side-by-side comparison.

Reads ZHIPU_API_KEY from env (set externally — never hardcode keys).

Usage::

    ZHIPU_API_KEY=<your key> PYTHONPATH=src python examples/demo_real_ragas_eval.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEMO_ROOT = REPO / ".ragdx_ragas_eval"
os.environ.setdefault("RAGDX_ROOT", str(DEMO_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make `examples` importable for the pipeline factory.
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

from ragdx import UnifiedEvaluator  # noqa: E402
from ragdx.schemas.models import DatasetRecord  # noqa: E402


def section(t: str) -> None:
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def build_zhipu_judge():
    """Build a ragas-compatible LLM judge backed by GLM-4-Flash."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Set ZHIPU_API_KEY (or OPENAI_API_KEY) before running this demo."
        )

    chat = ChatOpenAI(
        model="glm-4-flash",
        api_key=key,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )
    return LangchainLLMWrapper(chat)


def build_local_embeddings():
    """Local sentence-transformers embeddings (no API calls)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


def build_corpus_and_questions():
    """Load amnesty_qa, build corpus + return per-record DatasetRecord scaffolds."""
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
    """Build the FAISS LangChain chain at this top_k and run all questions."""
    from examples.langchain_pipeline_real import create_pipeline

    cfg = {
        "search_parameters": {"corpus_path": str(corpus_path).replace("\\", "/")},
        "runtime": {"retriever_k": top_k},
    }
    chain = create_pipeline(cfg)

    records: list[DatasetRecord] = []
    for q in questions:
        out = chain.invoke({"input": q["question"]})
        # ``create_retrieval_chain`` puts the combine-chain output under "answer".
        # Our _combine returns a dict {"answer", "citations", "contexts"}.
        inner = out.get("answer", "")
        if isinstance(inner, dict):
            answer_text = str(inner.get("answer", ""))
            ctxs = list(inner.get("contexts") or [])
            citations = list(inner.get("citations") or [])
        else:
            answer_text = str(inner)
            # Fall back to the retriever's docs at out["context"].
            docs = out.get("context") or []
            ctxs = [getattr(d, "page_content", str(d)) for d in docs]
            citations = []
        records.append(DatasetRecord(
            question=q["question"],
            ground_truth=q["ground_truth"],
            answer=answer_text,
            contexts=ctxs,
            reference_contexts=q["reference_contexts"],
        ))
    return records


def pick_lightweight_metrics():
    """Pick the 3 most informative ragas metrics that minimize LLM calls."""
    from ragas.metrics import (
        Faithfulness,
        ContextPrecision,
        AnswerRelevancy,
    )
    return [
        Faithfulness(),          # answer ↔ retrieved contexts consistency
        ContextPrecision(),      # are retrieved contexts relevant?
        AnswerRelevancy(),       # does answer address the question?
    ]


def main():
    section("STEP 1  --  Build corpus + load 5 amnesty_qa questions")
    corpus_path, questions = build_corpus_and_questions()
    print(f"Corpus path : {corpus_path}")
    print(f"Questions   : {len(questions)}")

    section("STEP 2  --  Construct ragas judge (GLM-4-Flash) + local embeddings")
    judge_llm = build_zhipu_judge()
    judge_emb = build_local_embeddings()
    metrics = pick_lightweight_metrics()
    print(f"Judge LLM   : glm-4-flash @ open.bigmodel.cn  (free)")
    print(f"Embeddings  : sentence-transformers/all-MiniLM-L6-v2  (local)")
    print(f"Metrics     : {[m.__class__.__name__ for m in metrics]}")

    evaluator = UnifiedEvaluator()
    summary = []

    for top_k in [1, 3, 6]:
        section(f"STEP 3.{top_k}  --  top_k={top_k}: pipeline -> evaluate")
        records = run_pipeline_for_topk(corpus_path, top_k, questions)
        print(f"  generated {len(records)} answers")

        # Embedding-proxy (always cheap)
        proxy = evaluator.evaluate(
            records, use_ragas=False, use_ragchecker=False, use_embedding=True,
        )
        proxy_metrics = {**proxy.retrieval, **proxy.generation, **proxy.e2e}

        # Real ragas with GLM judge
        real = evaluator.evaluate(
            records,
            use_ragas=True, use_ragchecker=False, use_embedding=False,
            run_ragas=True,
            ragas_kwargs={"llm": judge_llm, "embeddings": judge_emb, "metrics": metrics},
        )
        real_metrics = {**real.retrieval, **real.generation, **real.e2e}

        print("\n  PROXY  (TF-IDF cosine, local):")
        for k, v in sorted(proxy_metrics.items()):
            print(f"    {k:<24s} {v:.4f}")
        print("\n  REAL   (ragas + GLM-4-Flash):")
        for k, v in sorted(real_metrics.items()):
            print(f"    {k:<24s} {v:.4f}")

        summary.append({
            "top_k": top_k,
            "proxy": proxy_metrics,
            "real": real_metrics,
        })

    section("STEP 4  --  SIDE-BY-SIDE SUMMARY")
    common_keys = sorted({k for s in summary for k in s["proxy"]} | {k for s in summary for k in s["real"]})
    header = f"  {'metric':<24s} | " + " | ".join(f"k={s['top_k']:<2d} (proxy/real)" for s in summary)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in common_keys:
        row = [f"{key:<24s}"]
        for s in summary:
            p = s["proxy"].get(key)
            r = s["real"].get(key)
            cell = f"{p:.3f}/{r:.3f}" if (p is not None and r is not None) else (
                f"{p:.3f}/  -  " if p is not None else f"  -  /{r:.3f}"
            )
            row.append(f"{cell:<16s}")
        print("  " + " | ".join(row))

    out_json = DEMO_ROOT / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved summary -> {out_json}")


if __name__ == "__main__":
    main()
