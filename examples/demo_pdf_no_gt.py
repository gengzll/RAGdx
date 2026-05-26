"""End-to-end no-GT RAG pipeline on a PDF corpus.

Mirrors the structure of ``demo_optimize_gt_modes.py`` but for the
realistic "I have a PDF, no labels" scenario:

  Step 1  Load PDF  ->  ~500 chunks via pypdf + RecursiveCharacterTextSplitter
  Step 2  Synthesise 5 questions from random chunks (GLM-4-Flash)
  Step 3  AutoRAG search with Bayesian optimisation over
          (chunk_size, top_k, chunk_overlap), 10 trials
  Step 4  DSPy before/after using the AutoRAG winner config:
          - baseline RAG signature -> ragas eval (ref-free metric set)
          - MIPROv2 optimise        -> rerun + eval
          - composite delta vs production-grade default_objective("no_gt")

All cells call GLM-4-Flash via Zhipu (no offline path). The bundle is
saved to ``.ragdx_pdf_no_gt_demo/result.json`` and consumed by the same
Streamlit dashboard.

Usage::

    ZHIPU_API_KEY=<key> PYTHONPATH=src \\
      python examples/demo_pdf_no_gt.py path/to/your.pdf
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_PDF = REPO / "e0522-asmptesgreport.pdf"
OUTPUT_ROOT = REPO / ".ragdx_pdf_no_gt_demo"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_ROOT / "result.json"

KEY = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not KEY:
    raise SystemExit(
        "ZHIPU_API_KEY (or OPENAI_API_KEY) is required. Run with:\n"
        "    ZHIPU_API_KEY=<key> PYTHONPATH=src \\\n"
        "      python examples/demo_pdf_no_gt.py [path/to/your.pdf]"
    )

os.environ["OPENAI_API_KEY"] = KEY
os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# GLM rejects temperature < 0.01 -- clamp inside litellm.
import litellm  # noqa: E402

_orig_completion = litellm.completion
_orig_batch = litellm.batch_completion


def _clamp(kwargs):
    t = kwargs.get("temperature")
    if t is not None and 0 < t < 0.01:
        kwargs["temperature"] = 0.01
    return kwargs


litellm.completion = lambda *a, **kw: _orig_completion(*a, **_clamp(kw))
litellm.batch_completion = lambda *a, **kw: _orig_batch(*a, **_clamp(kw))


from ragdx import UnifiedEvaluator  # noqa: E402
from ragdx.datasets import synthesize_questions  # noqa: E402
from ragdx.loaders import load_pdf_chunks  # noqa: E402
from ragdx.optim.bayes_search import BayesianSearch  # noqa: E402
from ragdx.optim.dspy_adapter import DSPyAdapter  # noqa: E402
from ragdx.optim.objectives import default_objective  # noqa: E402
from ragdx.schemas.models import DatasetRecord  # noqa: E402


# ----------------------------------------- DSPy trial-score capture helper
class _MIPROTrialScoreCapture(logging.Handler):
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


# ---------------------------------------------------------------- runtime
def build_litellm_callable():
    def _call(prompt: str) -> str:
        resp = litellm.completion(
            model="openai/glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            api_key=KEY,
            api_base="https://open.bigmodel.cn/api/paas/v4",
            temperature=0.01,
            max_tokens=350,
            timeout=60,
        )
        return resp.choices[0].message.content or ""

    return _call


def build_dspy_lm():
    import dspy

    return dspy.LM(
        "openai/glm-4-flash",
        api_key=KEY, api_base="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.01, max_tokens=400, cache=False,
    )


def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )


def build_ragas_judge():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOpenAI(
        model="glm-4-flash", api_key=KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=0.01, timeout=60, max_retries=2,
    ))


def build_ragas_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    ))


def build_ragas_metrics():
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    return [faithfulness, answer_relevancy, context_precision]


# ------------------------------------------------ RAG primitives
def build_vector_store(chunks, embeddings):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_texts(chunks, embeddings)


def retrieve(vstore, question: str, top_k: int) -> list[str]:
    return [d.page_content for d in vstore.similarity_search(question, k=top_k)]


def generate_answer(question: str, contexts: list[str], lm) -> str:
    ctx_str = "\n---\n".join(contexts) if contexts else "(no context)"
    prompt = (
        "Answer the question using only the provided context.\n"
        "Be concise. Do not invent facts not in the context.\n\n"
        f"Context:\n{ctx_str}\n\nQuestion: {question}\n\nAnswer:"
    )
    try:
        return lm(prompt)
    except Exception as e:  # pragma: no cover - live LLM
        return f"<generation error: {e}>"


# ------------------------------------------------ eval helper
def evaluate_with_ragas(records, judge, ragas_embeddings, metrics) -> dict:
    evaluator = UnifiedEvaluator()
    try:
        result = evaluator.evaluate(
            records,
            use_ragas=True, run_ragas=True,
            use_ragchecker=False, use_embedding=False,
            ragas_kwargs={"llm": judge, "embeddings": ragas_embeddings, "metrics": metrics},
        )
        return {
            "scores": {**result.retrieval, **result.generation, **result.e2e},
            "skipped": result.metadata.get("skipped_metrics", {}),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "scores": {}}


# --------------------------------------------------------------- utils
def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def kv(label, value, width=22):
    print(f"  {label:<{width}s} {value}")


# ===================================================================
# STEP 3 — BO search over chunking + retrieval params
# ===================================================================
def run_bo_search(
    pdf_path: Path,
    questions: list[dict],
    objective,
    judge,
    ragas_embeddings,
    metrics,
    *,
    n_trials: int = 8,
    n_init: int = 3,
) -> dict:
    """Run BayesianSearch over (chunk_size, top_k, chunk_overlap). Each
    trial re-chunks the PDF + builds a fresh vector store (chunk_size
    affects both), evaluates ragas, reports composite score back."""
    search_space = {
        "chunk_size": [256, 512, 1024],
        "chunk_overlap": [0, 50, 100],
        "top_k": [1, 3, 5, 7],
    }
    print(f"  Search space (grid size = "
          f"{len(search_space['chunk_size']) * len(search_space['chunk_overlap']) * len(search_space['top_k'])}):")
    for k, v in search_space.items():
        print(f"    {k}: {v}")
    print(f"  BO: n_init={n_init}, max_trials={n_trials}")

    bo = BayesianSearch(search_space, n_init=n_init, max_trials=n_trials, seed=42)
    embeddings = build_embeddings()
    lm = build_litellm_callable()

    # Cache: (chunk_size, chunk_overlap) -> (chunks_count, vstore).
    vstore_cache: dict[tuple, tuple] = {}

    trials_out: list[dict] = []
    while bo.has_next():
        params = bo.next_params()
        t_start = time.time()
        print(f"\n  [Trial {len(bo.trials) + 1}/{n_trials}] {params}")

        key = (params["chunk_size"], params["chunk_overlap"])
        if key not in vstore_cache:
            loader_result = load_pdf_chunks(
                pdf_path,
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
            )
            print(f"    re-chunked: {len(loader_result.chunks)} chunks")
            vstore_cache[key] = (len(loader_result.chunks), build_vector_store(loader_result.chunks, embeddings))
        n_chunks, vstore = vstore_cache[key]

        # Generate answers for every question with this config
        answered = []
        for i, q in enumerate(questions):
            ctxs = retrieve(vstore, q["question"], params["top_k"])
            ans = generate_answer(q["question"], ctxs, lm)
            answered.append(DatasetRecord(
                question=q["question"],
                ground_truth=q.get("ground_truth", None),
                contexts=ctxs,
                answer=ans,
            ))
            print(f"      Q{i + 1}: {len(ctxs)} ctxs, answer={ans[:70]!r}")

        # ragas eval
        ev = evaluate_with_ragas(answered, judge, ragas_embeddings, metrics)
        scores = ev.get("scores", {})
        composite = objective.evaluate(scores)
        bo.report(params, composite["score"])

        elapsed = time.time() - t_start
        print(f"    scores: {scores}")
        print(f"    composite={composite['score']:.3f}  feasible={composite['feasible']}  "
              f"({elapsed:.1f}s)")
        trials_out.append({
            "trial_index": len(bo.trials) - 1,
            "params": params,
            "n_chunks": n_chunks,
            "scores": scores,
            "composite_score": composite["score"],
            "feasible": composite["feasible"],
            "violations": composite["violations"],
            "answers_preview": [a.answer[:200] for a in answered],
        })

    best = bo.best_trial
    print(f"\n  -> BO winner: params={best.params} composite={best.score:.3f}")
    return {
        "search_space": search_space,
        "n_init": n_init,
        "max_trials": n_trials,
        "objective_spec": objective.to_dict(),
        "trials": trials_out,
        "best_params": best.params,
        "best_composite": best.score,
    }


# ===================================================================
# STEP 4 — DSPy before/after using the BO-winning config
# ===================================================================
def run_dspy_before_after(records, judge, ragas_embeddings, metrics, objective) -> dict:
    import dspy

    student = build_dspy_lm()
    dspy.configure(lm=student)

    adapter = DSPyAdapter()
    baseline_program = adapter.build_program()

    def _run_program(program):
        out = []
        for r in records:
            ctx_str = "\n".join(r.contexts) if r.contexts else ""
            try:
                with dspy.context(lm=student):
                    pred = program(question=r.question, context=ctx_str)
                ans = str(getattr(pred, "answer", "") or "")
            except Exception as e:  # pragma: no cover
                ans = f"<error: {e}>"
            out.append(DatasetRecord(
                question=r.question,
                ground_truth=r.ground_truth,
                contexts=r.contexts,
                answer=ans,
            ))
        return out

    print("\n  (a) BASELINE — default RAG Signature ...")
    baseline_answered = _run_program(baseline_program)
    baseline_eval = evaluate_with_ragas(baseline_answered, judge, ragas_embeddings, metrics)
    print(f"      baseline scores: {baseline_eval.get('scores')}")

    print("\n  (b) MIPROv2 optimisation ...")
    capture = _MIPROTrialScoreCapture()
    capture.setLevel(logging.INFO)
    mipro_logger = logging.getLogger("dspy.teleprompt.mipro_optimizer_v2")
    mipro_logger.addHandler(capture)
    prior = mipro_logger.level
    mipro_logger.setLevel(logging.INFO)
    try:
        opt_result = adapter.optimize(
            records, student_lm=student, judge_lm=student,
            optimizer="MIPROv2", optimizer_kwargs={"auto": "light"},
        )
    finally:
        mipro_logger.removeHandler(capture)
        mipro_logger.setLevel(prior)
    print(f"      MIPROv2 done: {len(capture.scores_so_far)} trials, "
          f"best={max(capture.best_scores or [0]):.2f}")

    print("\n  (c) OPTIMIZED — re-run with optimised program ...")
    optimised_answered = _run_program(opt_result["optimized_program"])
    opt_eval = evaluate_with_ragas(optimised_answered, judge, ragas_embeddings, metrics)
    print(f"      optimised scores: {opt_eval.get('scores')}")

    baseline_scores = baseline_eval.get("scores", {}) or {}
    optimised_scores = opt_eval.get("scores", {}) or {}
    composite_baseline = objective.evaluate(baseline_scores)
    composite_optimized = objective.evaluate(optimised_scores)
    delta = {
        m: (optimised_scores.get(m, float("nan")) - baseline_scores.get(m, float("nan")))
        for m in sorted(set(baseline_scores) | set(optimised_scores))
    }
    print(f"\n  composite: baseline={composite_baseline['score']:.3f} -> "
          f"optimized={composite_optimized['score']:.3f} "
          f"(Δ={composite_optimized['score'] - composite_baseline['score']:+.3f})")

    return {
        "baseline_scores": baseline_scores,
        "optimized_scores": optimised_scores,
        "delta": delta,
        "composite": {
            "objective_spec": objective.to_dict(),
            "baseline": composite_baseline,
            "optimized": composite_optimized,
            "delta": composite_optimized["score"] - composite_baseline["score"],
        },
        "baseline_sample_answers": [a.answer for a in baseline_answered],
        "optimized_sample_answers": [a.answer for a in optimised_answered],
        "instructions": dict(opt_result["instructions"]),
        "demos": {name: [dict(d) for d in demos] for name, demos in opt_result["demos"].items()},
        "trial_scores": list(capture.scores_so_far),
        "best_score_progression": list(capture.best_scores),
        "gt_mode": opt_result["gt_mode"],
        "optimizer": opt_result["optimizer"],
        "trainset_size": opt_result["trainset_size"],
    }


# ===================================================================
# MAIN
# ===================================================================
def main(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    section("STEP 1 / 4 — Load PDF and chunk")
    print(f"  PDF: {pdf_path}")
    initial_load = load_pdf_chunks(pdf_path, chunk_size=512, chunk_overlap=50)
    kv("pages", initial_load.page_count)
    kv("chunks", len(initial_load.chunks))
    kv("raw_text_chars", initial_load.raw_text_chars)
    print(f"  sample chunk (first): {initial_load.chunks[0][:200]!r}")
    print(f"  sample chunk (mid):   {initial_load.chunks[len(initial_load.chunks)//2][:200]!r}")

    section("STEP 2 / 4 — Synthesise questions from the corpus (no-GT)")
    lm = build_litellm_callable()
    synthesized = synthesize_questions(
        initial_load.chunks, n=5,
        llm_callable=lm,
        chunks_per_question=2,
        max_passages_chars=1500,
    )
    print(f"  generated {len(synthesized)} questions:")
    for i, q in enumerate(synthesized):
        kv(f"  Q{i + 1}", q.question)
    questions = [{"question": q.question, "ground_truth": None} for q in synthesized]

    section("STEP 3 / 4 — AutoRAG search (BO over chunk_size + overlap + top_k)")
    objective = default_objective("no_gt")
    print(f"  Production objective: metrics={objective.metrics}")
    print(f"                        constraints={objective.constraints}")
    judge = build_ragas_judge()
    ragas_embeddings = build_ragas_embeddings()
    metrics = build_ragas_metrics()
    bo_result = run_bo_search(
        pdf_path, questions, objective, judge, ragas_embeddings, metrics,
        n_trials=8, n_init=3,
    )

    section("STEP 4 / 4 — DSPy before/after at BO winner config")
    best_params = bo_result["best_params"]
    print(f"  Best config from BO: {best_params}")
    loader = load_pdf_chunks(
        pdf_path,
        chunk_size=best_params["chunk_size"],
        chunk_overlap=best_params["chunk_overlap"],
    )
    vstore = build_vector_store(loader.chunks, build_embeddings())
    records_dspy = []
    for q in questions:
        ctxs = retrieve(vstore, q["question"], best_params["top_k"])
        records_dspy.append(DatasetRecord(
            question=q["question"],
            ground_truth=None,
            contexts=ctxs,
        ))
    dspy_result = run_dspy_before_after(records_dspy, judge, ragas_embeddings, metrics, objective)

    bundle = {
        "model": "openai/glm-4-flash",
        "model_endpoint": "https://open.bigmodel.cn/api/paas/v4",
        "source_pdf": str(pdf_path.name),
        "pdf_meta": initial_load.metadata,
        "questions": questions,
        "synthesized_meta": [
            {"question": q.question, "source_chunk_ids": q.source_chunk_ids}
            for q in synthesized
        ],
        "gt_mode": "no_gt",
        "objective_spec": objective.to_dict(),
        "autorag_bo": bo_result,
        "dspy_before_after": dspy_result,
    }
    OUTPUT_JSON.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Bundle saved -> {OUTPUT_JSON}")

    section("SUMMARY")
    print(f"  PDF                : {pdf_path.name} ({initial_load.page_count} pages)")
    print(f"  Synthesised Qs     : {len(questions)}")
    print(f"  BO trials          : {len(bo_result['trials'])}")
    print(f"  BO best params     : {best_params}")
    print(f"  BO best composite  : {bo_result['best_composite']:.3f}")
    print(f"  DSPy baseline      : {dspy_result['baseline_scores']}")
    print(f"  DSPy optimized     : {dspy_result['optimized_scores']}")
    print(f"  Composite Δ        : {dspy_result['composite']['delta']:+.3f}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    main(target)
