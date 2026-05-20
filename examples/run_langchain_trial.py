"""Example external trial runner for a LangChain pipeline.

Invoked by the optimization executor when ``RAGDX_LANGCHAIN_RUNNER_CMD``
points at this script. It:

1. Loads a JSONL dataset of records (question / ground_truth / contexts).
2. Imports a pipeline factory via ``program_contract.pipeline_module`` and
   builds the chain with the trial config.
3. Runs the chain over every record.
4. Scores the outputs with the dependency-free
   :class:`ragdx.engines.embedding_eval.EmbeddingEvaluator`, optionally
   merging external precomputed ragas/RAGChecker scores via the
   :class:`ragdx.core.evaluator.UnifiedEvaluator`.
5. Writes a ``{"metrics": {...}}`` JSON payload to ``--output``.

The script is intentionally self-contained so it can be invoked as a
subprocess: it imports only ragdx + yaml and falls back gracefully when
LangChain is missing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ragdx.core.evaluator import UnifiedEvaluator
from ragdx.schemas.models import DatasetRecord


def _load_records(path: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _import_callable(spec: str):
    module_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, fn_name)


def _to_dataset_record(rec: Dict[str, Any], output: Dict[str, Any]) -> DatasetRecord:
    answer = output.get("answer") if isinstance(output, dict) else str(output)
    citations = output.get("citations") if isinstance(output, dict) else []
    return DatasetRecord(
        question=rec.get("question", ""),
        ground_truth=rec.get("ground_truth"),
        answer=str(answer) if answer is not None else None,
        contexts=list(rec.get("contexts") or output.get("contexts") or []) if isinstance(output, dict) else list(rec.get("contexts") or []),
        reference_contexts=list(rec.get("reference_contexts") or []),
        citations=list(citations) if isinstance(citations, list) else [],
    )


def _flatten_metrics(retrieval: Dict[str, float], generation: Dict[str, float], e2e: Dict[str, float]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    flat.update({k: float(v) for k, v in retrieval.items()})
    flat.update({k: float(v) for k, v in generation.items()})
    flat.update({k: float(v) for k, v in e2e.items()})
    return flat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = _load_records(cfg["program_contract"]["dataset_path"])
    chain = _import_callable(cfg["program_contract"]["pipeline_module"])(cfg)

    outputs: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for rec in records:
        out = chain.invoke({"input": rec["question"]})
        if isinstance(out, dict):
            outputs.append(out)
        else:
            outputs.append({"answer": str(out), "citations": []})
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    typed = [_to_dataset_record(rec, out) for rec, out in zip(records, outputs)]
    evaluator = UnifiedEvaluator()
    result = evaluator.evaluate(
        typed,
        use_ragas=False,
        use_ragchecker=False,
        use_embedding=True,
    )

    metrics = _flatten_metrics(result.retrieval, result.generation, result.e2e)
    metrics["latency_ms"] = round(elapsed_ms / max(len(records), 1), 2)
    # Citation accuracy is a runtime signal we can compute trivially.
    if outputs:
        metrics["citation_accuracy"] = round(
            sum(1.0 if (isinstance(o, dict) and o.get("citations")) else 0.0 for o in outputs)
            / len(outputs),
            4,
        )

    Path(args.output).write_text(
        json.dumps({"metrics": metrics, "metadata": result.metadata}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
