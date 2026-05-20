"""Example external trial runner for a LlamaIndex pipeline.

Counterpart to ``run_langchain_trial.py`` for LlamaIndex query engines.
Output schema is identical:

    {"metrics": {<flat metric map>}, "metadata": {...}}

Scoring uses the local TF-IDF embedding proxy (no API calls); replace
``use_embedding=True`` with ``use_ragas=True, run_ragas=True`` and the
appropriate ``ragas_kwargs`` to switch to in-process ragas evaluation.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = _load_records(cfg["program_contract"]["dataset_path"])
    engine = _import_callable(cfg["program_contract"]["pipeline_module"])(cfg)

    outputs: List[str] = []
    started = time.perf_counter()
    for rec in records:
        outputs.append(str(engine.query(rec["question"])))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    typed = [
        DatasetRecord(
            question=rec.get("question", ""),
            ground_truth=rec.get("ground_truth"),
            answer=out,
            contexts=list(rec.get("contexts") or []),
            reference_contexts=list(rec.get("reference_contexts") or []),
        )
        for rec, out in zip(records, outputs)
    ]
    result = UnifiedEvaluator().evaluate(
        typed,
        use_ragas=False,
        use_ragchecker=False,
        use_embedding=True,
    )

    metrics: Dict[str, float] = {}
    metrics.update({k: float(v) for k, v in result.retrieval.items()})
    metrics.update({k: float(v) for k, v in result.generation.items()})
    metrics.update({k: float(v) for k, v in result.e2e.items()})
    metrics["latency_ms"] = round(elapsed_ms / max(len(records), 1), 2)

    Path(args.output).write_text(
        json.dumps({"metrics": metrics, "metadata": result.metadata}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
