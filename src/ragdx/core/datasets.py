"""
Dataset Loading and Processing Utilities

Main Idea:
This module provides utilities for loading, processing, and saving RAG evaluation datasets from various file formats. It handles the conversion between different data formats and the internal DatasetRecord model.

Functionalities:
- load_jsonl: Load dataset records from JSON Lines format
- load_json: Load dataset records from JSON format (array or object with 'records' key)
- load_csv: Load dataset records from CSV format with specific column mappings
- load_records: Unified loader that automatically detects format from file extension
- save_records_jsonl: Save dataset records to JSON Lines format

Supported formats:
- JSONL: One JSON object per line
- JSON: Array of objects or object with 'records' array
- CSV: Columns for question, ground_truth, answer, contexts (|| separated), reference_contexts (|| separated)

Usage:
Load a dataset:

    from ragdx.core.datasets import load_records

    records = load_records("evaluation_data.jsonl")
    for record in records:
        print(f"Q: {record.question}")
        print(f"A: {record.answer}")

Save processed records:

    save_records_jsonl(records, "processed_data.jsonl")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List

from ragdx.schemas.models import DatasetRecord


def load_jsonl(path: str | Path) -> List[DatasetRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(DatasetRecord(**json.loads(line)))
    return records


def load_json(path: str | Path) -> List[DatasetRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    return [DatasetRecord(**row) for row in payload]


def load_csv(path: str | Path) -> List[DatasetRecord]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contexts = row.get("contexts", "")
            ref_contexts = row.get("reference_contexts", "")
            rows.append(
                DatasetRecord(
                    question=row["question"],
                    ground_truth=row.get("ground_truth") or None,
                    answer=row.get("answer") or None,
                    contexts=[x for x in contexts.split("||") if x],
                    reference_contexts=[x for x in ref_contexts.split("||") if x],
                    metadata={k: v for k, v in row.items() if k not in {"question", "ground_truth", "answer", "contexts", "reference_contexts"}},
                )
            )
    return rows


def load_records(path: str | Path) -> List[DatasetRecord]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported dataset file type: {suffix}")


def save_records_jsonl(records: Iterable[DatasetRecord], path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
    return path
