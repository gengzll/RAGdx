"""Load ground-truth question sets from Excel / CSV tables.

The end-to-end experiment's with-GT path consumes a JSONL of
``{question, ground_truth, contexts?}`` records (see
:func:`ragdx.experiments._load_jsonl_questions`). This module lets a
user supply that ground truth as a spreadsheet instead:

* :func:`load_gt_table` reads ``.csv`` / ``.tsv`` / ``.xlsx`` / ``.xls``
  into a :class:`pandas.DataFrame`.
* :func:`detect_mapping` guesses which columns hold ``question`` /
  ``ground_truth`` / ``contexts`` by case-insensitive name match. The
  Streamlit studio uses this to pre-fill a column-mapping widget and
  falls back to it when the names don't match the convention.
* :func:`records_from_table` applies a mapping and produces
  :class:`~ragdx.schemas.models.DatasetRecord` objects.
* :func:`write_questions_jsonl` serialises records into the exact JSONL
  shape the experiment engine already reads, so the studio can hand the
  file to ``run_experiment(..., questions_path=...)`` unchanged.

The convention (matched case-insensitively, ignoring surrounding
whitespace and ``_``/`` `` differences):

* ``question`` field  ← columns named ``question`` / ``query`` / ``q`` / ``input``
* ``ground_truth``    ← ``ground_truth`` / ``answer`` / ``reference`` / ``expected`` / ``gt``
* ``contexts`` (opt.) ← ``contexts`` / ``context`` / ``reference_contexts`` / ``passages``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ragdx.schemas.models import DatasetRecord

# Logical field -> accepted column-name aliases (normalised).
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "q", "input", "prompt"),
    "ground_truth": (
        "groundtruth", "answer", "reference", "expected", "gt",
        "groundtruthanswer", "referenceanswer", "goldanswer", "gold",
    ),
    "contexts": ("contexts", "context", "referencecontexts", "passages", "evidence"),
}

# Fields the with-GT path cannot run without.
REQUIRED_FIELDS: tuple[str, ...] = ("question", "ground_truth")

# Delimiters used to split a single "contexts" cell into a list.
_CONTEXT_SPLITTERS = ("|", "\n", ";;")


def _normalise(name: str) -> str:
    """Lower-case a column name and strip spaces / underscores / hyphens."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def load_gt_table(path: str | Path) -> pd.DataFrame:
    """Read a ground-truth table into a DataFrame.

    Dispatches on the file extension: ``.csv`` / ``.tsv`` via
    :func:`pandas.read_csv`, ``.xlsx`` / ``.xls`` via
    :func:`pandas.read_excel` (requires ``openpyxl``).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".csv",):
        df = pd.read_csv(p)
    elif suffix in (".tsv",):
        df = pd.read_csv(p, sep="\t")
    elif suffix in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(p)  # needs openpyxl for xlsx/xlsm
    else:
        raise ValueError(
            f"Unsupported ground-truth file: {p.name!r}. "
            "Accepts .csv, .tsv, .xlsx, .xls."
        )
    if df.empty:
        raise ValueError(f"Ground-truth table {p.name!r} has no rows.")
    return df


def detect_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    """Best-guess mapping ``{logical_field: column_name_or_None}``.

    Matches each column against the alias table by normalised name.
    Returns ``None`` for any logical field with no matching column;
    callers check :data:`REQUIRED_FIELDS` against the result to decide
    whether to prompt the user for a manual mapping.
    """
    normalised = {_normalise(col): col for col in df.columns}
    mapping: dict[str, str | None] = {}
    for field, aliases in _FIELD_ALIASES.items():
        match: str | None = None
        for alias in aliases:
            if alias in normalised:
                match = normalised[alias]
                break
        mapping[field] = match
    return mapping


def missing_required(mapping: dict[str, str | None]) -> list[str]:
    """Required logical fields that ``mapping`` leaves unassigned."""
    return [f for f in REQUIRED_FIELDS if not mapping.get(f)]


def _split_contexts(value: Any) -> list[str]:
    """Coerce a single contexts cell into a list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    for sep in _CONTEXT_SPLITTERS:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def records_from_table(
    df: pd.DataFrame, mapping: dict[str, str | None] | None = None
) -> list[DatasetRecord]:
    """Convert a table + column mapping into ``DatasetRecord`` objects.

    ``mapping`` maps the logical fields ``question`` / ``ground_truth``
    / ``contexts`` to column names; ``None`` (the default) auto-detects
    via :func:`detect_mapping`. Rows with an empty question are skipped.
    Raises ``ValueError`` if a required field is unmapped.
    """
    mapping = mapping or detect_mapping(df)
    missing = missing_required(mapping)
    if missing:
        raise ValueError(
            "Cannot build ground-truth records: no column mapped to "
            f"{', '.join(missing)}. Available columns: {list(df.columns)}."
        )

    q_col = mapping["question"]
    gt_col = mapping["ground_truth"]
    ctx_col = mapping.get("contexts")

    records: list[DatasetRecord] = []
    for _, row in df.iterrows():
        question = "" if pd.isna(row[q_col]) else str(row[q_col]).strip()
        if not question:
            continue
        gt_val = row[gt_col]
        ground_truth = None if pd.isna(gt_val) else str(gt_val).strip() or None
        contexts = _split_contexts(row[ctx_col]) if ctx_col else []
        records.append(
            DatasetRecord(
                question=question, ground_truth=ground_truth, contexts=contexts
            )
        )
    if not records:
        raise ValueError("Ground-truth table produced no usable rows (all questions empty).")
    return records


def write_questions_jsonl(records: list[DatasetRecord], path: str | Path) -> Path:
    """Write records as JSONL in the shape the experiment engine reads.

    Mirrors :func:`ragdx.experiments._load_jsonl_questions`: one JSON
    object per line with ``question`` / ``ground_truth`` / ``contexts``.
    Returns the written path.
    """
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    {
                        "question": rec.question,
                        "ground_truth": rec.ground_truth,
                        "contexts": list(rec.contexts),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return p


__all__ = [
    "REQUIRED_FIELDS",
    "detect_mapping",
    "load_gt_table",
    "missing_required",
    "records_from_table",
    "write_questions_jsonl",
]
