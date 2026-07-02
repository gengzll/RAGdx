"""Tests for the Excel/CSV ground-truth loader (``ragdx.loaders.tabular``)."""

from __future__ import annotations

import pandas as pd
import pytest

from ragdx.experiments import _load_jsonl_questions
from ragdx.loaders.tabular import (
    detect_mapping,
    load_gt_table,
    missing_required,
    records_from_table,
    write_questions_jsonl,
)


def _write_csv(path, rows, header="question,ground_truth"):
    lines = [header, *rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_detect_mapping_by_convention():
    df = pd.DataFrame({"Question": ["q1"], "Ground Truth": ["a1"], "contexts": ["c1"]})
    mapping = detect_mapping(df)
    assert mapping["question"] == "Question"
    assert mapping["ground_truth"] == "Ground Truth"
    assert mapping["contexts"] == "contexts"
    assert missing_required(mapping) == []


def test_detect_mapping_aliases():
    # 'query' -> question, 'answer' -> ground_truth
    df = pd.DataFrame({"query": ["q"], "answer": ["a"]})
    mapping = detect_mapping(df)
    assert mapping["question"] == "query"
    assert mapping["ground_truth"] == "answer"
    assert mapping["contexts"] is None


def test_detect_mapping_reports_missing():
    df = pd.DataFrame({"foo": ["x"], "bar": ["y"]})
    mapping = detect_mapping(df)
    assert set(missing_required(mapping)) == {"question", "ground_truth"}


def test_records_from_csv_roundtrips_through_engine_loader(tmp_path):
    csv = _write_csv(
        tmp_path / "gt.csv",
        ["What is X?,X is a thing", "Who made Y?,Y was made by Z"],
    )
    df = load_gt_table(csv)
    records = records_from_table(df)
    assert [r.question for r in records] == ["What is X?", "Who made Y?"]
    assert [r.ground_truth for r in records] == ["X is a thing", "Y was made by Z"]

    # The JSONL we write must be readable by the experiment engine's loader.
    jsonl = write_questions_jsonl(records, tmp_path / "questions.jsonl")
    reloaded = _load_jsonl_questions(jsonl)
    assert [r.question for r in reloaded] == [r.question for r in records]
    assert [r.ground_truth for r in reloaded] == [r.ground_truth for r in records]


def test_contexts_cell_is_split(tmp_path):
    csv = _write_csv(
        tmp_path / "gt.csv",
        ["q1,a1,ctx one|ctx two|ctx three"],
        header="question,ground_truth,contexts",
    )
    df = load_gt_table(csv)
    records = records_from_table(df)
    assert records[0].contexts == ["ctx one", "ctx two", "ctx three"]


def test_explicit_mapping_overrides_names(tmp_path):
    csv = _write_csv(tmp_path / "gt.csv", ["ask this,expect that"], header="col_a,col_b")
    df = load_gt_table(csv)
    records = records_from_table(df, {"question": "col_a", "ground_truth": "col_b", "contexts": None})
    assert records[0].question == "ask this"
    assert records[0].ground_truth == "expect that"


def test_missing_required_field_raises(tmp_path):
    csv = _write_csv(tmp_path / "gt.csv", ["only one column"], header="notes")
    df = load_gt_table(csv)
    with pytest.raises(ValueError, match=r"question|ground_truth"):
        records_from_table(df)


def test_empty_question_rows_skipped(tmp_path):
    csv = _write_csv(tmp_path / "gt.csv", ["q1,a1", ",orphan answer", "q2,a2"])
    df = load_gt_table(csv)
    records = records_from_table(df)
    assert [r.question for r in records] == ["q1", "q2"]


def test_xlsx_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")  # noqa: F841
    xlsx = tmp_path / "gt.xlsx"
    pd.DataFrame({"question": ["q1", "q2"], "ground_truth": ["a1", "a2"]}).to_excel(
        xlsx, index=False
    )
    df = load_gt_table(xlsx)
    records = records_from_table(df)
    assert len(records) == 2
    assert records[1].ground_truth == "a2"


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("q,a", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_gt_table(p)
