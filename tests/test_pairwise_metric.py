"""Tests for the pairwise DSPy inner-loop metric."""

from __future__ import annotations

import pytest

dspy = pytest.importorskip("dspy")

from ragdx.optim._pairwise_dspy_metric import make_pairwise_metric  # noqa: E402


class _Example:
    def __init__(self, question: str, context: str = "ctx") -> None:
        self.question = question
        self.context = context


class _Pred:
    def __init__(self, answer: str) -> None:
        self.answer = answer


def _fake_judge_lm(winner: str, monkeypatch):
    """Patch dspy.Predict so the judge returns a fixed verdict."""

    class _Out:
        def __init__(self) -> None:
            self.winner = winner
            self.reason = "because"

    class _FakePredict:
        def __init__(self, _sig) -> None:
            pass

        def __call__(self, **kwargs):
            return _Out()

    monkeypatch.setattr(dspy, "Predict", _FakePredict)
    return object()  # the lm handle is only used inside dspy.context


def test_empty_candidate_scores_zero(monkeypatch):
    lm = _fake_judge_lm("A", monkeypatch)
    metric = make_pairwise_metric(lm, {"q": "baseline answer"})
    assert metric(_Example("q"), _Pred("")) == 0.0


def test_missing_baseline_is_neutral(monkeypatch):
    lm = _fake_judge_lm("A", monkeypatch)
    metric = make_pairwise_metric(lm, {})
    assert metric(_Example("q"), _Pred("some answer")) == 0.5


def test_tie_scores_half(monkeypatch):
    lm = _fake_judge_lm("tie", monkeypatch)
    metric = make_pairwise_metric(lm, {"q": "baseline"})
    assert metric(_Example("q"), _Pred("candidate")) == 0.5


def test_candidate_win_scores_one_regardless_of_slot(monkeypatch):
    # The deterministic order swap assigns the candidate to slot A for
    # some questions and slot B for others. Whichever slot it lands in,
    # a judge verdict for THAT slot must map to score 1.0.
    questions = [f"question-{i}" for i in range(8)]
    for q in questions:
        for verdict in ("A", "B"):
            lm = _fake_judge_lm(verdict, monkeypatch)
            metric = make_pairwise_metric(lm, {q: "baseline"})
            score = metric(_Example(q), _Pred("candidate"))
            # score is 1.0 when the verdict matches the candidate's slot,
            # 0.0 when it matches the baseline's slot — never anything else.
            assert score in (0.0, 1.0)
    # And across a set of questions both outcomes occur (slots really swap).
    lm = _fake_judge_lm("A", monkeypatch)
    metric = make_pairwise_metric(lm, {q: "baseline" for q in questions})
    scores = {metric(_Example(q), _Pred("candidate")) for q in questions}
    assert scores == {0.0, 1.0}


def test_unparseable_verdict_is_neutral(monkeypatch):
    lm = _fake_judge_lm("hmm, both answers have merit (a and b)", monkeypatch)
    metric = make_pairwise_metric(lm, {"q": "baseline"})
    assert metric(_Example("q"), _Pred("candidate")) == 0.5


def test_verbose_verdicts_are_tolerated(monkeypatch):
    lm = _fake_judge_lm("Answer B", monkeypatch)
    metric = make_pairwise_metric(lm, {"q": "baseline"})
    assert metric(_Example("q"), _Pred("candidate")) in (0.0, 1.0)


def test_feedback_mode_returns_prediction(monkeypatch):
    lm = _fake_judge_lm("tie", monkeypatch)
    metric = make_pairwise_metric(lm, {"q": "baseline"}, with_feedback=True)
    out = metric(_Example("q"), _Pred("candidate"))
    assert float(out.score) == 0.5
    assert "tie" in out.feedback.lower()
