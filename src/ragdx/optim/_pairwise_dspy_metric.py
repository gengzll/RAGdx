"""Pairwise LLM-judge metric for DSPy's prompt-optimization inner loop.

Absolute LLM-judge scores saturate: a permissive judge (GLM-4-Flash,
gpt-4o-mini, ...) hands out 0.9-1.0 to nearly every fluent, grounded
answer, so candidate prompts tie with the seed and the optimizer keeps
the baseline. Pairwise comparison sidesteps saturation entirely — the
judge sees the candidate's answer NEXT TO the baseline answer for the
same question and picks a winner:

* candidate wins  -> 1.0
* tie             -> 0.5
* baseline wins   -> 0.0

The seed program scores ~0.5 against itself by construction, so any
candidate scoring above 0.5 represents a *relative* improvement — a
signal that cannot saturate. Pairwise judging is also empirically far
more discriminative than absolute rubric scoring (the standard
LLM-as-judge result), and the judge's one-line reason doubles as the
textual feedback GEPA's reflection step consumes.

Position bias is mitigated with a deterministic order swap keyed on
the question hash: half the questions present the baseline as "A",
half as "B", so a judge that systematically favours one slot averages
out across the trainset.

Cost: 1 judge LLM call per ``metric()`` invocation — the same as the
``embed_rubric`` metric's rubric call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


def make_pairwise_metric(
    judge_lm: Any,
    baseline_answers: dict[str, str],
    *,
    with_feedback: bool = False,
) -> Callable[..., Any]:
    """Build a DSPy metric that scores answers by pairwise comparison.

    Parameters
    ----------
    judge_lm:
        A ``dspy.LM`` used for the comparison call. Typically
        ``runtime.dspy_lm``.
    baseline_answers:
        ``{question: baseline_answer}`` — the reference side of every
        comparison. Built from the baseline (seed prompt) run that the
        generation stage performs before optimization.
    with_feedback:
        When True, return ``dspy.Prediction(score=..., feedback=...)``
        (GEPA consumes the feedback text in its reflection step). When
        False, return a plain float (MIPROv2 / COPRO expect floats).

    Failure modes:

    * Empty candidate answer -> 0.0 ("produced nothing").
    * Missing/empty baseline answer for the question -> 0.5 (neutral;
      nothing to compare against, don't punish or reward).
    * Judge call raises or output unparseable -> 0.5 (neutral; a flaky
      judge call must not decide a comparison).
    """
    try:
        import dspy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "Building the pairwise DSPy metric requires DSPy. "
            "Install with `pip install ragdx[dspy]`."
        ) from exc

    class _PairwiseJudge(dspy.Signature):
        """Two answers to the same question, based on the same context.
        Decide which better answers the question USING ONLY the context.

        Judge on: (1) factual grounding in the context — penalize any
        invented detail; (2) completeness — does it address the whole
        question; (3) directness — no filler or hedging. Ignore
        superficial length differences: longer is not better.

        winner must be exactly "A", "B", or "tie".
        """

        question: str = dspy.InputField()
        context: str = dspy.InputField()
        answer_a: str = dspy.InputField()
        answer_b: str = dspy.InputField()
        winner: str = dspy.OutputField(desc='"A", "B", or "tie"')
        reason: str = dspy.OutputField(desc="one concise sentence")

    judge = dspy.Predict(_PairwiseJudge)

    def _result(score: float, feedback: str) -> Any:
        if with_feedback:
            return dspy.Prediction(score=score, feedback=feedback)
        return score

    def metric(example: Any, pred: Any, trace: Any | None = None) -> Any:
        question = str(getattr(example, "question", "") or "")
        ctx = getattr(example, "context", None) or getattr(example, "contexts", "")
        if isinstance(ctx, list):
            joined_ctx = "\n\n".join(str(c) for c in ctx if c)
        else:
            joined_ctx = str(ctx or "")
        candidate = str(getattr(pred, "answer", "") or "").strip()

        if not candidate:
            return _result(0.0, "The candidate produced an empty answer.")

        baseline = str(baseline_answers.get(question, "") or "").strip()
        if not baseline:
            return _result(0.5, "No baseline answer available for comparison.")

        # Deterministic position swap: half the trainset shows the
        # candidate in slot A, half in slot B.
        digest = hashlib.blake2b(
            question.encode("utf-8", errors="ignore"), digest_size=8,
        ).hexdigest()
        candidate_is_a = int(digest, 16) % 2 == 0
        a, b = (candidate, baseline) if candidate_is_a else (baseline, candidate)

        try:
            with dspy.context(lm=judge_lm):
                out = judge(
                    question=question,
                    context=joined_ctx,
                    answer_a=a,
                    answer_b=b,
                )
            raw_winner = str(getattr(out, "winner", "") or "").strip().lower()
            reason = str(getattr(out, "reason", "") or "").strip()
        except Exception as exc:  # pragma: no cover - depends on judge LM
            logger.debug("pairwise judge call failed (neutral 0.5): %s", exc)
            return _result(0.5, "Judge call failed; comparison inconclusive.")

        if raw_winner not in {"a", "b", "tie"}:
            # Tolerate verbose outputs like "Answer A" / "B is better".
            if "tie" in raw_winner or "equal" in raw_winner:
                raw_winner = "tie"
            elif "a" in raw_winner and "b" not in raw_winner:
                raw_winner = "a"
            elif "b" in raw_winner and "a" not in raw_winner:
                raw_winner = "b"
            else:
                logger.debug(
                    "pairwise judge output unparseable (%r); neutral 0.5",
                    raw_winner,
                )
                return _result(0.5, "Judge verdict unparseable; treated as tie.")

        if raw_winner == "tie":
            score = 0.5
        elif (raw_winner == "a") == candidate_is_a:
            score = 1.0
        else:
            score = 0.0
        candidate_label = "A" if candidate_is_a else "B"
        feedback = (
            f"Pairwise vs baseline (candidate was answer {candidate_label}): "
            f"{'candidate won' if score == 1.0 else ('tie' if score == 0.5 else 'baseline won')}. "
            f"Judge: {reason}"
        )
        return _result(score, feedback)

    metric.__ragdx_kind__ = "pairwise"  # type: ignore[attr-defined]
    return metric


__all__ = ["make_pairwise_metric"]
