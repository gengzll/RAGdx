"""DSPy adapter — spec rendering plus optional in-process optimisation.

The original adapter only rendered a spec describing what DSPy should do. This
revision keeps that behaviour (``build_optimizer_spec`` / ``run``) but adds a
runnable path:

* :meth:`build_metric_function` returns a ``metric(example, pred, trace=None)``
  callable suitable for any DSPy teleprompter. When the trainset has GT it
  compares ``pred.answer`` to ``example.answer`` (token-overlap F1 by default);
  when there's no GT it falls back to an LLM-as-judge that scores
  faithfulness + relevancy from the retrieved context.

* :meth:`build_trainset` packs ``DatasetRecord``s into ``dspy.Example``s with
  the right input/output fields depending on GT availability.

* :meth:`optimize` actually runs MIPROv2 / COPRO / BootstrapFewShot end-to-end
  and returns the optimised program together with the discovered instructions
  and demos.

DSPy is an optional dependency; the runnable path imports it lazily and raises
a clear ``ImportError`` if it's missing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ragdx.optim._gt_helpers import GTMode, gt_mode
from ragdx.schemas.models import DatasetRecord, OptimizationExperiment, ToolRunResult


# ---------------------------------------------------------------------------
# GT-mode metrics (pure Python — no DSPy needed for these themselves).
# ---------------------------------------------------------------------------
def _token_f1(pred: str, gt: str) -> float:
    """Light token-overlap F1, used as a stable GT-mode default metric.

    Avoids depending on embeddings/LLM calls for the metric itself so the
    DSPy optimizer keeps a fast inner loop. Callers wanting an embedding-based
    metric can pass their own via ``custom_metric``.
    """
    pred_tokens = [t for t in pred.lower().split() if t]
    gt_tokens = [t for t in gt.lower().split() if t]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: dict[str, int] = {}
    for t in pred_tokens:
        if t in gt_tokens:
            common[t] = common.get(t, 0) + 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class DSPyAdapter:
    # ---------------------------------------------------------------- spec API
    def build_optimizer_spec(
        self,
        experiment: OptimizationExperiment,
        parameters: dict[str, Any],
        *,
        records: Iterable[DatasetRecord] | None = None,
    ) -> dict[str, Any]:
        optimizer = parameters.get("optimizer", "MIPROv2")
        mode = self._resolve_mode(experiment, records)
        default_metric = "token_f1" if mode == "with_gt" else "llm_judge_faithfulness"
        objective_metric = experiment.parameters.get("objective_metric", default_metric)
        return {
            "framework": "dspy",
            "optimizer": optimizer,
            "gt_mode": mode,
            "objective_metric": objective_metric,
            "objectives": experiment.objectives,
            "program_contract": {
                "input_fields": ["question", "contexts"],
                "output_fields": ["answer", "citations"],
                "expected_tunable_parts": [
                    "instructions",
                    "fewshot_demos",
                    "citation_formatting",
                    "decomposition",
                ],
            },
            "search_parameters": parameters,
            "compile_hints": {
                "bayesian_optimizer": optimizer == "MIPROv2",
                "fewshot_enabled": parameters.get("fewshot_count", 0) > 0,
                "decomposition": parameters.get("decomposition", False),
                "metric_kind": (
                    "reference_based"
                    if mode == "with_gt"
                    else "reference_free_llm_judge"
                ),
            },
        }

    def run(
        self,
        experiment: OptimizationExperiment,
        parameters: dict[str, Any],
        *,
        records: Iterable[DatasetRecord] | None = None,
    ) -> ToolRunResult:
        spec = self.build_optimizer_spec(experiment, parameters, records=records)
        mode = spec["gt_mode"]
        if mode == "with_gt":
            note = (
                "DSPy spec rendered (with-GT mode). Attach your DSPy program, trainset, "
                "and a reference-based metric (e.g. token-F1 against example.answer). "
                "Or call DSPyAdapter.optimize(records, ...) to run end-to-end."
            )
        else:
            note = (
                "DSPy spec rendered (no-GT mode). The metric must be an LLM-as-judge "
                "scoring faithfulness/relevancy from contexts (since no example.answer "
                "is available). Or call DSPyAdapter.optimize(records, ..., judge_lm=...) "
                "to run end-to-end with the built-in judge."
            )
        return ToolRunResult(tool="dspy", success=True, payload=spec, note=note)

    # -------------------------------------------------------------- runnable
    def _resolve_mode(
        self,
        experiment: OptimizationExperiment | None,
        records: Iterable[DatasetRecord] | None,
    ) -> GTMode:
        if experiment is not None:
            explicit = experiment.parameters.get("gt_mode")
            if explicit in ("with_gt", "no_gt"):
                return explicit
        if records is not None:
            return gt_mode(records)
        return "no_gt"

    def build_metric_function(
        self,
        mode: GTMode,
        *,
        judge_lm: Any | None = None,
        custom_metric: Callable[[Any, Any], float] | None = None,
    ) -> Callable[..., float]:
        """Return a ``metric(example, pred, trace=None) -> float`` for DSPy.

        Parameters
        ----------
        mode: "with_gt" or "no_gt"; selects the scoring strategy.
        judge_lm: a ``dspy.LM`` (or compatible) used by the LLM-as-judge in
            no-GT mode. Required when ``mode == "no_gt"`` and ``custom_metric``
            is not supplied; ignored otherwise.
        custom_metric: optional override. When provided, this callable is
            returned unchanged so users can plug in their own scoring.
        """
        if custom_metric is not None:
            return custom_metric

        if mode == "with_gt":

            def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
                gt = getattr(example, "answer", "") or ""
                ans = getattr(pred, "answer", "") or ""
                return _token_f1(str(ans), str(gt))

            return _metric

        # no_gt — build an LLM-as-judge using DSPy primitives.
        try:
            import dspy  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on user env
            raise ImportError(
                "DSPy is required to build the no-GT LLM-as-judge metric. "
                "Install with `pip install ragdx[dspy]`."
            ) from exc

        if judge_lm is None:
            raise ValueError(
                "No-GT mode needs an LLM-as-judge. Pass `judge_lm=<dspy.LM-like>` "
                "or provide a `custom_metric`."
            )

        class FaithfulnessJudge(dspy.Signature):
            """Rate whether the answer is grounded in the provided context.

            A faithful answer makes only claims directly supported by the context.
            Score 1.0 = fully grounded, 0.5 = partially, 0.0 = fabricated.
            """

            question: str = dspy.InputField()
            context: str = dspy.InputField()
            answer: str = dspy.InputField()
            score: float = dspy.OutputField(desc="0.0 to 1.0")

        judge = dspy.Predict(FaithfulnessJudge)

        def _metric(example: Any, pred: Any, trace: Any | None = None) -> float:
            ctx = getattr(example, "context", "") or getattr(example, "contexts", "")
            if isinstance(ctx, list):
                ctx = "\n".join(str(c) for c in ctx)
            ans = getattr(pred, "answer", "") or ""
            with dspy.context(lm=judge_lm):
                out = judge(
                    question=getattr(example, "question", ""),
                    context=str(ctx),
                    answer=str(ans),
                )
            try:
                return max(0.0, min(1.0, float(out.score)))
            except (TypeError, ValueError):
                return 0.0

        return _metric

    def build_trainset(
        self,
        records: Sequence[DatasetRecord],
        *,
        mode: GTMode | None = None,
    ) -> list[Any]:
        """Pack ``DatasetRecord``s into ``dspy.Example``s.

        In ``with_gt`` mode each example exposes ``question`` + ``contexts`` as
        inputs and ``answer`` as the GT field. In ``no_gt`` mode no answer
        field is populated and only ``question`` + ``contexts`` are inputs.
        """
        try:
            import dspy  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "DSPy is required to build a trainset. Install with `pip install ragdx[dspy]`."
            ) from exc

        records = list(records)
        chosen_mode: GTMode = mode if mode is not None else gt_mode(records)
        examples: list[Any] = []
        for r in records:
            contexts = list(r.contexts)
            ctx_str = "\n".join(contexts) if contexts else ""
            if chosen_mode == "with_gt":
                examples.append(
                    dspy.Example(
                        question=r.question,
                        contexts=contexts,
                        context=ctx_str,
                        answer=(r.ground_truth or r.answer or ""),
                    ).with_inputs("question", "contexts", "context")
                )
            else:
                examples.append(
                    dspy.Example(
                        question=r.question,
                        contexts=contexts,
                        context=ctx_str,
                    ).with_inputs("question", "contexts", "context")
                )
        return examples

    def build_program(self, signature: Any | None = None, *, kind: str = "chain_of_thought") -> Any:
        """Return a DSPy module wrapping the given (or default RAG) signature."""
        try:
            import dspy  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise ImportError("DSPy is required. Install with `pip install ragdx[dspy]`.") from exc

        if signature is None:

            class DefaultRAGSignature(dspy.Signature):
                """Answer the question using only the retrieved context.

                Be concise. Do not invent facts that are not in the context.
                """

                question: str = dspy.InputField()
                context: str = dspy.InputField(desc="retrieved passages, one per line")
                answer: str = dspy.OutputField()

            signature = DefaultRAGSignature

        builder = {
            "chain_of_thought": dspy.ChainOfThought,
            "predict": dspy.Predict,
        }.get(kind, dspy.ChainOfThought)
        return builder(signature)

    def optimize(
        self,
        records: Sequence[DatasetRecord],
        *,
        program: Any | None = None,
        signature: Any | None = None,
        judge_lm: Any | None = None,
        student_lm: Any | None = None,
        optimizer: str = "MIPROv2",
        optimizer_kwargs: dict[str, Any] | None = None,
        custom_metric: Callable[[Any, Any], float] | None = None,
        mode: GTMode | None = None,
    ) -> dict[str, Any]:
        """Run a DSPy optimiser end-to-end and return the optimised program.

        Parameters
        ----------
        records: DatasetRecord trainset.
        program: pre-built DSPy module. If ``None``, one is built from
            ``signature`` (or the default RAG signature).
        signature: DSPy ``Signature`` class used when ``program`` is None.
        judge_lm: DSPy LM used by the no-GT LLM-as-judge metric.
        student_lm: DSPy LM used by the program being optimised. If provided,
            we call ``dspy.configure(lm=student_lm)`` for the duration of the
            optimisation.
        optimizer: ``"MIPROv2"`` (default), ``"COPRO"``, or ``"BootstrapFewShot"``.
        optimizer_kwargs: passed through to the teleprompter constructor.
        custom_metric: replaces the built-in metric entirely.
        mode: force GT mode; otherwise detected from the trainset.
        """
        try:
            import dspy  # type: ignore[import-not-found]
            from dspy.teleprompt import (  # type: ignore[import-not-found]
                COPRO,
                BootstrapFewShot,
                MIPROv2,
            )
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "DSPy is required for in-process optimisation. "
                "Install with `pip install ragdx[dspy]`."
            ) from exc

        chosen_mode: GTMode = mode if mode is not None else gt_mode(records)
        metric = self.build_metric_function(
            chosen_mode, judge_lm=judge_lm, custom_metric=custom_metric
        )
        prog = program if program is not None else self.build_program(signature)
        trainset = self.build_trainset(records, mode=chosen_mode)

        teleprompters: dict[str, Any] = {
            "MIPROv2": MIPROv2,
            "COPRO": COPRO,
            "BootstrapFewShot": BootstrapFewShot,
        }
        if optimizer not in teleprompters:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}; expected one of {sorted(teleprompters)}."
            )

        kwargs = dict(optimizer_kwargs or {})
        kwargs.setdefault("metric", metric)
        if optimizer == "MIPROv2":
            # MIPROv2 needs ``auto`` or explicit budgets — pick a cheap default.
            kwargs.setdefault("auto", "light")

        teleprompter = teleprompters[optimizer](**kwargs)

        ctx = dspy.context(lm=student_lm) if student_lm is not None else _noop_ctx()
        with ctx:
            optimized = teleprompter.compile(prog, trainset=trainset)

        # Extract instructions + demos so callers don't need to know DSPy internals.
        instructions: dict[str, str] = {}
        demos: dict[str, list[Any]] = {}
        try:
            for name, predictor in optimized.named_predictors():
                sig = getattr(predictor, "signature", None)
                if sig is not None and hasattr(sig, "instructions"):
                    instructions[name] = sig.instructions
                if hasattr(predictor, "demos"):
                    demos[name] = list(predictor.demos)
        except Exception:  # pragma: no cover - depends on dspy version
            pass

        return {
            "optimized_program": optimized,
            "instructions": instructions,
            "demos": demos,
            "gt_mode": chosen_mode,
            "optimizer": optimizer,
            "trainset_size": len(trainset),
        }


# Sentinel context manager when no student LM override is requested.
class _NoopContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def _noop_ctx() -> _NoopContext:
    return _NoopContext()
