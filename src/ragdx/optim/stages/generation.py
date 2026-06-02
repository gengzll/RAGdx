""":class:`GenerationOptimizer` -- optimize only the generator prompt.

Unlike the BO-driven stages, this one delegates to DSPy MIPROv2 to
evolve the prompt instruction + few-shot demos at the BO winner's
retrieval configuration. ``StageContext.records`` is expected to
already carry ``contexts`` (retrieved with the base config's
retriever); this stage doesn't re-retrieve.

The result's ``extras`` carry the full before/after artifacts the
experiment dashboard renders: baseline vs optimized instructions,
demo lists, per-record answer pairs, MIPROv2 inner-loop trial
scores. Same shape as the pre-PR3 ``_dspy_before_after`` return so
the bundle's ``dspy_a_b`` section is byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ragdx.optim.dspy_adapter import DSPyAdapter
from ragdx.optim.stages.base import StageContext, StageOptimizer, StageResult
from ragdx.schemas.models import DatasetRecord
from ragdx.schemas.rag_config import GeneratorSpec, RAGConfig
from ragdx.utils.logging import get_logger

logger = get_logger(__name__)


class GenerationOptimizer(StageOptimizer):
    """Optimize only the generator prompt via DSPy MIPROv2."""

    name: ClassVar[str] = "generation"

    # The BO-loop primitives don't apply here; we override optimize().
    def search_space(self, ctx: StageContext) -> dict[str, list]:
        # Documentation only: MIPROv2's "search space" is the set of
        # instruction candidates + few-shot demo subsets it discovers
        # at run time, not a discrete grid we can enumerate.
        return {"instruction_candidates": ["MIPROv2-discovered"]}

    def build_trial_config(
        self, base: RAGConfig, params: dict[str, Any]
    ) -> RAGConfig:
        # Override the generator's system_instruction with whatever
        # MIPROv2 returned.
        return base.with_override(
            generator=GeneratorSpec(
                provider=base.generator.provider,
                model=base.generator.model,
                api_base=base.generator.api_base,
                api_key=base.generator.api_key,
                system_instruction=params.get("system_instruction"),
                temperature=base.generator.temperature,
                max_tokens=base.generator.max_tokens,
                timeout=base.generator.timeout,
            ),
        )

    def optimize(self, ctx: StageContext) -> StageResult:
        """Run DSPy MIPROv2 baseline + optimization + re-eval.

        ``ctx.records`` must already carry retrieved contexts.
        ``ctx.runtime`` provides ``dspy_lm``, ``ragas_judge``,
        ``ragas_embeddings``, ``ragas_run_config``,
        ``system_instruction``, ``llm_max_concurrent``.
        """
        # Lazy imports: dspy + the MIPRO log handler aren't needed if
        # no caller invokes this stage.
        import dspy

        from ragdx.experiments import _evaluate_with_ragas, _MIPROTrialScoreCapture

        runtime = ctx.runtime
        records_with_ctxs = ctx.records

        dspy.configure(lm=runtime.dspy_lm)
        adapter = DSPyAdapter()
        # Hand the same system instruction to the DSPy baseline so its
        # before/after comparison shares the exact prompt the BO stage used.
        baseline_program = adapter.build_program(
            instruction=runtime.system_instruction,
        )

        baseline_instructions, baseline_demos = _extract_instructions_demos(
            baseline_program
        )

        def _run_program(program):
            out = []
            for r in records_with_ctxs:
                ctx_str = "\n".join(r.contexts) if r.contexts else ""
                try:
                    with dspy.context(lm=runtime.dspy_lm):
                        pred = program(question=r.question, context=ctx_str)
                    ans = str(getattr(pred, "answer", "") or "")
                except Exception as e:  # pragma: no cover - live LLM
                    ans = f"<error: {e}>"
                out.append(
                    DatasetRecord(
                        question=r.question,
                        ground_truth=r.ground_truth,
                        contexts=r.contexts,
                        answer=ans,
                    )
                )
            return out

        logger.info("[DSPy/%s] (a) baseline run", ctx.label)
        baseline_answered = _run_program(baseline_program)
        baseline_eval = _evaluate_with_ragas(
            baseline_answered,
            runtime.ragas_judge,
            runtime.ragas_embeddings,
            ctx.metrics,
            run_config=runtime.ragas_run_config,
        )

        # ----- Pick the inner-loop DSPy metric (PR6+) -----------------
        # Default: ragas composite for no-GT (where ``llm_judge``
        # saturates), token-F1 for with-GT (where it has a discriminative
        # GT to compare against). Users override via ``--dspy-metric``.
        dspy_metric_choice = getattr(ctx, "dspy_metric", "auto")
        if dspy_metric_choice == "auto":
            dspy_metric_choice = "ragas" if ctx.label == "no_gt" else "token_f1"
        custom_metric = None
        if dspy_metric_choice == "ragas":
            from ragdx.optim._ragas_dspy_metric import make_ragas_metric
            try:
                custom_metric = make_ragas_metric(
                    judge=runtime.ragas_judge,
                    embeddings=runtime.ragas_embeddings,
                )
                logger.info(
                    "[DSPy/%s] using ragas composite (context_precision + "
                    "faithfulness + answer_relevancy) as inner-loop metric",
                    ctx.label,
                )
            except ImportError:
                logger.warning(
                    "[DSPy/%s] ragas not available; falling back to "
                    "DSPy's llm_judge metric (saturates on permissive LMs)",
                    ctx.label,
                )
                dspy_metric_choice = "llm_judge"
        elif dspy_metric_choice == "token_f1":
            # DSPyAdapter.build_metric_function handles with_gt mode
            # natively. Nothing to pre-build.
            logger.info(
                "[DSPy/%s] using token-F1 vs GT as inner-loop metric",
                ctx.label,
            )
        else:
            logger.info(
                "[DSPy/%s] using DSPy llm_judge (faithfulness) as inner-loop metric",
                ctx.label,
            )

        logger.info("[DSPy/%s] (b) MIPROv2 optimisation", ctx.label)
        capture = _MIPROTrialScoreCapture()
        capture.setLevel(logging.INFO)
        mipro_logger = logging.getLogger("dspy.teleprompt.mipro_optimizer_v2")
        mipro_logger.addHandler(capture)
        prior = mipro_logger.level
        mipro_logger.setLevel(logging.INFO)

        # Capture the proposed candidate instructions. MIPROv2's
        # ``_propose_instructions`` returns a
        # ``{predictor_name: [instruction_str, ...]}`` dict, but it's a
        # local variable inside ``compile`` and not stored on self -- so
        # we monkey-patch the method for the duration of the call,
        # restore after. This lets the visualization show "what prompts
        # did MIPROv2 actually try", not just the final winner.
        proposed_capture: dict[str, list[str]] = {}
        try:
            import dspy.teleprompt.mipro_optimizer_v2 as _mipro_mod
            _orig_propose = _mipro_mod.MIPROv2._propose_instructions

            def _patched_propose(self, *args, **kwargs):
                result = _orig_propose(self, *args, **kwargs)
                # ``result`` is ``{predictor_name: [str, ...]}``. Copy
                # defensively so future MIPROv2 internal mutations don't
                # change what we report.
                try:
                    proposed_capture.update({
                        str(k): [str(s) for s in v] for k, v in dict(result).items()
                    })
                except Exception:  # pragma: no cover - defensive
                    pass
                return result

            _mipro_mod.MIPROv2._propose_instructions = _patched_propose
            _patch_installed = True
        except Exception as exc:  # pragma: no cover - dspy API drift
            logger.warning(
                "[DSPy/%s] could not patch MIPROv2._propose_instructions: %s",
                ctx.label, exc,
            )
            _patch_installed = False

        try:
            opt_result = adapter.optimize(
                records_with_ctxs,
                # Seed MIPROv2 with the same program the baseline used so
                # the user's system_instruction is what MIPROv2 starts
                # from (Instruction 0 in its candidate set). Without this
                # the optimizer silently rebuilds a default-signature
                # program and any system_instruction override is lost.
                program=baseline_program,
                student_lm=runtime.dspy_lm,
                judge_lm=runtime.dspy_lm,
                optimizer="MIPROv2",
                # PR6+: surface the search budget as ``mipro_auto`` via
                # StageContext + --mipro-auto CLI flag.
                # ``light`` (~3 candidates) is the default; bump to
                # ``medium`` (~10-20) when the seed needs serious
                # competition, or ``heavy`` (~30+) for a deep sweep.
                optimizer_kwargs={
                    "auto": getattr(ctx, "mipro_auto", "light"),
                    "num_threads": runtime.llm_max_concurrent,
                },
                # PR6+: when the user (or the default rule) picks
                # ``ragas``, swap DSPy's saturating llm_judge metric for
                # the ragas composite. ``custom_metric=None`` lets DSPy
                # fall back to its mode-based default.
                custom_metric=custom_metric,
            )
        finally:
            mipro_logger.removeHandler(capture)
            mipro_logger.setLevel(prior)
            if _patch_installed:
                _mipro_mod.MIPROv2._propose_instructions = _orig_propose

        logger.info("[DSPy/%s] (c) optimised re-run", ctx.label)
        optimised_answered = _run_program(opt_result["optimized_program"])
        opt_eval = _evaluate_with_ragas(
            optimised_answered,
            runtime.ragas_judge,
            runtime.ragas_embeddings,
            ctx.metrics,
            run_config=runtime.ragas_run_config,
        )

        baseline_scores = baseline_eval.get("scores", {}) or {}
        optimised_scores = opt_eval.get("scores", {}) or {}
        composite_baseline = ctx.objective.evaluate(baseline_scores)
        composite_optimized = ctx.objective.evaluate(optimised_scores)
        delta = {
            m: (
                optimised_scores.get(m, float("nan"))
                - baseline_scores.get(m, float("nan"))
            )
            for m in sorted(set(baseline_scores) | set(optimised_scores))
        }
        records_pairs = []
        for b, o in zip(baseline_answered, optimised_answered, strict=False):
            records_pairs.append({
                "question": b.question,
                "ground_truth": b.ground_truth,
                "contexts": list(b.contexts or []),
                "baseline_answer": b.answer or "",
                "optimized_answer": o.answer or "",
            })

        # MIPROv2's "winner" prompt is the system_instruction we'll
        # report. Pick the first predictor's instruction (typically the
        # only one in a single-step RAG program).
        opt_instructions = dict(opt_result["instructions"])
        winning_instruction = next(iter(opt_instructions.values()), None)
        best_config = self.build_trial_config(
            ctx.base_config, {"system_instruction": winning_instruction}
        )

        extras = {
            "baseline_scores": baseline_scores,
            "optimized_scores": optimised_scores,
            "delta": delta,
            "composite": {
                "objective_spec": ctx.objective.to_dict(),
                "baseline": composite_baseline,
                "optimized": composite_optimized,
                "delta": composite_optimized["score"] - composite_baseline["score"],
            },
            "baseline_sample_answers": [a.answer for a in baseline_answered],
            "optimized_sample_answers": [a.answer for a in optimised_answered],
            "records": records_pairs,
            "baseline_instructions": baseline_instructions,
            "baseline_demos": baseline_demos,
            "instructions": opt_instructions,
            "demos": {
                n: [dict(d) for d in demos]
                for n, demos in opt_result["demos"].items()
            },
            "trial_scores": list(capture.scores_so_far),
            "best_score_progression": list(capture.best_scores),
            # Configuration knobs the user picked for this run --
            # recorded so the bundle is self-describing and the HTML
            # report can show "this was run at auto=medium with
            # metric=ragas, hence the prompts vary widely / scores are
            # discriminative".
            "mipro_auto": getattr(ctx, "mipro_auto", "light"),
            "dspy_metric_used": dspy_metric_choice,
            # Every candidate instruction MIPROv2 proposed, keyed by
            # predictor name. The winning instruction (in ``instructions``
            # above) is one of these. Empty {} when the monkey-patch
            # failed (DSPy API drift) -- the rest of the bundle is
            # unaffected.
            "proposed_instructions_by_predictor": dict(proposed_capture),
            # Per-trial log: trial number, score, kind (default /
            # minibatch / full), and chosen parameters string
            # ``"{'predict': (instruction_idx, demo_idx)}"`` parsed from
            # MIPROv2's logger. The HTML renderer joins this with
            # ``proposed_instructions_by_predictor`` to display
            # "trial 3 → instruction text X → score 0.78".
            "trial_log": list(capture.trials),
            "gt_mode": opt_result["gt_mode"],
            "optimizer": opt_result["optimizer"],
            "trainset_size": opt_result["trainset_size"],
        }
        return StageResult(
            stage_name=self.name,
            search_space=self.search_space(ctx),
            trials=[],  # MIPROv2 trials live in extras["trial_scores"]
            best_params={"system_instruction": winning_instruction},
            best_config=best_config,
            best_composite=composite_optimized["score"],
            objective_spec=ctx.objective.to_dict(),
            n_init=0,
            max_trials=len(capture.scores_so_far),
            extras=extras,
        )


def _extract_instructions_demos(program) -> tuple[dict[str, str], dict[str, list]]:
    """Pull instructions + few-shot demos out of a DSPy program.

    Returns ``({predictor_name: instruction}, {predictor_name: demos})``.
    Survives DSPy API drift via a try/except (logged via the caller).
    """
    instructions: dict[str, str] = {}
    demos: dict[str, list] = {}
    try:
        for name, predictor in program.named_predictors():
            sig = getattr(predictor, "signature", None)
            if sig is not None and hasattr(sig, "instructions"):
                instructions[name] = sig.instructions or ""
            if hasattr(predictor, "demos"):
                demos[name] = [dict(d) for d in predictor.demos]
    except Exception:  # pragma: no cover - DSPy API surface drift
        pass
    return instructions, demos


__all__ = ["GenerationOptimizer"]
