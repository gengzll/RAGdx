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
        """Build the winning config from MIPROv2 / COPRO / BootstrapFewShot output.

        ``params`` keys depend on the optimizer that ran:

        * MIPROv2 → both ``system_instruction`` (str) and
          ``few_shot_demos`` (list[dict]).
        * COPRO → only ``system_instruction``; demos absent.
        * BootstrapFewShot → only ``few_shot_demos``; instruction absent.

        Absent keys leave the corresponding field on ``base`` unchanged
        (so a COPRO winner doesn't accidentally wipe base.generator
        .few_shot_demos and vice versa).
        """
        from ragdx.schemas.rag_config import FewShotDemo

        new_instruction = params.get(
            "system_instruction", base.generator.system_instruction,
        )
        demos_payload = params.get("few_shot_demos")
        if demos_payload is None:
            new_demos = list(base.generator.few_shot_demos)
        else:
            # Accept either list[dict] (from raw DSPy output) or
            # list[FewShotDemo] (when callers pre-build them).
            new_demos = [
                d if isinstance(d, FewShotDemo) else FewShotDemo(**d)
                for d in demos_payload
            ]
        return base.with_override(
            generator=GeneratorSpec(
                provider=base.generator.provider,
                model=base.generator.model,
                api_base=base.generator.api_base,
                api_key=base.generator.api_key,
                system_instruction=new_instruction,
                few_shot_demos=new_demos,
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

        # Phase-level checkpoint support (PR6+). Three phases save state:
        # (a) baseline_eval, (b) miprov2 winner, (c) optimised_eval.
        # Resuming from phase X skips phases up to and including X.
        checkpoint = getattr(ctx, "checkpoint", None)
        checkpoint_store = getattr(ctx, "checkpoint_store", None)
        phase_done = (checkpoint.generation_phase if checkpoint else "")
        artifacts = (
            dict(checkpoint.generation_artifacts) if checkpoint else {}
        )

        def _save_phase(phase: str, **extra: Any) -> None:
            """Persist phase progress. No-op when checkpoint isn't wired."""
            if checkpoint is None or checkpoint_store is None:
                return
            artifacts.update(extra)
            checkpoint.generation_artifacts = dict(artifacts)
            checkpoint.generation_phase = phase
            checkpoint.stage_label = ctx.label
            try:
                checkpoint_store.save(checkpoint)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[DSPy/%s] checkpoint save failed (continuing): %s",
                    ctx.label, exc,
                )

        # ----- Phase (a): baseline run -------------------------------
        if phase_done in {"baseline", "miprov2", "re_eval"}:
            logger.info(
                "[DSPy/%s] (a) baseline -- replayed from checkpoint",
                ctx.label,
            )
            # Reconstruct baseline_answered from stored records (no LLM calls).
            baseline_records_payload = artifacts.get(
                "baseline_records_payload", []
            )
            baseline_answered = [
                DatasetRecord(
                    question=r.get("question", ""),
                    ground_truth=r.get("ground_truth"),
                    contexts=list(r.get("contexts") or []),
                    answer=r.get("answer", ""),
                )
                for r in baseline_records_payload
            ]
            baseline_eval = {
                "scores": artifacts.get("baseline_scores", {}),
                "skipped": [],
            }
        else:
            logger.info("[DSPy/%s] (a) baseline run", ctx.label)
            baseline_answered = _run_program(baseline_program)
            baseline_eval = _evaluate_with_ragas(
                baseline_answered,
                runtime.ragas_judge,
                runtime.ragas_embeddings,
                ctx.metrics,
                run_config=runtime.ragas_run_config,
            )
            _save_phase(
                "baseline",
                baseline_scores=dict(baseline_eval.get("scores", {}) or {}),
                baseline_records_payload=[
                    {
                        "question": r.question,
                        "ground_truth": r.ground_truth,
                        "contexts": list(r.contexts or []),
                        "answer": r.answer or "",
                    }
                    for r in baseline_answered
                ],
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

        # ----- Phase (b): MIPROv2 ------------------------------------
        # MIPROv2 is monolithic from the outside -- DSPy doesn't expose
        # ``study.add_trial`` resumption points. So on resume from
        # "miprov2" we rebuild the winning program from the stored
        # ``winning_instruction`` text (cheap) and skip MIPROv2 entirely.
        if phase_done in {"miprov2", "re_eval"}:
            logger.info(
                "[DSPy/%s] (b) MIPROv2 -- resumed from checkpoint, "
                "skipping ~%d trial(s)",
                ctx.label,
                len(artifacts.get("trial_scores", [])),
            )
            winning_instruction_ck = artifacts.get("winning_instruction", "")
            # Rebuild a fresh program at the winning instruction. No LLM
            # calls -- the program object is just a thin wrapper around
            # a Signature.
            optimized_program = adapter.build_program(
                instruction=winning_instruction_ck or runtime.system_instruction,
            )
            opt_result = {
                "optimized_program": optimized_program,
                "instructions": dict(
                    artifacts.get("opt_instructions", {})
                ) or {"predict": winning_instruction_ck or ""},
                "demos": {n: list(d) for n, d in (artifacts.get("opt_demos") or {}).items()},
                "gt_mode": ctx.label,
                "optimizer": "MIPROv2",
                "trainset_size": len(records_with_ctxs),
            }
            proposed_capture = {
                k: list(v)
                for k, v in (artifacts.get("proposed_instructions_by_predictor") or {}).items()
            }

            class _CapturePlaceholder:
                """Stand-in for the live ``_MIPROTrialScoreCapture`` so
                the per-trial fields downstream still resolve when we
                resumed past phase (b)."""

                scores_so_far = list(artifacts.get("trial_scores") or [])
                best_scores = list(artifacts.get("best_score_progression") or [])
                trials = list(artifacts.get("trial_log") or [])

            capture = _CapturePlaceholder()
            _patch_installed = False
        else:
            logger.info("[DSPy/%s] (b) MIPROv2 optimisation", ctx.label)
            capture = _MIPROTrialScoreCapture()
            capture.setLevel(logging.INFO)
            mipro_logger = logging.getLogger(
                "dspy.teleprompt.mipro_optimizer_v2"
            )
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
                    try:
                        proposed_capture.update({
                            str(k): [str(s) for s in v]
                            for k, v in dict(result).items()
                        })
                    except Exception:  # pragma: no cover
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

            # Map ctx.dspy_optimizer to DSPyAdapter's optimizer string
            # + the right optimizer_kwargs for each algorithm.
            _opt_choice = getattr(ctx, "dspy_optimizer", "mipro")
            _dspy_compile_kwargs: dict[str, Any] = {}
            if _opt_choice == "mipro":
                _dspy_opt_str = "MIPROv2"
                _dspy_opt_kwargs = {
                    "auto": getattr(ctx, "mipro_auto", "light"),
                    "num_threads": runtime.llm_max_concurrent,
                }
            elif _opt_choice == "copro":
                # COPRO: instruction-only iterative rewrite.
                # breadth=candidates per round; depth=rounds; ~30-50
                # LLM calls total for the defaults.
                _dspy_opt_str = "COPRO"
                _dspy_opt_kwargs = {
                    "breadth": 10,
                    "depth": 3,
                    "init_temperature": 1.4,
                    "track_stats": True,
                }
                # COPRO requires eval_kwargs at compile time. Pass the
                # runtime's concurrency budget so it parallelises like
                # MIPROv2 does.
                _dspy_compile_kwargs = {
                    "eval_kwargs": {
                        "display_progress": False,
                        "num_threads": runtime.llm_max_concurrent,
                    },
                }
            elif _opt_choice == "bootstrap_fewshot":
                # BootstrapFewShot: demo-only generation. Runs the
                # seed program on the trainset, keeps successful
                # (q, a) pairs as few-shot demos. No instruction change.
                _dspy_opt_str = "BootstrapFewShot"
                _dspy_opt_kwargs = {
                    "max_bootstrapped_demos": 4,
                    "max_labeled_demos": 4,
                    "max_rounds": 1,
                }
            elif _opt_choice == "gepa":
                # GEPA (Genetic-Pareto reflective evolution, paper
                # arXiv:2507.19457): uses LLM reflection on execution
                # traces to evolve text components (instructions).
                # Marked experimental in DSPy.
                #
                # GEPA's ``auto`` setting assumes a fast LM (~30 calls
                # per second) and budgets ~392 rollouts for "light".
                # On slow / rate-limited LMs like GLM-4-Flash this
                # turns into a 16+ hour run -- so we translate
                # ``--mipro-auto`` into an explicit
                # ``max_metric_calls`` instead, with budgets sized
                # for production iteration speed (not paper-reproduction
                # rigor):
                #
                #   light  -> 30  calls (~3-5 reflections, ~15-30 min)
                #   medium -> 100 calls (~10-15 reflections, ~1 hour)
                #   heavy  -> 300 calls (~30-50 reflections, ~3 hours)
                #
                # Users who want paper-faithful settings can hand-edit
                # _dspy_opt_kwargs in code or use a custom dspy script.
                _gepa_budget_map = {
                    "light": 30, "medium": 100, "heavy": 300,
                }
                _mipro_auto = getattr(ctx, "mipro_auto", "light")
                _dspy_opt_str = "GEPA"
                _dspy_opt_kwargs = {
                    "max_metric_calls": _gepa_budget_map.get(_mipro_auto, 30),
                    # reflection_lm: the (typically stronger) LM that
                    # GEPA uses to propose new instructions from
                    # execution traces. We share the student LM by
                    # default; for production use a stronger model
                    # via a dedicated factory hook later.
                    "reflection_lm": runtime.dspy_lm,
                    "num_threads": runtime.llm_max_concurrent,
                    "track_stats": True,
                    # Our ragas composite tops out at ~2.8 (weighted
                    # sum), so the [0, 1] default convergence range
                    # GEPA expects doesn't match. We just turn off the
                    # "skip if score == perfect_score" shortcut so
                    # GEPA runs the full budget regardless.
                    "skip_perfect_score": False,
                }
            else:  # pragma: no cover - validated at CLI layer
                raise ValueError(
                    f"Unknown dspy_optimizer={_opt_choice!r}; expected "
                    "mipro / copro / bootstrap_fewshot / gepa."
                )
            logger.info(
                "[DSPy/%s] running %s with kwargs=%s",
                ctx.label, _dspy_opt_str, _dspy_opt_kwargs,
            )
            try:
                opt_result = adapter.optimize(
                    records_with_ctxs,
                    program=baseline_program,
                    student_lm=runtime.dspy_lm,
                    judge_lm=runtime.dspy_lm,
                    optimizer=_dspy_opt_str,
                    optimizer_kwargs=_dspy_opt_kwargs,
                    compile_kwargs=_dspy_compile_kwargs,
                    custom_metric=custom_metric,
                )
            finally:
                mipro_logger.removeHandler(capture)
                mipro_logger.setLevel(prior)
                if _patch_installed:
                    _mipro_mod.MIPROv2._propose_instructions = _orig_propose

            # Save phase (b) outcome so a phase-(c) crash doesn't lose
            # the multi-trial MIPROv2 result.
            _winner_for_save = next(
                iter(dict(opt_result["instructions"]).values()), ""
            )
            _save_phase(
                "miprov2",
                winning_instruction=_winner_for_save,
                opt_instructions=dict(opt_result["instructions"]),
                opt_demos={
                    n: [dict(d) for d in (demos or [])]
                    for n, demos in (opt_result.get("demos") or {}).items()
                },
                proposed_instructions_by_predictor=dict(proposed_capture),
                trial_scores=list(capture.scores_so_far),
                best_score_progression=list(capture.best_scores),
                trial_log=list(capture.trials),
            )

        # ----- Phase (c): optimised re-run --------------------------
        if phase_done == "re_eval":
            logger.info(
                "[DSPy/%s] (c) optimised re-run -- replayed from checkpoint",
                ctx.label,
            )
            optimised_records_payload = artifacts.get(
                "optimised_records_payload", []
            )
            optimised_answered = [
                DatasetRecord(
                    question=r.get("question", ""),
                    ground_truth=r.get("ground_truth"),
                    contexts=list(r.get("contexts") or []),
                    answer=r.get("answer", ""),
                )
                for r in optimised_records_payload
            ]
            opt_eval = {
                "scores": artifacts.get("optimised_scores", {}),
                "skipped": [],
            }
        else:
            logger.info("[DSPy/%s] (c) optimised re-run", ctx.label)
            optimised_answered = _run_program(opt_result["optimized_program"])
            opt_eval = _evaluate_with_ragas(
                optimised_answered,
                runtime.ragas_judge,
                runtime.ragas_embeddings,
                ctx.metrics,
                run_config=runtime.ragas_run_config,
            )
            _save_phase(
                "re_eval",
                optimised_scores=dict(opt_eval.get("scores", {}) or {}),
                optimised_records_payload=[
                    {
                        "question": r.question,
                        "ground_truth": r.ground_truth,
                        "contexts": list(r.contexts or []),
                        "answer": r.answer or "",
                    }
                    for r in optimised_answered
                ],
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

        # The winner's "instruction" + "demos" come from opt_result.
        # Pick the first predictor (typical for single-step RAG).
        opt_instructions = dict(opt_result["instructions"])
        winning_instruction = next(iter(opt_instructions.values()), None)
        opt_demos_raw = (opt_result.get("demos") or {})
        winning_demos_raw = next(
            iter(opt_demos_raw.values()), []
        ) if opt_demos_raw else []

        # Convert DSPy ``Example`` objects to ragdx FewShotDemo dicts.
        # DSPy stores demos as ``dspy.Example`` which behaves like a
        # dict (``.toDict()`` or ``dict(example)``). We only keep the
        # fields ragdx's :class:`FewShotDemo` knows about.
        def _demo_to_dict(d: Any) -> dict:
            try:
                payload = dict(d)
            except (TypeError, ValueError):
                payload = {}
                for attr in ("question", "answer", "reasoning", "context"):
                    val = getattr(d, attr, None)
                    if val is not None:
                        payload[attr] = val
            return {
                "question": str(payload.get("question") or ""),
                "answer": str(payload.get("answer") or ""),
                "reasoning": (
                    str(payload["reasoning"]) if payload.get("reasoning") else None
                ),
                "context": (
                    str(payload["context"]) if payload.get("context") else None
                ),
            }

        winning_demos_dicts = [
            _demo_to_dict(d) for d in winning_demos_raw
        ] if winning_demos_raw else []
        # Filter out demos missing question/answer (defensive).
        winning_demos_dicts = [
            d for d in winning_demos_dicts if d["question"] and d["answer"]
        ]

        # Gate which fields land in ``best_config`` based on the
        # optimizer that actually ran. PR7+ deliberate design:
        # COPRO writes instruction only; BootstrapFewShot writes
        # demos only; MIPROv2 writes both.
        _opt_choice = getattr(ctx, "dspy_optimizer", "mipro")
        if _opt_choice in {"copro", "gepa"}:
            # Both COPRO and GEPA evolve only the instruction text;
            # demos are not part of their search space.
            best_params_payload = {"system_instruction": winning_instruction}
        elif _opt_choice == "bootstrap_fewshot":
            best_params_payload = {"few_shot_demos": winning_demos_dicts}
        else:  # mipro (default) or unrecognised fallback
            best_params_payload = {
                "system_instruction": winning_instruction,
                "few_shot_demos": winning_demos_dicts,
            }
        best_config = self.build_trial_config(
            ctx.base_config, best_params_payload,
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
            "dspy_optimizer_used": _opt_choice,
            "winning_demos": winning_demos_dicts,
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
            best_params=dict(best_params_payload),
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
