# Diagnosis and Optimization Logic

## 1. Diagnosis pipeline

Diagnosis begins with an `EvaluationResult` and proceeds through one or more of the following layers:

1. rule-based analysis
2. causal graph construction and propagation
3. optional LLM refinement
4. optional rule plus LLM summary mode

## 2. Rule-based diagnosis

The rule-based analyzer uses metric levels, gaps, traces, and feedback to form hypotheses such as:

- corpus chunking defect
- retrieval recall defect
- retrieval precision defect
- context packing defect
- grounding defect
- citation binding defect
- judge or metric instability
- distribution shift

## 3. Layered causal graph

The diagnosis layer includes an explicit `CausalGraph` consisting of:

- `nodes`: `CausalSignal`
- `edges`: `CausalEdge`

The graph stores:

- priors
- posteriors
- evidence
- weighted causal relations

This supports more structured diagnosis than a flat list of heuristics.

## 4. Learned priors

The local store can maintain causal priors in:

- `.ragdx/causal/priors.json`

These priors can be updated from diagnosis reports and feedback, allowing some persistence across runs.

## 5. LLM diagnosis

When enabled, LLM diagnosis refines or summarizes the rule-based diagnosis. It should be understood as:

- a reasoning layer on top of structured evidence
- not a replacement for deterministic checks

The recommended production pattern is:

- run rule-based diagnosis by default
- use LLM diagnosis for ambiguous cases or richer summaries

## 6. Planning logic

The planner converts diagnosis into an `OptimizationPlan` with a set of experiments. Important planning ideas include:

- baseline-relative targets
- metric direction awareness
- stage-aware planning
- objective weights distinct from metric targets
- hard or near-hard constraints
- optional LLM planner refinement

## 7. Stages

The plan can include experiments in the following stages:

- `corpus`
- `retrieval`
- `generation`
- `orchestration`
- `joint`

Stage choice depends on the diagnosis.

## 8. Objectives vs targets vs constraints

This distinction is critical.

### Objective weights

Stored in `objectives` or `parameters.objective_weights`.

These are trade-off coefficients, not metric targets.

### Target thresholds and target specs

Stored in:

- `parameters.target_thresholds`
- `parameters.target_specs`

These specify what the optimizer should aim for relative to baseline.

### Constraint bounds

Stored in:

- `constraints`
- `parameters.constraint_bounds`

These define feasibility limits such as maximum hallucination or latency.

## 9. Multi-objective optimization

The optimizer handles multiple goals simultaneously, such as:

- answer correctness
- context recall
- context precision
- citation accuracy
- latency
- cost

This is important because real RAG changes often improve one dimension while hurting another.

## 10. Feasibility and Pareto logic

Each trial may have:

- `feasible`
- `constraint_violations`
- `feasibility_penalty`

Session state may include:

- `pareto_front_ids`
- `feasible_pareto_front_ids`
- `hypervolume`
- `feasible_hypervolume`

This allows the tool to separate globally strong candidates from candidates that are actually acceptable under constraints.

## 11. Search strategies

### Bayesian

A model-based search approach suitable when the search budget is limited and trial evaluations are expensive.

### Pareto evolutionary

A broader search approach suitable when trade-offs are complex and the search space is highly mixed or discrete.

## 12. Heavy BO backend

If configured, the system can attempt to use a heavier Bayesian optimization backend such as Ax.

## 13. LLM planner

The optional LLM planner takes into account:

- baseline metrics
- diagnosis summary
- metric gaps
- heuristic plan proposal

It can refine:

- objective weights
- target thresholds
- search-space focus
- trial budget
- rationale notes


## 14 Details on Rules
**How the diagnosis rules work**

The engine first computes a metric gap for each metric: for normal metrics, `gap = threshold - value`; for lower-is-better metrics like `hallucination` and `noise_sensitivity`, `gap = value - threshold`; anything below 0 is clipped to 0 ([thresholds.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/core/thresholds.py:31>), [root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:113>)). Default targets are fairly strict: `context_precision=0.80`, `context_recall=0.80`, `faithfulness=0.90`, `hallucination=0.10`, `citation_accuracy=0.85`, `answer_correctness=0.85`, etc.

There are really 2 rule layers:

1. Direct hypothesis rules in `analyze()`
   These generate the human-readable diagnosis items ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:333>)):

- `context_recall` low while `context_precision` is okay:
  diagnosis = evidence miss / retrieval recall problem.
- `context_precision` low:
  diagnosis = retrieval noise / weak ranking.
- `faithfulness` low while recall is at least moderate:
  diagnosis = generation is not grounding well on the evidence.
- `noise_sensitivity` high or `hallucination` high:
  diagnosis = fragile generation / unsupported reasoning.
- `answer_correctness` low but nothing else is clearly dominant:
  diagnosis = inconclusive pipeline-level issue.
- `citation_accuracy` low:
  diagnosis = citation binding is weaker than answer generation.

2. Causal node rules in `_node_evidence()`
   These score 8 root-cause nodes and then propagate influence through a causal graph ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:220>), [root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:296>)):

- `corpus_chunking_defect`
  Triggered by recall gap, `document_structure_preserved=False`, or too few retrieved chunks.
- `retrieval_recall_defect`
  Triggered mainly by low `context_recall`, especially if precision is still good.
- `retrieval_precision_defect`
  Triggered by low `context_precision`, high hallucination, weak reranking, or too many chunks without rerank.
- `context_packing_defect`
  Triggered by low `context_utilization`, too many chunks, or enough evidence existing but noise still dominating.
- `grounding_defect`
  Triggered by low `faithfulness`, low `answer_correctness`, high hallucination, hallucination feedback, sparse verification spans, or high noise sensitivity.
- `citation_binding_defect`
  Triggered by low `citation_accuracy`, especially when answer quality is okay but citation mapping is still weak.
- `judge_or_metric_instability`
  Triggered by low evaluator agreement or `judge_prompt_changed`.
- `distribution_shift`
  Triggered by `dataset_shift` / `domain_shift`, high negative feedback, answerless responses, or some trace anomalies.

**Important implementation details**

The diagnosis is not just simple `if/else`. Each causal node starts from a prior probability, adds evidence deltas, converts that through logit/sigmoid math, and then receives upstream influence from parent causes over 3 propagation passes ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:292>), [root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:304>)). Example: `retrieval_precision_defect -> grounding_defect` and `grounding_defect -> citation_binding_defect`.

The priors are adaptive. They can be loaded from `.ragdx/causal/priors.json`, adjusted by metadata, feedback, and traces, then written back after diagnosis ([07-optimization-and-diagnosis.md](</d:/Codes/rag_diagnosis_lib_v2/docs/07-optimization-and-diagnosis.md:41>), [run_store.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/storage/run_store.py:174>), [root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:179>)). So the system “learns” modestly from past runs.

** Rule map**
Below is the concrete rule map for the diagnosis logic in [root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:333>), tied back to the diagnosis doc [07-optimization-and-diagnosis.md](</d:/Codes/rag_diagnosis_lib_v2/docs/07-optimization-and-diagnosis.md:12>).

| Rule / node | Main trigger | What it means | Recommended action | Optimization candidate |
|---|---|---|---|---|
| `evidence miss despite acceptable retrieval precision` | `context_recall < 0.80` and `context_precision >= 0.80` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:353>)) | Retriever is finding mostly clean results, but not enough relevant evidence | Increase recall, tune chunk size/overlap, inspect query rewriting and filters | `autorag_pipeline_search`, `corpus_chunking_search` |
| `retrieval noise or weak ranking quality` | `context_precision < 0.80` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:359>)) | Retriever is bringing in too much irrelevant text, which contaminates later stages | Add/tune reranker, reduce top-k, use metadata or section-aware filtering | `autorag_pipeline_search` |
| `generator is not grounding sufficiently on retrieved evidence` | `faithfulness < 0.90` and `context_recall >= 0.7` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:364>)) | Evidence exists, but the answer generator is not using it reliably | Use evidence-backed prompt templates, citation-first answers, verification/compression stages | `dspy_prompt_optimization` |
| `answer is fragile under distractors or unsupported reasoning` | `noise_sensitivity > 0.20` or `hallucination > 0.10` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:370>)) | The model overweights noisy context or invents unsupported claims | Constrain answers to cited evidence, improve packing, add verifier | `dspy_prompt_optimization`, `joint_ablation_eval` |
| `end-to-end quality is weak but component-level signals are inconclusive` | `answer_correctness < 0.85` and no stronger hypothesis was created ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:375>)) | Overall system is underperforming, but metrics do not isolate one bottleneck cleanly | Run controlled ablations across retriever, reranker, and prompt | `joint_ablation_eval` |
| `citation mapping is weaker than answer generation` | `citation_accuracy < 0.85` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:380>)) | The answer may be decent, but references are not attached correctly | Enforce sentence-level citations, return passage IDs from synthesis | none explicitly added here |

**Causal-node rules**

These are lower-level root-cause signals scored in `_node_evidence()` and then propagated through the causal graph.

| Causal node | Evidence used | Interpretation | Default action |
|---|---|---|---|
| `corpus_chunking_defect` | Recall gap, `document_structure_preserved=False`, low average retrieved chunks ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:239>)) | Ingestion/chunking is harming evidence availability | Parser and chunking search before retriever tuning |
| `retrieval_recall_defect` | Low `context_recall`, especially when precision is still acceptable ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:245>)) | Missing relevant evidence | Expand recall with hybrid retrieval, rewriting, larger candidate pools |
| `retrieval_precision_defect` | Low `context_precision`, high hallucination, low rerank coverage ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:249>)) | Retrieved evidence is noisy | Strengthen reranking and filtering |
| `context_packing_defect` | Low `context_utilization`, many chunks, enough evidence but poor use ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:255>)) | Good evidence may be present but poorly ordered or crowded out | Tune packing order, context budget, section-aware packing |
| `grounding_defect` | Low `faithfulness`, low `answer_correctness`, high hallucination, hallucination feedback, low verify coverage ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:261>)) | Answer generation is not staying anchored to evidence | Use grounded templates, verification, citation-first prompting |
| `citation_binding_defect` | Low `citation_accuracy`, especially when faithfulness is fine ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:270>)) | Claims and citations are not linked precisely | Bind citations at sentence/claim level |
| `judge_or_metric_instability` | Low evaluator agreement or `judge_prompt_changed` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:274>)) | The measurement system may be unreliable | Audit judges, compare to human labels, recalibrate |
| `distribution_shift` | `dataset_shift`/`domain_shift`, high negative feedback, answerless responses ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:279>)) | Traffic/data has moved away from benchmark assumptions | Cluster failing traffic, review shift, expand benchmark |

**Trace-based rules**

Trace data can increase suspicion for some nodes even if metric gaps alone are weak ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:195>)):

- `chunks <= 2`: boosts `corpus_chunking_defect` and `retrieval_recall_defect`
- `chunks >= 9` and no rerank span: boosts `retrieval_precision_defect`
- `chunks >= 9`: boosts `context_packing_defect`
- answer has `0` citations: boosts `citation_binding_defect`
- latency `> 6500 ms`: slightly boosts `distribution_shift`
- no verify span: slightly boosts `grounding_defect`

**Feedback and prior-learning rules**

The diagnosis is adaptive, not fixed:

- hallucination feedback rate `> 0.15`: raises prior for `grounding_defect`
- negative feedback rate `> 0.35`: raises prior for `distribution_shift`
- average chunks `< 3.5`: raises prior for `corpus_chunking_defect`
- metadata can directly override priors via `causal_prior_updates`
- updated priors are persisted in `.ragdx/causal/priors.json` ([root_cause.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/engines/root_cause.py:179>), [run_store.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/storage/run_store.py:194>))

**How to read the output**

The report contains two related but different things:

- `hypotheses`: direct, user-facing diagnoses from the top-level rules
- `causal_signals`: probabilistic root-cause nodes after evidence scoring and graph propagation

So a report can say the main bottleneck is low recall, while also showing a high posterior for `corpus_chunking_defect` because chunking may be the upstream cause of that low recall.


## 15. Cheat Sheet

- `context_recall` low, `context_precision` okay:
  Do retrieval recall work first.
  Check chunking, overlap, query rewriting, metadata filters, hybrid retrieval, larger candidate pool.

- `context_precision` low:
  Do retrieval precision work first.
  Add or tune reranking, reduce `top_k`, tighten filters, separate broad recall from final evidence selection.

- `faithfulness` low, and recall is not terrible:
  Do generation grounding work.
  Tighten prompts so every claim must be backed by retrieved evidence, require citations, add verification.

- `hallucination` high or `noise_sensitivity` high:
  Treat as grounding plus noisy-context failure.
  Improve context packing, constrain answers to cited evidence, add verifier, inspect retrieval noise.

- `context_utilization` low:
  Suspect context packing.
  Reorder chunks, reduce context clutter, use section-aware packing, tune context budget.

- `citation_accuracy` low:
  Fix citation binding separately from answer quality.
  Return passage IDs, require sentence-level citations, bind claims to spans explicitly.

- `answer_correctness` low, but other metrics do not clearly point somewhere:
  Run ablations.
  Hold one layer fixed at a time and compare retriever, reranker, packing, and prompt variants.

- evaluator agreement low:
  Do not trust the diagnosis too much yet.
  Audit the metric prompt, compare against human labels, recalibrate judges.

- high negative feedback, answerless responses, or domain-shift metadata:
  Suspect distribution shift.
  Cluster failures, compare production queries to benchmark queries, expand the eval set.

**Default thresholds**

From [thresholds.py](</d:/Codes/rag_diagnosis_lib_v2/src/ragdx/core/thresholds.py:31>):

- `context_precision`: `0.80`
- `context_recall`: `0.80`
- `faithfulness`: `0.90`
- `context_utilization`: `0.75`
- `noise_sensitivity`: `0.20` lower is better
- `hallucination`: `0.10` lower is better
- `answer_correctness`: `0.85`
- `citation_accuracy`: `0.85`

**Good order of operations**

- If recall is bad: fix corpus/retrieval before prompt tuning.
- If precision is bad: fix ranking/filtering before prompt tuning.
- If retrieval is decent but faithfulness is bad: fix generation grounding.
- If citations are bad but faithfulness is okay: fix citation formatting/binding.
- If nothing is clear: run controlled ablations, not random tuning.

**Practical examples**

- `context_recall=0.55`, `context_precision=0.84`:
  First move: retrieval recall and chunking.

- `context_precision=0.62`, `hallucination=0.18`:
  First move: reranking/filtering, then grounding.

- `faithfulness=0.74`, `context_recall=0.81`:
  First move: generator prompt and verification, not retriever.

- `citation_accuracy=0.60`, `faithfulness=0.92`:
  First move: citation binding/output schema.

