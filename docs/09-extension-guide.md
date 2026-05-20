# Extension Guide

## 1. Adding a new evaluator adapter

To support a new evaluator:

1. create an adapter under `src/ragdx/engines`
2. map external evaluator output into normalized metrics
3. optionally preserve the raw tool output under `EvaluationResult.raw_tool_outputs`
4. extend `UnifiedEvaluator` if needed

Key design rule:

- adapters should normalize into `EvaluationResult`, not force the rest of the system to know evaluator-specific formats

## 2. Adding a new diagnosis source

You can extend diagnosis in two ways:

### Rule-based extension

- update `engines/root_cause.py`
- add new causal nodes, edges, evidence logic, and hypotheses

### LLM-based extension

- update `engines/llm_diagnosis.py`
- refine prompts and output schema handling

## 3. Adding a new optimization tool

To add a new optimization tool:

1. create an adapter under `src/ragdx/optim`
2. define how configs are emitted
3. define runner integration and output expectations
4. update the planner to route specific experiment types to the new tool

## 4. Adding a new runtime framework

To add a new runtime such as Haystack or custom internal pipelines:

1. create a runtime adapter under `src/ragdx/optim`
2. define the metadata contract
3. create example pipeline and trial scripts
4. add an environment variable template for the runner command
5. update the planner and executor routing if required

## 5. Extending the search space

Search spaces are defined per experiment. You can extend them by modifying:

- planner search-space generation
- runtime adapter config mapping
- executor parameter suggestion logic

## 6. Extending constraints

Constraints should be explicit and directional. Typical additions may include:

- policy violation rate
- refusal error rate
- cost per thousand queries
- answer length ceiling
- tool call budget

Update:

- planning logic
- feasibility checks
- dashboard visualization

## 7. Extending reports and dashboard views

For reporting:

- update `utils/reporting.py`
- update `storage/run_store.py` markdown export

For dashboard:

- update `ui/dashboard.py`
- ensure new fields are exposed cleanly from the saved models

## 8. Testing guidance

Every extension should add or update tests under `tests/`.

Recommended coverage:

- schema validation
- plan generation
- execution behavior
- constraint handling
- reporting behavior
- regression tests for human-readable explanations
