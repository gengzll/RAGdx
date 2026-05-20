# Examples

## 1. Diagnose a normalized evaluation file

```bash
ragdx diagnose examples/demo_evaluation.json
```

## 2. Diagnose with both rule and LLM reasoning

```bash
export OPENAI_API_KEY=your_key
export RAGDX_OPENAI_MODEL=gpt-5.4-thinking
ragdx diagnose examples/demo_evaluation.json --use-both
```

## 3. Generate a human-readable plan

```bash
ragdx plan examples/demo_evaluation.json --human-readable
```

## 4. Generate a plan with LLM planning refinement

```bash
ragdx plan examples/demo_evaluation.json --use-llm-planner --human-readable
```

## 5. Save a run and export a report

```bash
ragdx save examples/demo_evaluation.json --name baseline-demo
ragdx runs
ragdx export-report <RUN_ID> run_report.md
```

## 6. Compare two evaluations

```bash
ragdx compare examples/demo_evaluation.json examples/demo_evaluation_baseline.json
```

## 7. Simulate optimization

```bash
ragdx optimize examples/demo_evaluation.json --strategy bayesian --budget 8 --mode simulate
```

## 8. Prepare configs without executing

```bash
ragdx optimize examples/demo_evaluation.json --strategy pareto_evolutionary --budget 10 --mode prepare_only
```

## 9. Execute LangChain trials

```bash
pip install -e ".[langchain]"
export RAGDX_LANGCHAIN_RUNNER_CMD='python examples/run_langchain_trial.py --config {config} --output {output}'
ragdx optimize examples/demo_evaluation_langchain.json --strategy bayesian --budget 6 --mode execute
```

## 10. Execute LlamaIndex trials

```bash
pip install -e ".[llamaindex]"
export RAGDX_LLAMAINDEX_RUNNER_CMD='python examples/run_llamaindex_trial.py --config {config} --output {output}'
ragdx optimize examples/demo_evaluation_llamaindex.json --strategy bayesian --budget 6 --mode execute
```

## 11. Show runner templates

```bash
ragdx show-runner-templates
```

## 12. Normalize external tool outputs

```bash
ragdx normalize-tools --ragas-json ragas_output.json --ragchecker-json ragchecker_output.json --output-json normalized.json
```

## 13. Attach feedback

```bash
ragdx attach-feedback <RUN_ID> examples/feedback_events.json
ragdx feedback-summary
```

## 14. Explain a plan JSON file

```bash
ragdx explain-plan saved_plan.json
```

## 15. Python-level usage

```python
from ragdx.core.diagnosis import RAGDiagnosisEngine
from ragdx.optim.planner import OptimizationPlanner
from ragdx.schemas.models import EvaluationResult

result = EvaluationResult(
    retrieval={"context_recall": 0.72, "context_precision": 0.68},
    generation={"faithfulness": 0.81, "response_relevancy": 0.79},
    e2e={"answer_correctness": 0.74, "citation_accuracy": 0.77},
)

engine = RAGDiagnosisEngine()
report = engine.diagnose(result)
plan = OptimizationPlanner().build_plan(report, result=result, strategy="bayesian", budget=8)

print(report.summary)
print(plan.objective_metric)
```
