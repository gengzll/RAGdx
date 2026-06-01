# ragdx run report: demo3-optimized

- Run ID: `5fe0a5837a23`
- Created at: 2026-06-01T02:17:28.561338+00:00
- Baseline run: `46c0fcbe5954`
- Tags: none
- Latest optimization session: `none`

## Summary
Primary bottleneck: retrieval noise or weak ranking quality. 1 diagnosis hypotheses were generated. Lead causal node: retrieval_precision_defect with posterior 0.97.

- Diagnosis confidence: 0.84
- Feedback events attached: 0
- Query traces attached: 0

## Retrieval metrics
- context_precision: 0.5556

## Generation metrics
- faithfulness: 1.0000

## End-to-end metrics

## Evaluator agreement

## Hypotheses
- **retrieval noise or weak ranking quality** (retrieval, severity=high, confidence=0.84)

## Causal signals
- **retrieval_precision_defect** (retrieval) posterior=0.97, prior=0.95
- **corpus_chunking_defect** (retrieval) posterior=0.96, prior=0.95
- **context_packing_defect** (generation) posterior=0.95, prior=0.95
- **grounding_defect** (generation) posterior=0.95, prior=0.95
- **retrieval_recall_defect** (retrieval) posterior=0.95, prior=0.95

## Planned experiments
- **corpus-chunking-search** stage=`corpus` via `manual` targeting `retrieval`: Optimize parsing, structure preservation, and chunking before retrieval tuning.
- **retrieval-pipeline-search** stage=`retrieval` via `autorag` targeting `retrieval`: Optimize retrieval, reranking, and context packing for better evidence quality.
- **generator-prompt-optimization** stage=`generation` via `dspy` targeting `generation`: Optimize grounded answer synthesis, citation behavior, and verification.