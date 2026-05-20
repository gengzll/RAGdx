# ragdx Documentation Set

This folder contains a detailed markdown documentation set for `ragdx`, a RAG evaluation, diagnosis, optimization, and monitoring workbench.

## Documentation map

- [01-overview.md](01-overview.md): scope, design goals, and end-to-end lifecycle
- [02-architecture.md](02-architecture.md): component-level architecture and package structure
- [03-data-models.md](03-data-models.md): schemas, files, traces, feedback, and session artifacts
- [04-workflows.md](04-workflows.md): step-by-step operational workflows
- [05-cli-and-dashboard.md](05-cli-and-dashboard.md): command-line interface and dashboard behavior
- [06-configuration.md](06-configuration.md): environment variables, package extras, runners, and configuration patterns
- [07-optimization-and-diagnosis.md](07-optimization-and-diagnosis.md): metrics, causal graph, LLM reasoning, optimization strategy, and constraints
- [08-runtime-integrations.md](08-runtime-integrations.md): DSPy, AutoRAG, LangChain, and LlamaIndex integration patterns
- [09-extension-guide.md](09-extension-guide.md): how to extend the library with new metrics, tools, runtimes, and optimization strategies
- [10-examples.md](10-examples.md): practical examples and typical end-to-end usage patterns
- [11-limitations-and-roadmap.md](11-limitations-and-roadmap.md): current limitations, design trade-offs, and roadmap suggestions

## Suggested reading order

For a first pass:

1. `01-overview.md`
2. `02-architecture.md`
3. `04-workflows.md`
4. `05-cli-and-dashboard.md`
5. `06-configuration.md`
6. `10-examples.md`

For implementation and extension work:

1. `03-data-models.md`
2. `07-optimization-and-diagnosis.md`
3. `08-runtime-integrations.md`
4. `09-extension-guide.md`
