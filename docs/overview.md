# RiskLab Diagnostics Agent Overview

RiskLab Diagnostics Agent is a compact, implemented reasoning-and-diagnostics system for model-risk and stress-testing questions. The completed project combines a frozen corpus, prepared quantitative artifacts, hybrid retrieval, a deterministic diagnostics tool, a workflow layer, local traces, and a canonical evaluation artifact.

## Completed Scope

The system handles bounded questions that combine model-risk governance, validation evidence, drift, calibration, score stability, and stress-testing interpretation. It demonstrates evidence separation by answering from document support, computed diagnostics, or both, and by refusing unsupported requests when available evidence is insufficient.

## Evidence Model

The system separates four evidence and state layers:

- **Document evidence:** retrieved chunks from the frozen corpus, with document IDs, chunk IDs, page metadata, and retrieval scores.
- **Computed evidence:** structured output from the deterministic diagnostics tool for diagnostics and stress modes.
- **Workflow decisions:** route, tool mode, confidence, review flag, retrieved document IDs, and trace metadata.
- **Abstention state:** explicit unsupported or weak-support outcomes.

This separation is the central design point. A route decision is not evidence. A computed metric is not a policy citation. A citation is not a numeric calculation. Keeping these layers separate makes the result easier to inspect.

## Runtime Flow

At a high level, the workflow is:

1. receive the user question;
2. classify the question as policy, numeric, mixed, or unsupported;
3. retrieve document evidence from the chunked corpus;
4. invoke `risk_diagnostics_tool` when numeric diagnostics or stress evidence is required;
5. synthesize a bounded answer from available evidence;
6. verify support and abstain when the answer cannot be supported;
7. persist a local trace.

The primary runtime entrypoint is the `risklab` CLI. The workflow also writes local traces to `outputs/traces/runs.jsonl`.

## Corpus And Artifacts

The corpus is fixed to three source PDFs:

- `docs/domain/Model_Validation_Report.pdf`
- `docs/domain/SR11_7_Model_Risk_Management.pdf`
- `docs/domain/Basel_Stress_Testing_Principles_2018.pdf`

The prepared corpus artifact is `data/corpus/chunks.jsonl`. Structured diagnostics and stress artifacts live under `data/structured/`.

## Boundaries

The system is not a production model-risk platform, regulatory compliance engine, or broad financial analysis environment. It is a bounded decision-support prototype with explicit limitations. The most important measured limitation is mixed multi-source retrieval coverage, especially when a question requires both internal validation evidence and SR 11-7 support.

For the complete design and results discussion, see the [final technical report](report/final_report.pdf).
