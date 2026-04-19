# RiskLab Diagnostics Agent

- **Final Report:** [Open the report](docs/report/final_report.pdf)
- **Project Overview:** [Open the overview](docs/overview.md)

RiskLab Diagnostics Agent is a compact reasoning-and-diagnostics system for model-risk and stress-testing questions. It combines controlled document evidence, computed quantitative evidence, deterministic verification, abstention, local traces, and canonical evaluation in one CLI-first workflow.

The system is intentionally narrow. It is not a broad regulatory platform or a general finance interface. Its purpose is to show how model-risk questions can be answered with explicit evidence boundaries: retrieved document support remains separate from computed diagnostics, and both remain separate from routing, confidence, verification, and abstention decisions.

## Key Ideas

- **Evidence separation:** final answers distinguish document evidence, computed evidence, workflow decisions, and verification or abstention state.
- **Deterministic diagnostics tool:** `risk_diagnostics_tool` computes diagnostics and stress evidence from prepared structured artifacts instead of asking the language model to infer numeric results.
- **Abstention:** unsupported, weakly supported, or ambiguous multi-mode requests can return an explicit refusal instead of a forced answer.
- **Evaluation versus interactive use:** CLI queries are useful for inspection; the canonical evaluation artifact is the repeatable system-level evidence.

## System Overview

The implemented workflow has five main layers:

1. **Bounded inputs:** a frozen three-document corpus under `docs/domain/`, chunked into `data/corpus/chunks.jsonl`, plus structured diagnostics and stress artifacts under `data/structured/`.
2. **Hybrid retrieval:** BM25 and dense retrieval return citation-ready document chunks with stable identifiers and scores.
3. **Computed evidence:** the deterministic diagnostics tool returns structured output for `diagnostics` and `stress` modes.
4. **Workflow control:** the graph classifies the query, retrieves support, invokes the diagnostics tool when needed, synthesizes an answer, verifies support, and can abstain.
5. **Inspection and evaluation:** traces are written to `outputs/traces/runs.jsonl`; canonical evaluation writes `outputs/eval_results/eval_results.json`.

For the detailed design, results, and limitations, see the [final technical report](docs/report/final_report.pdf).

## How To Run

Install the project environment:

```bash
uv sync --locked
```

For live query and canonical evaluation runs, set `GEMINI_API_KEY` in `.env` or in the shell environment. `.env.example` provides the expected local configuration keys.

Run a query through the real workflow:

```bash
uv run risklab query --clean "What does the validation report indicate about the champion versus challenger on OOT discrimination, calibration, and score stability?"
```

Useful query flags:

- `--verbose` prints route, retrieval scores, limitations, and extra metadata.
- `--json` emits machine-readable output.
- `--allow-fallback` allows the configured fallback path if provider calls fail.

Run canonical evaluation:

```bash
uv run risklab eval --clean --mode canonical
```

Emit the evaluation artifact to stdout:

```bash
uv run risklab eval --clean --mode canonical --json
```

The module entrypoint remains available:

```bash
PYTHONPATH=src .venv/bin/python -m eval --mode canonical
```

## Evaluation Artifact

The canonical evaluation output is [outputs/eval_results/eval_results.json](outputs/eval_results/eval_results.json).

It records run metadata, aggregate metric summaries, slice-level results, question-level results, and failure examples. The canonical artifact covers 18 curated questions across policy, numeric, mixed, and unsupported cases. It measures route and tool correctness, retrieval source coverage, groundedness structure, numeric consistency, and abstention correctness.

The artifact is intentionally more important than a single hand-picked query. It shows both strengths and the measured weakness: mixed diagnostics-governance prompts can miss expected SR 11-7 support.

## Repository Structure

```text
data/
  corpus/             Chunked corpus artifacts.
  eval/               Curated evaluation questions.
  structured/         Diagnostics, stress, metric, and scenario artifacts.
docs/
  domain/             Frozen source PDFs.
  report/             Final report PDF.
  overview.md         Concise system overview.
  evaluation.md       Evaluation artifact guide.
outputs/
  eval_results/       Canonical evaluation artifact.
  traces/             Local workflow traces.
src/
  app/                CLI and configuration boundary.
  eval/               Evaluation runner and metrics.
  graph/              Query workflow and provider boundary.
  ingestion/          Corpus ingestion.
  retrieval/          Hybrid retrieval.
  schemas/            Shared typed records.
  tools/              Deterministic diagnostics tool.
  traces/             Trace persistence.
tests/                Smoke and regression checks.
```

## Scope Boundaries

RiskLab Diagnostics Agent is a bounded decision-support prototype. It does not perform production model approval, broad regulatory compliance review, external deployment, or general-purpose financial analysis. Numeric outputs are diagnostic and tied to the prepared structured artifacts. Document support is limited to the frozen corpus.

## Additional Documentation

- [System overview](docs/overview.md)
- [Evaluation guide](docs/evaluation.md)
- [Final technical report](docs/report/final_report.pdf)
