# Evaluation Guide

The canonical evaluation artifact is [outputs/eval_results/eval_results.json](../outputs/eval_results/eval_results.json).

It is the primary repeatable evidence for system behavior. Interactive CLI queries are useful for inspection, but the evaluation artifact records the measured behavior across a curated set of 18 questions.

## What It Measures

The evaluation set covers policy, numeric, mixed, and unsupported questions. It measures:

- route and tool correctness;
- retrieval source coverage;
- groundedness structure;
- numeric consistency;
- abstention correctness;
- bounded answer adequacy as a review-oriented field.

The metrics are intentionally narrow. They check whether the implemented workflow follows its evidence contract; they do not claim broad benchmark coverage.

## Canonical Artifact Summary

The canonical artifact reports:

- 18 completed questions;
- no runtime errors;
- route and tool correctness passing on all evaluated questions;
- groundedness structure passing on all evaluated questions;
- numeric consistency passing on all evaluated numeric cases;
- abstention correctness passing on all unsupported cases;
- two retrieval source coverage failures, both in mixed cases.

The mixed-case retrieval failures are important. They show that the system is strongest on document-only, numeric, and clearly unsupported questions, while mixed diagnostics-governance questions can miss expected SR 11-7 support.

## Run The Evaluation

```bash
uv run risklab eval --clean --mode canonical
```

To print the artifact:

```bash
uv run risklab eval --clean --mode canonical --json
```

The compatibility module entrypoint is:

```bash
PYTHONPATH=src .venv/bin/python -m eval --mode canonical
```

Canonical evaluation uses the real runtime path and requires the configured provider and retriever environment. Environment blockers are reported explicitly rather than converted into partial quality scores.
