from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer

from eval import run_eval
from graph import ProviderConfigurationError, ProviderError, run_query
from retrieval import RetrievalLoadError
from schemas import DiagnosticsOutput, EvalRunMode, EvalRunStatus, EvidenceSourceType, GraphState, StressOutput
from tools import StructuredArtifactLoadError, computed_source_id_for_mode

from .config import load_settings


app = typer.Typer(
    add_completion=False,
    help="Thin CLI for the RiskLab diagnostics workflow.",
    no_args_is_help=True,
)

_DETERMINISTIC_METRIC_LABELS = {
    "route_and_tool_correctness": "Route/tool correctness",
    "retrieval_source_coverage": "Retrieval coverage",
    "groundedness_structure": "Groundedness structure",
    "numeric_consistency": "Numeric consistency",
    "abstention_correctness": "Abstention correctness",
}

_EVAL_EXIT_CODES = {
    EvalRunStatus.COMPLETED.value: 0,
    EvalRunStatus.ENVIRONMENT_BLOCKED.value: 1,
    EvalRunStatus.RUNNER_FAILED.value: 1,
}
_CLEAN_ENV_OVERRIDES = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
}
_BENIGN_BACKEND_PATTERNS = (
    "Warning: You are sending unauthenticated requests to the HF Hub.",
    "Loading weights:",
    "BertModel LOAD REPORT from:",
)
_BENIGN_BACKEND_EXACT_LINES = {
    "Notes:",
    "Key                     | Status     |  |",
    "embeddings.position_ids | UNEXPECTED |  |",
}

RuntimeCallT = TypeVar("RuntimeCallT")


@app.command("query")
def query_command(
    question: str = typer.Argument(..., help="Question to run through the real workflow."),
    json_output: bool = typer.Option(False, "--json", help="Emit structured JSON output."),
    verbose: bool = typer.Option(False, "--verbose", help="Show extra metadata, scores, and limitations."),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Suppress known benign backend noise for cleaner demo output.",
    ),
    allow_fallback: bool = typer.Option(
        False,
        "--allow-fallback",
        help="Allow the workflow's existing provider fallback path when the provider call fails.",
    ),
) -> None:
    settings = load_settings()
    try:
        state = _invoke_runtime_call(
            lambda: run_query(
                question,
                settings=settings,
                allow_fallback=allow_fallback,
            ),
            clean=clean,
        )
    except ProviderConfigurationError as exc:
        _emit_query_error(
            status="environment_blocked",
            message=str(exc),
            json_output=json_output,
            hint=f"Set {settings.provider.api_key_env_var} in .env or your environment and retry.",
        )
        raise typer.Exit(code=1)
    except RetrievalLoadError as exc:
        _emit_query_error(
            status="environment_blocked",
            message=str(exc),
            json_output=json_output,
            hint="Ensure the ready corpus artifacts exist under data/corpus/ and retry.",
        )
        raise typer.Exit(code=1)
    except StructuredArtifactLoadError as exc:
        _emit_query_error(
            status="environment_blocked",
            message=str(exc),
            json_output=json_output,
            hint="Ensure the ready structured artifacts exist under data/structured/ and retry.",
        )
        raise typer.Exit(code=1)
    except ProviderError as exc:
        _emit_query_error(
            status="provider_error",
            message=str(exc),
            json_output=json_output,
            hint="Retry after resolving provider availability or use --allow-fallback if appropriate.",
        )
        raise typer.Exit(code=1)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        _emit_query_error(
            status="runtime_error",
            message=f"{type(exc).__name__}: {exc}",
            json_output=json_output,
        )
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(
            json.dumps(
                _build_query_payload(
                    state,
                    trace_path=settings.paths.trace_runs_path,
                    verbose=verbose,
                ),
                indent=2,
                ensure_ascii=True,
            )
        )
        raise typer.Exit(code=0)

    typer.echo(
        _render_query_output(
            state,
            trace_path=settings.paths.trace_runs_path,
            verbose=verbose,
        )
    )
    raise typer.Exit(code=0)


@app.command("eval")
def eval_command(
    mode: EvalRunMode = typer.Option(
        EvalRunMode.CANONICAL,
        "--mode",
        help="Evaluation mode. Canonical uses the real runtime path; hermetic_verify is a thin wrapper over the runner.",
    ),
    questions_path: Path | None = typer.Option(None, "--questions-path", help="Optional path to an eval JSONL file."),
    output_path: Path | None = typer.Option(None, "--output-path", help="Optional output artifact path."),
    allow_fallback: bool = typer.Option(
        False,
        "--allow-fallback",
        help="Allow the workflow's existing provider fallback path when the provider call fails.",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Suppress known benign backend noise for cleaner demo output.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the full eval artifact as JSON."),
) -> None:
    try:
        artifact = _invoke_runtime_call(
            lambda: run_eval(
                mode=mode,
                questions_path=questions_path,
                output_path=output_path,
                allow_fallback=allow_fallback,
            ),
            clean=clean,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "status": "runtime_error",
                        "error_type": "runtime_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        else:
            typer.echo("EVAL RUNTIME_ERROR")
            typer.echo(f"Message: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1)

    if json_output:
        typer.echo(json.dumps(artifact, indent=2, ensure_ascii=True))
    else:
        typer.echo(_render_eval_output(artifact))

    status = str(artifact.get("run_metadata", {}).get("status"))
    raise typer.Exit(code=_EVAL_EXIT_CODES.get(status, 1))


def main() -> None:
    app()


def _build_query_payload(
    state: GraphState,
    *,
    trace_path: Path,
    verbose: bool,
) -> dict[str, Any]:
    final_answer = state.final_answer
    payload: dict[str, Any] = {
        "status": "abstained" if state.abstained else "answered",
        "query": state.query,
        "run_id": state.run_id,
        "query_type": state.query_type.value if state.query_type is not None else None,
        "classification_source": state.classification_source,
        "tool_used": state.tool_output is not None,
        "tool_mode": state.tool_mode.value if state.tool_mode is not None else None,
        "confidence": state.confidence.value if state.confidence is not None else None,
        "abstained": state.abstained,
        "unsupported_reason": state.unsupported_reason,
        "executive_answer": final_answer.executive_answer if final_answer is not None else None,
        "document_evidence": _build_document_evidence(state, verbose=verbose),
        "computed_evidence": _build_computed_evidence(state, verbose=verbose),
        "latency_ms": state.latency_ms,
        "trace_path": str(trace_path),
    }
    if verbose:
        payload["review_flag"] = state.review_flag
        payload["limitations"] = list(final_answer.limitations) if final_answer is not None else []
    return payload


def _build_document_evidence(state: GraphState, *, verbose: bool) -> list[dict[str, Any]]:
    if state.final_answer is None:
        return []

    chunks_by_id = {chunk.chunk_id: chunk for chunk in state.retrieved_chunks}
    evidence_rows: list[dict[str, Any]] = []
    for item in state.final_answer.evidence:
        if item.source_type is not EvidenceSourceType.DOCUMENT:
            continue
        chunk = chunks_by_id.get(item.source_id)
        row: dict[str, Any] = {
            "source_id": item.source_id,
            "doc_id": item.doc_id or (chunk.doc_id if chunk is not None else None),
            "page": item.page if item.page is not None else (chunk.page if chunk is not None else None),
            "section_path": chunk.section_path if chunk is not None else None,
            "support": item.support,
        }
        if verbose and chunk is not None:
            row["scores"] = {
                "sparse": chunk.sparse_score,
                "dense": chunk.dense_score,
                "fused": chunk.fused_score,
            }
            row["topic_tags"] = list(chunk.topic_tags)
        evidence_rows.append(row)
    return evidence_rows


def _build_computed_evidence(state: GraphState, *, verbose: bool) -> dict[str, Any] | None:
    if state.tool_output is None or state.tool_mode is None:
        return None

    if isinstance(state.tool_output, DiagnosticsOutput):
        payload: dict[str, Any] = {
            "source_id": computed_source_id_for_mode(state.tool_mode),
            "mode": state.tool_output.mode,
            "summary": state.tool_output.summary,
            "top_drift_features": [
                {
                    "feature": feature["feature"],
                    "flag": feature["flag"],
                    "psi": feature["psi"],
                }
                for feature in state.tool_output.metrics.top_drift_features[:3]
            ],
            "calibration_slopes": {
                "champion_oot": state.tool_output.metrics.calibration.get("champion_oot", {}).get("calibration_slope"),
                "challenger_oot": state.tool_output.metrics.calibration.get("challenger_oot", {}).get("calibration_slope"),
            },
            "score_shift_psi": {
                "champion_logit": state.tool_output.metrics.score_shift.get("champion_logit", {}).get("score_psi"),
                "challenger_lgbm": state.tool_output.metrics.score_shift.get("challenger_lgbm", {}).get("score_psi"),
            },
        }
    else:
        payload = {
            "source_id": computed_source_id_for_mode(state.tool_mode),
            "mode": state.tool_output.mode,
            "summary": state.tool_output.summary,
            "mean_pd_delta": state.tool_output.metrics.delta_mean_pd.model_dump(mode="json"),
            "tail_pd_delta": state.tool_output.metrics.delta_tail_pd.model_dump(mode="json"),
            "el_proxy_delta": state.tool_output.metrics.delta_el_proxy.model_dump(mode="json"),
            "monotonicity_passed": state.tool_output.metrics.monotonicity_passed,
        }

    if verbose:
        payload["quality_flags"] = list(state.tool_output.quality_flags)
        payload["limitations"] = list(state.tool_output.limitations)
    return payload


def _render_query_output(
    state: GraphState,
    *,
    trace_path: Path,
    verbose: bool,
) -> str:
    final_answer = state.final_answer
    document_evidence = _build_document_evidence(state, verbose=verbose)
    computed_evidence = _build_computed_evidence(state, verbose=verbose)
    trace_display_path = _display_path(trace_path)
    lines = [
        "ABSTAINED" if state.abstained else "ANSWERED",
        final_answer.executive_answer if final_answer is not None else "No final answer was produced.",
        f"Route: {state.query_type.value if state.query_type is not None else 'unknown'}",
        f"Grounding: {_grounding_mode(state)}",
        f"Tool: {state.tool_mode.value if state.tool_mode is not None else 'none'}",
        f"Confidence: {state.confidence.value if state.confidence is not None else 'unknown'}",
        f"Sources used: {_sources_used_summary(document_evidence, computed_evidence)}",
    ]
    if state.review_flag:
        lines.append("Review flag: true")
    if state.unsupported_reason is not None:
        lines.append(f"Unsupported reason: {state.unsupported_reason}")

    if document_evidence:
        lines.extend(["", "Document Evidence"])
        evidence_index = 1
        for doc_id, items in _group_document_evidence(document_evidence):
            lines.append(doc_id)
            for item in items:
                header = f"{evidence_index}. p.{item['page']}"
                if item.get("section_path"):
                    header += f" | {item['section_path']}"
                lines.append(header)
                lines.append(f"   {item['support']}")
                if verbose and "scores" in item:
                    scores = item["scores"]
                    lines.append(
                        "   scores: "
                        f"sparse={_format_value(scores['sparse'])}, "
                        f"dense={_format_value(scores['dense'])}, "
                        f"fused={_format_value(scores['fused'])}"
                    )
                    if item.get("topic_tags"):
                        lines.append(f"   topic_tags: {', '.join(item['topic_tags'])}")
                evidence_index += 1

    if computed_evidence is not None:
        lines.extend(["", "Computed Evidence"])
        lines.append(f"Source: {computed_evidence['source_id']}")
        lines.append(f"Mode: {computed_evidence['mode']}")
        lines.append(f"Summary: {computed_evidence['summary']}")
        if computed_evidence["mode"] == "diagnostics":
            drift_summary = "; ".join(
                f"{feature['feature']} ({feature['flag']}, PSI {_format_value(feature['psi'])})"
                for feature in computed_evidence["top_drift_features"]
            )
            lines.append(f"Top drift features: {drift_summary}")
            lines.append(
                "OOT calibration slopes: "
                f"champion={_format_value(computed_evidence['calibration_slopes']['champion_oot'])}, "
                f"challenger={_format_value(computed_evidence['calibration_slopes']['challenger_oot'])}"
            )
            lines.append(
                "Score-shift PSI: "
                f"champion={_format_value(computed_evidence['score_shift_psi']['champion_logit'])}, "
                f"challenger={_format_value(computed_evidence['score_shift_psi']['challenger_lgbm'])}"
            )
        else:
            lines.append(
                "Mean PD delta: "
                f"mild={_format_value(computed_evidence['mean_pd_delta'].get('mild'))}, "
                f"severe={_format_value(computed_evidence['mean_pd_delta'].get('severe'))}"
            )
            lines.append(
                "Tail PD delta: "
                f"mild={_format_value(computed_evidence['tail_pd_delta'].get('mild'))}, "
                f"severe={_format_value(computed_evidence['tail_pd_delta'].get('severe'))}"
            )
            lines.append(
                "EL proxy delta: "
                f"mild={_format_value(computed_evidence['el_proxy_delta'].get('mild'))}, "
                f"severe={_format_value(computed_evidence['el_proxy_delta'].get('severe'))}"
            )
            lines.append(
                "Monotonicity: "
                + ("passed" if computed_evidence["monotonicity_passed"] else "failed")
            )
        if verbose:
            quality_flags = computed_evidence.get("quality_flags", [])
            if quality_flags:
                lines.append(f"Quality flags: {', '.join(quality_flags)}")
            limitations = computed_evidence.get("limitations", [])
            if limitations:
                lines.append("Limitations:")
                lines.extend(f"- {limitation}" for limitation in limitations)

    if verbose and final_answer is not None:
        lines.extend(["", f"Classification source: {state.classification_source or 'unavailable'}"])
        if final_answer.limitations:
            lines.append("Answer limitations:")
            lines.extend(f"- {limitation}" for limitation in final_answer.limitations)
        if trace_display_path != str(trace_path):
            lines.append(f"Trace path: {trace_path}")

    lines.extend(["", f"Run: {state.run_id or 'unavailable'}", f"Trace: {trace_display_path}"])
    return "\n".join(lines)


def _render_eval_output(artifact: dict[str, Any]) -> str:
    run_metadata = artifact.get("run_metadata", {})
    summary = artifact.get("summary", {})
    status = str(run_metadata.get("status", "runner_failed"))
    lines = [
        f"EVAL {status.upper()}",
        (
            f"mode={run_metadata.get('mode')} | "
            f"total_questions={summary.get('total_questions', run_metadata.get('total_questions'))} | "
            f"completed_questions={summary.get('completed_questions', 0)} | "
            f"runtime_error_count={summary.get('runtime_error_count', 0)} | "
            f"environment_blocked_count={summary.get('environment_blocked_count', 0)}"
        ),
    ]

    blocker_reason = run_metadata.get("blocker_reason")
    if blocker_reason:
        lines.append(f"Reason: {blocker_reason}")

    metric_summaries = summary.get("metric_summaries", {})
    deterministic_rows = []
    for metric_name, label in _DETERMINISTIC_METRIC_LABELS.items():
        metric_summary = metric_summaries.get(metric_name)
        if not metric_summary:
            continue
        deterministic_rows.append(
            f"- {label}: {metric_summary['pass_count']}/{metric_summary['evaluated_count']} pass"
            + (
                f" ({metric_summary['pass_rate']:.4f})"
                if metric_summary.get("pass_rate") is not None
                else ""
            )
        )
    if deterministic_rows:
        lines.extend(["", "Deterministic Metrics"])
        lines.extend(deterministic_rows)

    failure_tag_counts = summary.get("failure_tag_counts", {})
    if failure_tag_counts:
        ordered_failures = sorted(
            failure_tag_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
        lines.extend(["", "Top Failure Tags"])
        lines.extend(f"- {tag}: {count}" for tag, count in ordered_failures[:5])

    failure_examples = artifact.get("failure_examples", [])
    if failure_examples:
        lines.extend(["", "Failure Examples"])
        for example in failure_examples[:2]:
            tool_mode = example.get("tool_mode") or "none"
            lines.append(
                "- "
                f"{example.get('question_id')} | "
                f"tags={','.join(example.get('failure_tags', []))} | "
                f"query_type={example.get('query_type')} | "
                f"tool={tool_mode}"
            )
            lines.append(str(example.get("question", "")))

    output_path = run_metadata.get("output_path")
    if output_path:
        lines.extend(["", f"Artifact: {output_path}"])
    return "\n".join(lines)


def _emit_query_error(
    *,
    status: str,
    message: str,
    json_output: bool,
    hint: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "error_type": status,
        "message": message,
    }
    if hint is not None:
        payload["hint"] = hint

    if json_output:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    typer.echo(status.upper())
    typer.echo(f"Message: {message}")
    if hint is not None:
        typer.echo(f"Hint: {hint}")


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _grounding_mode(state: GraphState) -> str:
    evidence = state.final_answer.evidence if state.final_answer is not None else []
    document_present = any(item.source_type is EvidenceSourceType.DOCUMENT for item in evidence)
    computed_present = any(item.source_type is EvidenceSourceType.COMPUTED for item in evidence)
    if document_present and computed_present:
        return "document_plus_computed"
    if document_present:
        return "document_only"
    if computed_present:
        return "computed_only"
    return "none"


def _group_document_evidence(
    evidence_rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        doc_id = str(row.get("doc_id") or "unknown_document")
        grouped.setdefault(doc_id, []).append(row)
    return list(grouped.items())


def _sources_used_summary(
    document_evidence: list[dict[str, Any]],
    computed_evidence: dict[str, Any] | None,
) -> str:
    if not document_evidence and computed_evidence is None:
        return "none"

    counts: dict[str, int] = {}
    ordered_doc_ids: list[str] = []
    for item in document_evidence:
        doc_id = str(item.get("doc_id") or "unknown_document")
        if doc_id not in counts:
            ordered_doc_ids.append(doc_id)
            counts[doc_id] = 0
        counts[doc_id] += 1

    parts = [f"{doc_id} ({counts[doc_id]} chunk{'s' if counts[doc_id] != 1 else ''})" for doc_id in ordered_doc_ids]
    if computed_evidence is not None:
        parts.append(f"{computed_evidence['source_id']} (computed)")
    return "; ".join(parts)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _invoke_runtime_call(operation: Callable[[], RuntimeCallT], *, clean: bool) -> RuntimeCallT:
    if not clean:
        return operation()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    previous_env = {key: os.environ.get(key) for key in _CLEAN_ENV_OVERRIDES}
    captured_exception: Exception | None = None
    result: RuntimeCallT | None = None

    try:
        os.environ.update(_CLEAN_ENV_OVERRIDES)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                result = operation()
            except Exception as exc:  # pragma: no cover - exercised through CLI boundaries
                captured_exception = exc
    finally:
        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value

    residual_output = _filter_benign_backend_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())
    if residual_output:
        typer.echo(residual_output, err=True)

    if captured_exception is not None:
        raise captured_exception
    return result


def _filter_benign_backend_output(*streams: str) -> str:
    kept_lines: list[str] = []
    for stream in streams:
        normalized_stream = stream.replace("\r", "\n")
        for raw_line in normalized_stream.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                continue
            if _is_benign_backend_line(line):
                continue
            kept_lines.append(line)
    return "\n".join(kept_lines)


def _is_benign_backend_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in _BENIGN_BACKEND_EXACT_LINES:
        return True
    if stripped.startswith("------------------------+"):
        return True
    if stripped.startswith("- UNEXPECTED"):
        return True
    return any(pattern in stripped for pattern in _BENIGN_BACKEND_PATTERNS)
