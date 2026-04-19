from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import ProjectSettings, load_settings
from graph import run_query
from retrieval import get_default_retriever
from schemas import (
    AnswerEvidenceItem,
    EvalMetricResult,
    EvalMetricStatus,
    EvalQuestion,
    EvalRunMode,
    EvalRunStatus,
)


class EvalRunnerError(RuntimeError):
    """Base error for Phase 6 evaluation execution."""


class EvalDatasetError(EvalRunnerError):
    """Raised when the evaluation dataset is invalid."""


_CANONICAL_INTER_QUERY_DELAY_SECONDS = 7.0
_CANONICAL_RATE_LIMIT_RETRY_DELAY_SECONDS = 65.0


def load_eval_questions(
    *,
    questions_path: Path | None = None,
    settings: ProjectSettings | None = None,
) -> list[EvalQuestion]:
    resolved_settings = settings or load_settings()
    resolved_questions_path = questions_path or resolved_settings.paths.eval_questions_path
    if not resolved_questions_path.exists():
        raise EvalDatasetError(f"Missing eval questions artifact: {resolved_questions_path}")

    questions: list[EvalQuestion] = []
    raw_lines = resolved_questions_path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"Invalid JSON in eval questions on line {line_number}.") from exc
        try:
            questions.append(EvalQuestion.model_validate(payload))
        except Exception as exc:
            raise EvalDatasetError(f"Invalid eval question on line {line_number}: {exc}") from exc

    if not questions:
        raise EvalDatasetError("Eval questions artifact is empty.")

    duplicate_ids = [question_id for question_id, count in Counter(q.question_id for q in questions).items() if count > 1]
    if duplicate_ids:
        raise EvalDatasetError(f"Duplicate eval question IDs found: {', '.join(sorted(duplicate_ids))}")

    return questions


def run_eval(
    *,
    mode: EvalRunMode,
    settings: ProjectSettings | None = None,
    questions_path: Path | None = None,
    output_path: Path | None = None,
    provider: Any | None = None,
    retriever: Any | None = None,
    trace_store: Any | None = None,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    resolved_settings = settings or load_settings()
    resolved_questions_path = questions_path or resolved_settings.paths.eval_questions_path
    resolved_output_path = _resolve_output_path(
        mode=mode,
        settings=resolved_settings,
        output_path=output_path,
    )
    started_at_utc = _utc_now_iso()
    dependency_status: dict[str, Any] = {
        "provider_configured": resolved_settings.provider.is_configured,
        "provider_mode": "real" if mode is EvalRunMode.CANONICAL else "injected",
        "retriever_mode": "real" if mode is EvalRunMode.CANONICAL else "injected",
    }

    try:
        questions = load_eval_questions(
            questions_path=resolved_questions_path,
            settings=resolved_settings,
        )
    except EvalDatasetError as exc:
        artifact = _build_status_artifact(
            mode=mode,
            run_status=EvalRunStatus.RUNNER_FAILED,
            questions_path=resolved_questions_path,
            output_path=resolved_output_path,
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now_iso(),
            dependency_status=dependency_status,
            blocker_reason=str(exc),
            total_questions=0,
        )
        _maybe_write_artifact(resolved_output_path, artifact)
        return artifact

    preflight_reason: str | None = None
    if mode is EvalRunMode.CANONICAL:
        if provider is not None or retriever is not None:
            preflight_reason = "Canonical mode must use the real provider and retriever path."
            preflight_status = EvalRunStatus.RUNNER_FAILED
        elif not resolved_settings.provider.is_configured:
            preflight_reason = (
                f"Canonical evaluation requires {resolved_settings.provider.api_key_env_var} to be configured."
            )
            preflight_status = EvalRunStatus.ENVIRONMENT_BLOCKED
        else:
            try:
                get_default_retriever()
            except Exception as exc:
                dependency_status["retriever_ready"] = False
                dependency_status["retriever_error"] = f"{type(exc).__name__}: {exc}"
                preflight_reason = "Canonical evaluation could not initialize the real retriever."
                preflight_status = EvalRunStatus.ENVIRONMENT_BLOCKED
            else:
                dependency_status["retriever_ready"] = True
                preflight_status = EvalRunStatus.COMPLETED
    else:
        if provider is None or retriever is None:
            preflight_reason = "Hermetic verification requires injected provider and retriever dependencies."
            preflight_status = EvalRunStatus.RUNNER_FAILED
        else:
            dependency_status["retriever_ready"] = True
            preflight_status = EvalRunStatus.COMPLETED

    if preflight_reason is not None:
        artifact = _build_status_artifact(
            mode=mode,
            run_status=preflight_status,
            questions_path=resolved_questions_path,
            output_path=resolved_output_path,
            started_at_utc=started_at_utc,
            finished_at_utc=_utc_now_iso(),
            dependency_status=dependency_status,
            blocker_reason=preflight_reason,
            total_questions=len(questions),
        )
        _maybe_write_artifact(resolved_output_path, artifact)
        return artifact

    question_results: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        try:
            result_state = _run_query_with_rate_control(
                question=question.question,
                mode=mode,
                settings=resolved_settings,
                retriever=retriever,
                provider=provider,
                trace_store=trace_store,
                allow_fallback=allow_fallback,
                delay_before_seconds=_inter_query_delay_seconds(
                    mode=mode,
                    question_index=index,
                ),
            )
            question_results.append(
                _build_question_result(
                    question=question,
                    state=result_state,
                    mode=mode,
                )
            )
        except Exception as exc:
            question_results.append(
                _build_runtime_error_result(
                    question=question,
                    mode=mode,
                    error=exc,
                )
            )

    summary = _build_summary(question_results)
    slices = _build_slices(question_results)
    artifact = {
        "run_metadata": {
            "mode": mode.value,
            "status": EvalRunStatus.COMPLETED.value,
            "started_at_utc": started_at_utc,
            "finished_at_utc": _utc_now_iso(),
            "questions_path": str(resolved_questions_path),
            "output_path": str(resolved_output_path) if resolved_output_path is not None else None,
            "dependency_status": dependency_status,
            "total_questions": len(questions),
        },
        "summary": summary,
        "slices": slices,
        "questions": question_results,
        "failure_examples": _build_failure_examples(question_results),
    }
    _maybe_write_artifact(resolved_output_path, artifact)
    return artifact


def _inter_query_delay_seconds(*, mode: EvalRunMode, question_index: int) -> float:
    if mode is not EvalRunMode.CANONICAL or question_index == 0:
        return 0.0
    return _CANONICAL_INTER_QUERY_DELAY_SECONDS


def _run_query_with_rate_control(
    *,
    question: str,
    mode: EvalRunMode,
    settings: ProjectSettings,
    retriever: Any | None,
    provider: Any | None,
    trace_store: Any | None,
    allow_fallback: bool,
    delay_before_seconds: float,
) -> Any:
    if delay_before_seconds > 0.0:
        time.sleep(delay_before_seconds)

    try:
        return run_query(
            question,
            settings=settings,
            retriever=retriever,
            provider=provider,
            trace_store=trace_store,
            allow_fallback=allow_fallback,
        )
    except Exception as exc:
        if mode is not EvalRunMode.CANONICAL or not _is_rate_limit_error(exc):
            raise
        time.sleep(_CANONICAL_RATE_LIMIT_RETRY_DELAY_SECONDS)
        return run_query(
            question,
            settings=settings,
            retriever=retriever,
            provider=provider,
            trace_store=trace_store,
            allow_fallback=allow_fallback,
        )


def _is_rate_limit_error(error: Exception) -> bool:
    lowered_messages = " ".join(_exception_messages(error)).lower()
    return any(
        pattern in lowered_messages
        for pattern in (
            "429",
            "quota exceeded",
            "rate limit",
            "resource_exhausted",
            "retrydelay",
            "retry in",
        )
    )


def _exception_messages(error: BaseException) -> list[str]:
    messages: list[str] = []
    seen_ids: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return messages


def _resolve_output_path(
    *,
    mode: EvalRunMode,
    settings: ProjectSettings,
    output_path: Path | None,
) -> Path | None:
    if output_path is not None:
        return output_path
    if mode is EvalRunMode.CANONICAL:
        return settings.paths.eval_results_dir / "eval_results.json"
    return None


def _maybe_write_artifact(output_path: Path | None, artifact: dict[str, Any]) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _build_status_artifact(
    *,
    mode: EvalRunMode,
    run_status: EvalRunStatus,
    questions_path: Path,
    output_path: Path | None,
    started_at_utc: str,
    finished_at_utc: str,
    dependency_status: dict[str, Any],
    blocker_reason: str,
    total_questions: int,
) -> dict[str, Any]:
    return {
        "run_metadata": {
            "mode": mode.value,
            "status": run_status.value,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "questions_path": str(questions_path),
            "output_path": str(output_path) if output_path is not None else None,
            "dependency_status": dependency_status,
            "total_questions": total_questions,
            "blocker_reason": blocker_reason,
        },
        "summary": {
            "total_questions": total_questions,
            "completed_questions": 0,
            "runtime_error_count": 0,
            "environment_blocked_count": total_questions if run_status is EvalRunStatus.ENVIRONMENT_BLOCKED else 0,
            "failure_tag_counts": {},
            "metric_summaries": {},
            "confidence_alignment": {},
        },
        "slices": {},
        "questions": [],
        "failure_examples": [],
    }


def _build_question_result(
    *,
    question: EvalQuestion,
    state: Any,
    mode: EvalRunMode,
) -> dict[str, Any]:
    metrics = _score_question(question=question, state=state)
    failure_tags = _derive_failure_tags(question=question, state=state, metrics=metrics)
    final_answer = state.final_answer.model_dump(mode="json") if state.final_answer is not None else None
    tool_output = state.tool_output.model_dump(mode="json") if state.tool_output is not None else None

    return {
        "question_id": question.question_id,
        "question": question.question,
        "mode": mode.value,
        "case_slice": question.case_slice.value,
        "challenge_level": question.challenge_level.value,
        "expected": question.model_dump(mode="json"),
        "run_status": "completed",
        "run_id": state.run_id,
        "classification_source": state.classification_source,
        "query_type": state.query_type.value if state.query_type is not None else None,
        "retrieved_doc_ids": list(state.retrieved_doc_ids),
        "retrieved_chunk_ids": [chunk.chunk_id for chunk in state.retrieved_chunks],
        "retrieved_chunks": [chunk.model_dump(mode="json") for chunk in state.retrieved_chunks],
        "tool_used": state.tool_output is not None,
        "tool_mode": state.tool_mode.value if state.tool_mode is not None else None,
        "tool_output": tool_output,
        "final_answer": final_answer,
        "confidence": state.confidence.value if state.confidence is not None else None,
        "abstained": state.abstained,
        "unsupported_reason": state.unsupported_reason,
        "latency_ms": state.latency_ms,
        "evidence_composition": _actual_answer_mode(state.final_answer.evidence if state.final_answer is not None else []),
        "metrics": {name: metric.model_dump(mode="json") for name, metric in metrics.items()},
        "failure_tags": failure_tags,
        "automatic_pass": _automatic_pass(metrics),
        "runtime_error": None,
    }


def _build_runtime_error_result(
    *,
    question: EvalQuestion,
    mode: EvalRunMode,
    error: Exception,
) -> dict[str, Any]:
    error_message = f"{type(error).__name__}: {error}"
    metrics = {
        "route_and_tool_correctness": _error_metric(error_message),
        "retrieval_source_coverage": _error_metric(error_message),
        "groundedness_structure": _error_metric(error_message),
        "numeric_consistency": _error_metric(error_message),
        "abstention_correctness": _error_metric(error_message),
        "bounded_answer_adequacy": _error_metric(error_message),
    }
    return {
        "question_id": question.question_id,
        "question": question.question,
        "mode": mode.value,
        "case_slice": question.case_slice.value,
        "challenge_level": question.challenge_level.value,
        "expected": question.model_dump(mode="json"),
        "run_status": "runtime_error",
        "run_id": None,
        "classification_source": None,
        "query_type": None,
        "retrieved_doc_ids": [],
        "retrieved_chunk_ids": [],
        "retrieved_chunks": [],
        "tool_used": False,
        "tool_mode": None,
        "tool_output": None,
        "final_answer": None,
        "confidence": None,
        "abstained": None,
        "unsupported_reason": None,
        "latency_ms": None,
        "evidence_composition": "unavailable",
        "metrics": {name: metric.model_dump(mode="json") for name, metric in metrics.items()},
        "failure_tags": ["runtime_error"],
        "automatic_pass": False,
        "runtime_error": error_message,
    }


def _score_question(*, question: EvalQuestion, state: Any) -> dict[str, EvalMetricResult]:
    return {
        "route_and_tool_correctness": _score_route_and_tool(question=question, state=state),
        "retrieval_source_coverage": _score_retrieval_source_coverage(question=question, state=state),
        "groundedness_structure": _score_groundedness_structure(question=question, state=state),
        "numeric_consistency": _score_numeric_consistency(question=question, state=state),
        "abstention_correctness": _score_abstention(question=question, state=state),
        "bounded_answer_adequacy": _score_answer_adequacy(question=question, state=state),
    }


def _score_route_and_tool(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    details = {
        "expected_query_type": question.question_type.value,
        "actual_query_type": state.query_type.value if state.query_type is not None else None,
        "expected_tool_use": question.expected_tool_use,
        "actual_tool_use": state.tool_output is not None,
        "expected_tool_mode": question.expected_tool_mode.value if question.expected_tool_mode is not None else None,
        "actual_tool_mode": state.tool_mode.value if state.tool_mode is not None else None,
    }
    failures: list[str] = []
    if state.query_type != question.question_type:
        failures.append("query_type_mismatch")
    if (state.tool_output is not None) != question.expected_tool_use:
        failures.append("tool_use_mismatch")
    if question.expected_tool_use and state.tool_mode != question.expected_tool_mode:
        failures.append("tool_mode_mismatch")
    if not failures:
        return EvalMetricResult(
            status=EvalMetricStatus.PASS,
            details=details,
        )
    return EvalMetricResult(
        status=EvalMetricStatus.FAIL,
        reason=", ".join(failures),
        details=details,
    )


def _score_retrieval_source_coverage(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    if not question.expected_sources:
        return EvalMetricResult(
            status=EvalMetricStatus.NOT_APPLICABLE,
            reason="No expected sources were declared for this eval question.",
        )

    evidence_items = state.final_answer.evidence if state.final_answer is not None else []
    evidence_source_ids = {item.source_id for item in evidence_items}
    evidence_doc_ids = {item.doc_id for item in evidence_items if item.doc_id is not None}
    retrieved_doc_ids = set(state.retrieved_doc_ids)

    missing_sources = []
    for expected_source in question.expected_sources:
        if expected_source in retrieved_doc_ids:
            continue
        if expected_source in evidence_source_ids:
            continue
        if expected_source in evidence_doc_ids:
            continue
        missing_sources.append(expected_source)

    details = {
        "expected_sources": list(question.expected_sources),
        "retrieved_doc_ids": sorted(retrieved_doc_ids),
        "evidence_source_ids": sorted(evidence_source_ids),
        "evidence_doc_ids": sorted(value for value in evidence_doc_ids if value is not None),
    }
    if not missing_sources:
        return EvalMetricResult(status=EvalMetricStatus.PASS, details=details)
    details["missing_sources"] = missing_sources
    return EvalMetricResult(
        status=EvalMetricStatus.FAIL,
        reason="Missing expected evidence sources.",
        details=details,
    )


def _score_groundedness_structure(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    actual_mode = _actual_answer_mode(state.final_answer.evidence if state.final_answer is not None else [])
    details = {
        "expected_answer_mode": question.expected_answer_mode.value,
        "actual_answer_mode": actual_mode,
        "abstained": state.abstained,
    }
    if question.expected_answer_mode.value == "abstain":
        if state.abstained:
            return EvalMetricResult(status=EvalMetricStatus.PASS, details=details)
        return EvalMetricResult(
            status=EvalMetricStatus.FAIL,
            reason="Expected abstention but the workflow returned a supported answer.",
            details=details,
        )
    if state.abstained:
        return EvalMetricResult(
            status=EvalMetricStatus.FAIL,
            reason="Returned abstention for a question that expected a supported answer.",
            details=details,
        )
    if actual_mode == question.expected_answer_mode.value:
        return EvalMetricResult(status=EvalMetricStatus.PASS, details=details)
    return EvalMetricResult(
        status=EvalMetricStatus.FAIL,
        reason="Evidence composition does not match the expected answer mode.",
        details=details,
    )


def _score_numeric_consistency(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    if question.numeric_check is None:
        return EvalMetricResult(
            status=EvalMetricStatus.NOT_APPLICABLE,
            reason="No numeric check was declared for this eval question.",
        )

    if state.tool_output is None or state.final_answer is None:
        return EvalMetricResult(
            status=EvalMetricStatus.FAIL,
            reason="Required tool-backed state is missing for numeric evaluation.",
            details={"expected_tool_use": question.expected_tool_use},
        )

    resolved_payload = {
        "numeric_summary": state.final_answer.numeric_summary,
        "tool_output": state.tool_output.model_dump(mode="json"),
    }
    missing_keys: list[str] = []
    mismatched_values: dict[str, dict[str, Any]] = {}
    direction_failures: dict[str, dict[str, Any]] = {}

    for dotted_path in question.numeric_check.expected_keys:
        if not _path_exists(resolved_payload, dotted_path):
            missing_keys.append(dotted_path)

    for dotted_path, expected_value in question.numeric_check.expected_values.items():
        found, actual_value = _resolve_path(resolved_payload, dotted_path)
        if not found or not _values_match(actual_value, expected_value):
            mismatched_values[dotted_path] = {
                "expected": expected_value,
                "actual": actual_value if found else None,
            }

    for dotted_path, expected_direction in question.numeric_check.expected_direction.items():
        found, actual_value = _resolve_path(resolved_payload, dotted_path)
        if not found or not _direction_matches(actual_value, expected_direction):
            direction_failures[dotted_path] = {
                "expected_direction": expected_direction,
                "actual": actual_value if found else None,
            }

    details = {
        "checked_keys": list(question.numeric_check.expected_keys),
        "checked_values": dict(question.numeric_check.expected_values),
        "checked_directions": dict(question.numeric_check.expected_direction),
    }
    if not missing_keys and not mismatched_values and not direction_failures:
        return EvalMetricResult(status=EvalMetricStatus.PASS, details=details)

    if missing_keys:
        details["missing_keys"] = missing_keys
    if mismatched_values:
        details["mismatched_values"] = mismatched_values
    if direction_failures:
        details["direction_failures"] = direction_failures
    return EvalMetricResult(
        status=EvalMetricStatus.FAIL,
        reason="Numeric output does not satisfy the declared checks.",
        details=details,
    )


def _score_abstention(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    if not question.expected_abstain and not state.abstained:
        return EvalMetricResult(
            status=EvalMetricStatus.NOT_APPLICABLE,
            reason="This eval question does not expect abstention.",
        )

    details = {
        "expected_abstain": question.expected_abstain,
        "actual_abstained": state.abstained,
        "expected_unsupported_reason": question.expected_unsupported_reason,
        "actual_unsupported_reason": state.unsupported_reason,
    }
    if question.expected_abstain and state.abstained:
        if state.unsupported_reason is None:
            return EvalMetricResult(
                status=EvalMetricStatus.FAIL,
                reason="Abstention did not include an unsupported_reason.",
                details=details,
            )
        if (
            question.expected_unsupported_reason is not None
            and state.unsupported_reason != question.expected_unsupported_reason
        ):
            return EvalMetricResult(
                status=EvalMetricStatus.FAIL,
                reason="Abstention reason did not match the expected unsupported_reason.",
                details=details,
            )
        return EvalMetricResult(status=EvalMetricStatus.PASS, details=details)

    if question.expected_abstain and not state.abstained:
        return EvalMetricResult(
            status=EvalMetricStatus.FAIL,
            reason="Expected abstention but the workflow answered the question.",
            details=details,
        )

    return EvalMetricResult(
        status=EvalMetricStatus.FAIL,
        reason="Unexpected abstention for a question that expected a supported answer.",
        details=details,
    )


def _score_answer_adequacy(*, question: EvalQuestion, state: Any) -> EvalMetricResult:
    if question.expected_abstain or state.abstained:
        return EvalMetricResult(
            status=EvalMetricStatus.NOT_APPLICABLE,
            reason="Adequacy review is not tracked for abstained answers.",
        )
    if not question.reference_points:
        return EvalMetricResult(
            status=EvalMetricStatus.NOT_APPLICABLE,
            reason="No reviewer reference points were declared for this question.",
        )
    return EvalMetricResult(
        status=EvalMetricStatus.NEEDS_REVIEW,
        reason="Reviewer check required for executive-answer adequacy.",
        details={"reference_points": list(question.reference_points)},
    )


def _derive_failure_tags(
    *,
    question: EvalQuestion,
    state: Any,
    metrics: dict[str, EvalMetricResult],
) -> list[str]:
    tags: list[str] = []
    route_metric = metrics["route_and_tool_correctness"]
    if route_metric.status is EvalMetricStatus.FAIL:
        if state.query_type != question.question_type:
            tags.append("classification_route_failure")
        if (state.tool_output is not None) != question.expected_tool_use or (
            question.expected_tool_use and state.tool_mode != question.expected_tool_mode
        ):
            tags.append("tool_use_failure")

    if metrics["retrieval_source_coverage"].status is EvalMetricStatus.FAIL:
        tags.append("retrieval_coverage_failure")
    if metrics["groundedness_structure"].status is EvalMetricStatus.FAIL:
        tags.append("evidence_composition_failure")
    if metrics["numeric_consistency"].status is EvalMetricStatus.FAIL:
        tags.append("numeric_consistency_failure")
    if question.expected_abstain != state.abstained or metrics["abstention_correctness"].status is EvalMetricStatus.FAIL:
        tags.append("abstention_failure")
    if state.unsupported_reason == "document_tool_mismatch":
        tags.append("document_tool_mismatch")
    if metrics["bounded_answer_adequacy"].status is EvalMetricStatus.FAIL:
        tags.append("answer_adequacy_failure")
    return _dedupe_preserve_order(tags)


def _build_summary(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    failure_tag_counts = Counter()
    confidence_alignment: dict[str, dict[str, Any]] = {}

    for result in question_results:
        failure_tag_counts.update(result["failure_tags"])

    deterministic_metrics = (
        "route_and_tool_correctness",
        "retrieval_source_coverage",
        "groundedness_structure",
        "numeric_consistency",
        "abstention_correctness",
    )
    for confidence in ("high", "medium", "low"):
        matching_results = [result for result in question_results if result["confidence"] == confidence]
        if not matching_results:
            continue
        automatic_pass_count = sum(1 for result in matching_results if result["automatic_pass"])
        confidence_alignment[confidence] = {
            "n": len(matching_results),
            "automatic_pass_count": automatic_pass_count,
            "automatic_pass_rate": round(automatic_pass_count / len(matching_results), 4),
        }

    metric_summaries = {
        metric_name: _aggregate_metric(question_results, metric_name)
        for metric_name in (
            *deterministic_metrics,
            "bounded_answer_adequacy",
        )
    }

    return {
        "total_questions": len(question_results),
        "completed_questions": sum(1 for result in question_results if result["run_status"] == "completed"),
        "runtime_error_count": sum(1 for result in question_results if result["run_status"] == "runtime_error"),
        "environment_blocked_count": 0,
        "failure_tag_counts": dict(sorted(failure_tag_counts.items())),
        "metric_summaries": metric_summaries,
        "confidence_alignment": confidence_alignment,
    }


def _aggregate_metric(question_results: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    counts = Counter()
    for result in question_results:
        counts[result["metrics"][metric_name]["status"]] += 1
    evaluated_count = counts[EvalMetricStatus.PASS.value] + counts[EvalMetricStatus.FAIL.value]
    pass_count = counts[EvalMetricStatus.PASS.value]
    return {
        "pass_count": pass_count,
        "fail_count": counts[EvalMetricStatus.FAIL.value],
        "error_count": counts[EvalMetricStatus.ERROR.value],
        "not_applicable_count": counts[EvalMetricStatus.NOT_APPLICABLE.value],
        "needs_review_count": counts[EvalMetricStatus.NEEDS_REVIEW.value],
        "evaluated_count": evaluated_count,
        "pass_rate": round(pass_count / evaluated_count, 4) if evaluated_count else None,
    }


def _build_slices(question_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "question_type": defaultdict(list),
        "case_slice": defaultdict(list),
        "expected_tool_mode": defaultdict(list),
        "answered_vs_abstained": defaultdict(list),
        "classification_source": defaultdict(list),
        "evidence_composition": defaultdict(list),
    }

    for result in question_results:
        grouped["question_type"][result["expected"]["question_type"]].append(result)
        grouped["case_slice"][result["case_slice"]].append(result)
        grouped["expected_tool_mode"][result["expected"]["expected_tool_mode"] or "none"].append(result)
        grouped["answered_vs_abstained"][_answered_slice_value(result)].append(result)
        grouped["classification_source"][result["classification_source"] or "unavailable"].append(result)
        grouped["evidence_composition"][result["evidence_composition"]].append(result)

    slice_output: dict[str, list[dict[str, Any]]] = {}
    for slice_name, slice_groups in grouped.items():
        slice_rows = []
        for slice_value, slice_results in sorted(slice_groups.items()):
            failure_counts = Counter()
            for result in slice_results:
                failure_counts.update(result["failure_tags"])
            slice_rows.append(
                {
                    "value": slice_value,
                    "n": len(slice_results),
                    "runtime_error_count": sum(1 for result in slice_results if result["run_status"] == "runtime_error"),
                    "metric_summaries": {
                        metric_name: _aggregate_metric(slice_results, metric_name)
                        for metric_name in (
                            "route_and_tool_correctness",
                            "retrieval_source_coverage",
                            "groundedness_structure",
                            "numeric_consistency",
                            "abstention_correctness",
                            "bounded_answer_adequacy",
                        )
                    },
                    "top_failure_tags": [
                        {"failure_tag": failure_tag, "count": count}
                        for failure_tag, count in failure_counts.most_common(5)
                    ],
                }
            )
        slice_output[slice_name] = slice_rows
    return slice_output


def _build_failure_examples(question_results: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    examples = []
    for result in question_results:
        if not result["failure_tags"]:
            continue
        examples.append(
            {
                "question_id": result["question_id"],
                "question": result["question"],
                "failure_tags": list(result["failure_tags"]),
                "run_status": result["run_status"],
                "query_type": result["query_type"],
                "classification_source": result["classification_source"],
                "tool_mode": result["tool_mode"],
                "unsupported_reason": result["unsupported_reason"],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _actual_answer_mode(evidence_items: list[AnswerEvidenceItem] | list[dict[str, Any]]) -> str:
    document_present = False
    computed_present = False
    for item in evidence_items:
        source_type = item.source_type.value if isinstance(item, AnswerEvidenceItem) else item.get("source_type")
        if source_type == "document":
            document_present = True
        elif source_type == "computed":
            computed_present = True

    if document_present and computed_present:
        return "document_plus_computed"
    if document_present:
        return "document_only"
    if computed_present:
        return "computed_only"
    return "none"


def _automatic_pass(metrics: dict[str, EvalMetricResult]) -> bool:
    for metric_name, metric in metrics.items():
        if metric_name == "bounded_answer_adequacy":
            continue
        if metric.status not in {EvalMetricStatus.PASS, EvalMetricStatus.NOT_APPLICABLE}:
            return False
    return True


def _answered_slice_value(result: dict[str, Any]) -> str:
    if result["run_status"] == "runtime_error":
        return "runtime_error"
    if result["abstained"]:
        return "abstained"
    return "answered"


def _error_metric(error_message: str) -> EvalMetricResult:
    return EvalMetricResult(
        status=EvalMetricStatus.ERROR,
        reason=error_message,
    )


def _path_exists(payload: dict[str, Any], dotted_path: str) -> bool:
    found, _ = _resolve_path(payload, dotted_path)
    return found


def _resolve_path(payload: Any, dotted_path: str) -> tuple[bool, Any]:
    current = payload
    for segment in dotted_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                return False, None
            current = current[segment]
            continue
        if isinstance(current, list):
            if not segment.isdigit():
                return False, None
            index = int(segment)
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def _values_match(actual_value: Any, expected_value: Any) -> bool:
    if isinstance(expected_value, float) and isinstance(actual_value, (float, int)):
        return math.isclose(float(actual_value), expected_value, rel_tol=1e-6, abs_tol=1e-6)
    return actual_value == expected_value


def _direction_matches(actual_value: Any, expected_direction: str) -> bool:
    if expected_direction == "positive":
        return isinstance(actual_value, (int, float)) and float(actual_value) > 0.0
    if expected_direction == "negative":
        return isinstance(actual_value, (int, float)) and float(actual_value) < 0.0
    if expected_direction == "true":
        return actual_value is True
    if expected_direction == "false":
        return actual_value is False
    if expected_direction == "non_decreasing":
        if not isinstance(actual_value, list) or not all(isinstance(item, (int, float)) for item in actual_value):
            return False
        return all(float(left) <= float(right) for left, right in zip(actual_value, actual_value[1:]))
    return False


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 6 evaluation workflow.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in EvalRunMode],
        default=EvalRunMode.CANONICAL.value,
        help="Evaluation mode. Canonical writes the project eval artifact; hermetic_verify is intended for injected tests.",
    )
    parser.add_argument("--questions-path", type=Path, default=None, help="Optional path to an eval JSONL file.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional output artifact path.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow the workflow's existing provider fallback path when the provider call fails.",
    )
    args = parser.parse_args(argv)

    artifact = run_eval(
        mode=EvalRunMode(args.mode),
        questions_path=args.questions_path,
        output_path=args.output_path,
        allow_fallback=args.allow_fallback,
    )

    print(
        json.dumps(
            {
                "mode": artifact["run_metadata"]["mode"],
                "status": artifact["run_metadata"]["status"],
                "total_questions": artifact["summary"]["total_questions"],
                "completed_questions": artifact["summary"]["completed_questions"],
                "runtime_error_count": artifact["summary"]["runtime_error_count"],
                "environment_blocked_count": artifact["summary"]["environment_blocked_count"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0 if artifact["run_metadata"]["status"] == EvalRunStatus.COMPLETED.value else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
