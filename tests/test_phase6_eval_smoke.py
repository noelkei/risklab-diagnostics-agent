from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import eval.runner as runner_module
import pytest

from app import load_settings
from eval import EvalDatasetError, load_eval_questions, run_eval
from graph.workflow import _classify_query_heuristically
from schemas import EvalRunMode, QueryClassification, QueryType, RetrievalChunk
from traces import JsonlTraceStore


def _make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    page: int,
    section_path: str,
    text: str,
    topic_tags: list[str],
) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        page=page,
        section_path=section_path,
        chunk_type="text",
        topic_tags=topic_tags,
        authority_level="internal" if doc_id == "Model_Validation_Report" else "supervisory",
        document_role="internal_case_primary_source"
        if doc_id == "Model_Validation_Report"
        else "supervisory_guidance",
    )


POLICY_CHUNKS = [
    _make_chunk(
        chunk_id="chunk:sr11:p4:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=4,
        section_path="Governance, Policies, and Controls",
        text="SR 11-7 expects strong governance, effective challenge, and ongoing monitoring around model validation.",
        topic_tags=["governance", "validation"],
    ),
    _make_chunk(
        chunk_id="chunk:sr11:p3:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=3,
        section_path="Model Validation",
        text="Model limitations and uncertainty should be understood and documented as part of model use.",
        topic_tags=["governance", "validation"],
    ),
]


DIAGNOSTICS_CHUNKS = [
    _make_chunk(
        chunk_id="chunk:model:p8:1",
        doc_id="Model_Validation_Report",
        page=8,
        section_path="Diagnostics and Monitoring",
        text="The report highlights material PSI drift in revol util and weaker OOT calibration for the champion model.",
        topic_tags=["drift", "calibration", "validation"],
    ),
    _make_chunk(
        chunk_id="chunk:model:p12:1",
        doc_id="Model_Validation_Report",
        page=12,
        section_path="Calibration Review",
        text="Calibration slope and intercept worsen out of time while the challenger remains better calibrated.",
        topic_tags=["calibration", "challenger", "performance"],
    ),
]


class StubRetriever:
    def search(self, query: str) -> list[RetrievalChunk]:
        lowered = query.lower()
        if "governance expectations" in lowered:
            return [chunk.model_copy(deep=True) for chunk in POLICY_CHUNKS]
        if "drift features" in lowered:
            return [chunk.model_copy(deep=True) for chunk in DIAGNOSTICS_CHUNKS]
        return []


class StubProvider:
    def classify_query(self, query: str) -> QueryClassification:
        return QueryClassification(
            query_type=QueryType.POLICY,
            tool_required=False,
            reason_code="provider_policy",
        )

    def synthesize_executive_answer(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence,
        numeric_summary,
        limitations,
    ) -> str:
        if numeric_summary and isinstance(numeric_summary.get("summary"), str):
            return str(numeric_summary["summary"])
        if evidence:
            return evidence[0].support
        return f"Stub synthesis for {query_type.value}."


class FailingSynthesisProvider(StubProvider):
    def synthesize_executive_answer(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence,
        numeric_summary,
        limitations,
    ) -> str:
        if numeric_summary is not None:
            raise RuntimeError("synthetic eval failure")
        return super().synthesize_executive_answer(
            query=query,
            query_type=query_type,
            evidence=evidence,
            numeric_summary=numeric_summary,
            limitations=limitations,
        )


def _write_question_subset(path: Path, question_ids: list[str]) -> None:
    questions_by_id = {question.question_id: question for question in load_eval_questions()}
    rows = [questions_by_id[question_id].model_dump(mode="json") for question_id in question_ids]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def test_phase6_dataset_has_expected_distribution() -> None:
    questions = load_eval_questions()

    assert len(questions) == 18
    assert Counter(question.question_type.value for question in questions) == {
        "policy": 4,
        "numeric": 7,
        "mixed": 4,
        "unsupported": 3,
    }
    assert Counter(question.case_slice.value for question in questions) == {
        "policy": 4,
        "diagnostics": 4,
        "stress": 3,
        "mixed": 4,
        "unsupported": 3,
    }
    assert sum(question.challenge_level.value == "adversarial_light" for question in questions) >= 2
    assert sum(
        question.challenge_level.value in {"borderline_abstain", "multi_mode_conflict"}
        for question in questions
    ) >= 3


def test_phase6_dataset_rows_align_with_heuristic_router() -> None:
    for question in load_eval_questions():
        classification = _classify_query_heuristically(question.question)

        assert classification is not None
        assert classification.query_type == question.question_type
        assert classification.tool_required == question.expected_tool_use
        assert classification.tool_mode == question.expected_tool_mode


def test_phase6_hermetic_runner_produces_structured_results(tmp_path: Path) -> None:
    questions_path = tmp_path / "eval_questions.jsonl"
    output_path = tmp_path / "eval_results.json"
    trace_store = JsonlTraceStore(path=tmp_path / "outputs" / "traces" / "runs.jsonl")
    _write_question_subset(
        questions_path,
        ["policy-001", "numeric-001", "unsupported-001"],
    )

    artifact = run_eval(
        mode=EvalRunMode.HERMETIC_VERIFY,
        questions_path=questions_path,
        output_path=output_path,
        provider=StubProvider(),
        retriever=StubRetriever(),
        trace_store=trace_store,
    )

    assert artifact["run_metadata"]["mode"] == "hermetic_verify"
    assert artifact["run_metadata"]["status"] == "completed"
    assert artifact["summary"]["total_questions"] == 3
    assert artifact["summary"]["completed_questions"] == 3
    assert artifact["summary"]["runtime_error_count"] == 0
    assert artifact["summary"]["metric_summaries"]["route_and_tool_correctness"]["pass_count"] == 3
    assert artifact["summary"]["metric_summaries"]["groundedness_structure"]["pass_count"] == 3
    assert artifact["summary"]["metric_summaries"]["numeric_consistency"]["pass_count"] == 1
    assert artifact["summary"]["metric_summaries"]["abstention_correctness"]["pass_count"] == 1
    assert artifact["failure_examples"] == []
    assert artifact["slices"]["question_type"]
    assert artifact["slices"]["case_slice"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["run_metadata"]["mode"] == "hermetic_verify"


def test_phase6_canonical_rate_control_retries_once_on_rate_limit(monkeypatch) -> None:
    sleep_calls: list[float] = []
    call_count = 0

    def fake_run_query(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
        return {"status": "ok"}

    monkeypatch.setattr(runner_module, "run_query", fake_run_query)
    monkeypatch.setattr(runner_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = runner_module._run_query_with_rate_control(
        question="What governance expectations does SR11-7 set for model validation?",
        mode=EvalRunMode.CANONICAL,
        settings=load_settings(),
        retriever=None,
        provider=None,
        trace_store=None,
        allow_fallback=False,
        delay_before_seconds=runner_module._CANONICAL_INTER_QUERY_DELAY_SECONDS,
    )

    assert result == {"status": "ok"}
    assert call_count == 2
    assert sleep_calls == [
        runner_module._CANONICAL_INTER_QUERY_DELAY_SECONDS,
        runner_module._CANONICAL_RATE_LIMIT_RETRY_DELAY_SECONDS,
    ]


def test_phase6_canonical_preflight_blocks_without_provider_config(tmp_path: Path) -> None:
    base_settings = load_settings()
    blocked_settings = replace(
        base_settings,
        provider=replace(base_settings.provider, api_key=None),
    )
    questions_path = tmp_path / "eval_questions.jsonl"
    output_path = tmp_path / "eval_results.json"
    _write_question_subset(
        questions_path,
        ["policy-001", "numeric-001", "unsupported-001"],
    )

    artifact = run_eval(
        mode=EvalRunMode.CANONICAL,
        settings=blocked_settings,
        questions_path=questions_path,
        output_path=output_path,
    )

    assert artifact["run_metadata"]["mode"] == "canonical"
    assert artifact["run_metadata"]["status"] == "environment_blocked"
    assert artifact["summary"]["environment_blocked_count"] == 3
    assert artifact["questions"] == []
    assert artifact["slices"] == {}
    assert artifact["failure_examples"] == []
    assert "GEMINI_API_KEY" in artifact["run_metadata"]["blocker_reason"]
    assert json.loads(output_path.read_text(encoding="utf-8"))["run_metadata"]["status"] == "environment_blocked"


def test_phase6_duplicate_question_ids_fail_fast(tmp_path: Path) -> None:
    output_path = tmp_path / "eval_results.json"
    duplicate_row = {
        "question_id": "dup-001",
        "question": "Should we buy NVDA stock next week?",
        "question_type": "unsupported",
        "case_slice": "unsupported",
        "challenge_level": "standard",
        "expected_sources": [],
        "expected_answer_mode": "abstain",
        "expected_tool_use": False,
        "expected_tool_mode": None,
        "expected_abstain": True,
        "expected_unsupported_reason": "out_of_scope",
        "numeric_check": None,
        "reference_points": [],
        "notes": "duplicate id test",
    }
    questions_path = tmp_path / "eval_questions.jsonl"
    questions_path.write_text(
        "\n".join(json.dumps(duplicate_row, ensure_ascii=True) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    artifact = run_eval(
        mode=EvalRunMode.HERMETIC_VERIFY,
        questions_path=questions_path,
        output_path=output_path,
        provider=StubProvider(),
        retriever=StubRetriever(),
    )

    assert artifact["run_metadata"]["status"] == "runner_failed"
    assert artifact["summary"]["total_questions"] == 0
    assert artifact["questions"] == []


def test_phase6_contradictory_eval_row_fails_fast(tmp_path: Path) -> None:
    invalid_row = {
        "question_id": "bad-001",
        "question": "What are the calibration issues?",
        "question_type": "numeric",
        "case_slice": "diagnostics",
        "challenge_level": "standard",
        "expected_sources": ["Model_Validation_Report"],
        "expected_answer_mode": "document_plus_computed",
        "expected_tool_use": True,
        "expected_tool_mode": None,
        "expected_abstain": False,
        "expected_unsupported_reason": None,
        "numeric_check": {"expected_keys": ["numeric_summary.tool_mode"], "expected_values": {}, "expected_direction": {}},
        "reference_points": ["Should use the diagnostics tool."],
        "notes": "invalid row test",
    }
    questions_path = tmp_path / "eval_questions.jsonl"
    questions_path.write_text(json.dumps(invalid_row, ensure_ascii=True) + "\n", encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="expected_tool_mode"):
        load_eval_questions(questions_path=questions_path)


def test_phase6_runtime_error_rows_are_excluded_from_metric_rates(tmp_path: Path) -> None:
    questions_path = tmp_path / "eval_questions.jsonl"
    output_path = tmp_path / "eval_results.json"
    trace_store = JsonlTraceStore(path=tmp_path / "outputs" / "traces" / "runs.jsonl")
    _write_question_subset(
        questions_path,
        ["policy-001", "numeric-001"],
    )

    artifact = run_eval(
        mode=EvalRunMode.HERMETIC_VERIFY,
        questions_path=questions_path,
        output_path=output_path,
        provider=FailingSynthesisProvider(),
        retriever=StubRetriever(),
        trace_store=trace_store,
    )

    route_summary = artifact["summary"]["metric_summaries"]["route_and_tool_correctness"]

    assert artifact["summary"]["total_questions"] == 2
    assert artifact["summary"]["completed_questions"] == 1
    assert artifact["summary"]["runtime_error_count"] == 1
    assert route_summary["pass_count"] == 1
    assert route_summary["error_count"] == 1
    assert route_summary["evaluated_count"] == 1
    assert route_summary["pass_rate"] == 1.0
    assert any(result["run_status"] == "runtime_error" for result in artifact["questions"])
    assert artifact["failure_examples"][0]["run_status"] == "runtime_error"
