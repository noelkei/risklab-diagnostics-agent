from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from app import load_settings
from app import cli
from graph.provider import ProviderConfigurationError
from schemas import (
    AnswerEvidenceItem,
    ConfidenceLevel,
    DiagnosticsMetrics,
    DiagnosticsOutput,
    EvalRunMode,
    EvalRunStatus,
    EvidenceSourceType,
    FinalAnswer,
    GraphState,
    QueryType,
    RetrievalChunk,
    ScenarioDeltaMap,
    StressMetrics,
    StressOutput,
    ToolMode,
)
from tools import computed_source_id_for_mode


def _make_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    page: int,
    section_path: str,
    text: str,
    topic_tags: list[str],
    sparse_score: float | None = None,
    dense_score: float | None = None,
    fused_score: float | None = None,
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
        sparse_score=sparse_score,
        dense_score=dense_score,
        fused_score=fused_score,
    )


def _document_evidence(chunks: list[RetrievalChunk]) -> list[AnswerEvidenceItem]:
    return [
        AnswerEvidenceItem(
            source_type=EvidenceSourceType.DOCUMENT,
            source_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            page=chunk.page,
            support=chunk.text,
        )
        for chunk in chunks
    ]


def _settings_with_trace(tmp_path: Path):
    base_settings = load_settings()
    return replace(
        base_settings,
        paths=replace(
            base_settings.paths,
            trace_runs_path=tmp_path / "outputs" / "traces" / "runs.jsonl",
        ),
    )


def _build_state(
    *,
    query: str,
    query_type: QueryType,
    retrieved_chunks: list[RetrievalChunk],
    executive_answer: str,
    confidence: ConfidenceLevel,
    tool_mode: ToolMode | None = None,
    tool_output: DiagnosticsOutput | StressOutput | None = None,
    abstained: bool = False,
    unsupported_reason: str | None = None,
    review_flag: bool = False,
    classification_source: str = "heuristic",
) -> GraphState:
    evidence = _document_evidence(retrieved_chunks)
    if tool_output is not None and tool_mode is not None:
        evidence.append(
            AnswerEvidenceItem(
                source_type=EvidenceSourceType.COMPUTED,
                source_id=computed_source_id_for_mode(tool_mode),
                support=tool_output.summary,
            )
        )
    final_answer = FinalAnswer(
        executive_answer=executive_answer,
        evidence=evidence,
        numeric_summary={"tool_mode": tool_mode.value} if tool_mode is not None else None,
        limitations=["This workflow is limited to the frozen three-document corpus and prepared structured artifacts."],
        confidence=confidence,
        abstained=abstained,
        review_flag=review_flag,
    )
    return GraphState(
        query=query,
        run_id="run-phase7-cli",
        query_type=query_type,
        classification_source=classification_source,  # type: ignore[arg-type]
        retrieved_chunks=retrieved_chunks,
        retrieved_doc_ids=[chunk.doc_id for chunk in retrieved_chunks],
        tool_required=tool_output is not None,
        tool_mode=tool_mode,
        tool_output=tool_output,
        final_answer=final_answer,
        confidence=confidence,
        abstained=abstained,
        review_flag=review_flag,
        unsupported_reason=unsupported_reason,
        latency_ms=123,
    )


def _completed_eval_artifact(*, mode: str = "canonical", output_path: str | None = None) -> dict[str, object]:
    return {
        "run_metadata": {
            "mode": mode,
            "status": EvalRunStatus.COMPLETED.value,
            "started_at_utc": "2026-03-24T09:00:00Z",
            "finished_at_utc": "2026-03-24T09:01:00Z",
            "questions_path": "/tmp/eval_questions.jsonl",
            "output_path": output_path,
            "dependency_status": {
                "provider_configured": True,
                "provider_mode": "real",
                "retriever_mode": "real",
            },
            "total_questions": 3,
        },
        "summary": {
            "total_questions": 3,
            "completed_questions": 3,
            "runtime_error_count": 0,
            "environment_blocked_count": 0,
            "failure_tag_counts": {"retrieval_coverage_failure": 1},
            "metric_summaries": {
                "route_and_tool_correctness": {"pass_count": 3, "evaluated_count": 3, "pass_rate": 1.0},
                "retrieval_source_coverage": {"pass_count": 2, "evaluated_count": 3, "pass_rate": 0.6667},
                "groundedness_structure": {"pass_count": 3, "evaluated_count": 3, "pass_rate": 1.0},
                "numeric_consistency": {"pass_count": 1, "evaluated_count": 1, "pass_rate": 1.0},
                "abstention_correctness": {"pass_count": 1, "evaluated_count": 1, "pass_rate": 1.0},
            },
            "confidence_alignment": {},
        },
        "slices": {},
        "questions": [],
        "failure_examples": [],
    }


def test_phase7_cli_query_policy_output_is_document_grounded(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    chunk = _make_chunk(
        chunk_id="chunk:sr11:p4:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=4,
        section_path="Governance, Policies, and Controls",
        text="SR 11-7 expects strong governance, effective challenge, and ongoing monitoring.",
        topic_tags=["governance", "validation"],
        sparse_score=2.1,
        dense_score=0.8,
        fused_score=0.18,
    )
    state = _build_state(
        query="What governance expectations does SR11-7 set for model validation?",
        query_type=QueryType.POLICY,
        retrieved_chunks=[chunk],
        executive_answer="SR 11-7 expects strong governance and effective challenge around model validation.",
        confidence=ConfidenceLevel.HIGH,
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", lambda *args, **kwargs: state)

    result = CliRunner().invoke(cli.app, ["query", state.query])

    assert result.exit_code == 0
    assert "ANSWERED" in result.stdout
    assert "Route: policy" in result.stdout
    assert "Grounding: document_only" in result.stdout
    assert "Sources used: SR11_7_Model_Risk_Management (1 chunk)" in result.stdout
    assert "Document Evidence" in result.stdout
    assert "SR11_7_Model_Risk_Management" in result.stdout
    assert "Computed Evidence" not in result.stdout
    assert "Run: run-phase7-cli" in result.stdout
    assert f"Trace: {settings.paths.trace_runs_path}" in result.stdout


def test_phase7_cli_query_numeric_verbose_output_separates_computed_evidence(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    chunk = _make_chunk(
        chunk_id="chunk:model:p8:1",
        doc_id="Model_Validation_Report",
        page=8,
        section_path="Diagnostics and Monitoring",
        text="Material PSI drift appears in revol util and calibration weakens out of time.",
        topic_tags=["drift", "calibration", "validation"],
        sparse_score=1.7,
        dense_score=0.9,
        fused_score=0.19,
    )
    tool_output = DiagnosticsOutput(
        summary="Champion OOT ranking remains usable, but calibration weakened and material drift requires monitoring.",
        metrics=DiagnosticsMetrics(
            top_drift_features=[
                {"feature": "revol util", "flag": "RED", "psi": 0.2902},
                {"feature": "application type", "flag": "AMBER", "psi": 0.2457},
                {"feature": "int rate", "flag": "AMBER", "psi": 0.1271},
            ],
            score_shift={
                "champion_logit": {"score_psi": 0.025877},
                "challenger_lgbm": {"score_psi": 0.015908},
            },
            calibration={
                "champion_oot": {"calibration_slope": 0.837},
                "challenger_oot": {"calibration_slope": 0.9739},
            },
        ),
        quality_flags=["monitor_material_drift"],
        limitations=["Diagnostics are bounded to the frozen artifact set."],
    )
    state = _build_state(
        query="How do the champion and challenger compare on OOT calibration slope?",
        query_type=QueryType.NUMERIC,
        retrieved_chunks=[chunk],
        executive_answer="The challenger is better calibrated out of time than the champion.",
        confidence=ConfidenceLevel.MEDIUM,
        tool_mode=ToolMode.DIAGNOSTICS,
        tool_output=tool_output,
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", lambda *args, **kwargs: state)

    result = CliRunner().invoke(cli.app, ["query", state.query, "--verbose"])

    assert result.exit_code == 0
    assert "Grounding: document_plus_computed" in result.stdout
    assert (
        "Sources used: Model_Validation_Report (1 chunk); risk_diagnostics_tool:diagnostics (computed)"
        in result.stdout
    )
    assert "Document Evidence" in result.stdout
    assert "Computed Evidence" in result.stdout
    assert "Source: risk_diagnostics_tool:diagnostics" in result.stdout
    assert "Top drift features:" in result.stdout
    assert "OOT calibration slopes:" in result.stdout
    assert "Score-shift PSI:" in result.stdout
    assert "Classification source: heuristic" in result.stdout
    assert "Quality flags: monitor_material_drift" in result.stdout


def test_phase7_cli_query_mixed_output_includes_document_and_stress_evidence(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    chunks = [
        _make_chunk(
            chunk_id="chunk:model:p10:1",
            doc_id="Model_Validation_Report",
            page=10,
            section_path="Stress Testing Results",
            text="Mild and severe scenarios increase PD and EL proxy relative to baseline.",
            topic_tags=["stress_testing", "sensitivity"],
        ),
        _make_chunk(
            chunk_id="chunk:basel:p10:1",
            doc_id="Basel_Stress_Testing_Principles_2018",
            page=10,
            section_path="Stress Testing Principles",
            text="Basel expects stress results to feed governance, scenario design, and risk management.",
            topic_tags=["stress_testing", "governance"],
        ),
    ]
    tool_output = StressOutput(
        summary="Mild and severe scenarios increase mean PD, tail PD, and EL proxy monotonically relative to baseline.",
        metrics=StressMetrics(
            baseline={"mean_pd": 0.190617},
            mild={"mean_pd": 0.195145},
            severe={"mean_pd": 0.200342},
            delta_mean_pd=ScenarioDeltaMap(mild=0.004528, severe=0.009725),
            delta_tail_pd=ScenarioDeltaMap(mild=0.00918, severe=0.018973),
            delta_el_proxy=ScenarioDeltaMap(mild=13.718138, severe=29.387),
            monotonicity_passed=True,
        ),
    )
    state = _build_state(
        query="Based on Basel principles and the stress scenarios, how do mild and severe cases change PD and what does that imply for governance use?",
        query_type=QueryType.MIXED,
        retrieved_chunks=chunks,
        executive_answer="Stress raises PD monotonically, and Basel-style governance should treat the results as an oversight input rather than a standalone decision rule.",
        confidence=ConfidenceLevel.MEDIUM,
        tool_mode=ToolMode.STRESS,
        tool_output=tool_output,
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", lambda *args, **kwargs: state)

    result = CliRunner().invoke(cli.app, ["query", state.query])

    assert result.exit_code == 0
    assert "Grounding: document_plus_computed" in result.stdout
    assert (
        "Sources used: Model_Validation_Report (1 chunk); Basel_Stress_Testing_Principles_2018 (1 chunk); "
        "risk_diagnostics_tool:stress (computed)"
    ) in result.stdout
    assert "Document Evidence" in result.stdout
    assert "Computed Evidence" in result.stdout
    assert "Source: risk_diagnostics_tool:stress" in result.stdout
    assert "Mean PD delta:" in result.stdout
    assert "Monotonicity: passed" in result.stdout
    assert "Basel_Stress_Testing_Principles_2018" in result.stdout


def test_phase7_cli_query_abstention_returns_exit_zero(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    state = _build_state(
        query="Explain the PSI drift findings and how mild and severe stress scenarios change EL proxy in one answer.",
        query_type=QueryType.UNSUPPORTED,
        retrieved_chunks=[],
        executive_answer="I can't support a single answer because this MVP handles only one quantitative mode per query.",
        confidence=ConfidenceLevel.LOW,
        abstained=True,
        unsupported_reason="multi_mode_request",
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", lambda *args, **kwargs: state)

    result = CliRunner().invoke(cli.app, ["query", state.query])

    assert result.exit_code == 0
    assert "ABSTAINED" in result.stdout
    assert "Unsupported reason: multi_mode_request" in result.stdout


def test_phase7_cli_query_clean_suppresses_known_backend_noise(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    chunk = _make_chunk(
        chunk_id="chunk:basel:p8:1",
        doc_id="Basel_Stress_Testing_Principles_2018",
        page=8,
        section_path="Stress testing governance",
        text="Governance should identify stakeholders and support challenge.",
        topic_tags=["governance", "stress_testing"],
    )
    state = _build_state(
        query="Under the Basel stress testing principles, what responsibilities do the board, senior management, and internal audit have in governing the stress testing framework?",
        query_type=QueryType.POLICY,
        retrieved_chunks=[chunk],
        executive_answer="Basel expects formal governance, challenge, and oversight around the stress testing framework.",
        confidence=ConfidenceLevel.HIGH,
    )

    def _stub_run_query(*args, **kwargs):
        print(
            "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads."
        )
        print("Loading weights: 100%|███████████████████████████████████████████████████████████████|", file=sys.stderr)
        print("BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2")
        print("Key                     | Status     |  | ")
        print("------------------------+------------+--+-")
        print("embeddings.position_ids | UNEXPECTED |  | ")
        print("Notes:")
        print("- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.")
        return state

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", _stub_run_query)

    result = CliRunner().invoke(cli.app, ["query", state.query, "--clean"])

    assert result.exit_code == 0
    assert "ANSWERED" in result.stdout
    assert "HF Hub" not in result.stdout
    assert "Loading weights:" not in result.stdout
    assert "BertModel LOAD REPORT" not in result.stdout


def test_phase7_cli_query_json_is_parseable_and_passes_allow_fallback(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    chunk = _make_chunk(
        chunk_id="chunk:sr11:p4:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=4,
        section_path="Governance, Policies, and Controls",
        text="SR 11-7 expects strong governance, effective challenge, and ongoing monitoring.",
        topic_tags=["governance", "validation"],
    )
    state = _build_state(
        query="What governance expectations does SR11-7 set for model validation?",
        query_type=QueryType.POLICY,
        retrieved_chunks=[chunk],
        executive_answer="SR 11-7 expects strong governance and effective challenge around model validation.",
        confidence=ConfidenceLevel.HIGH,
    )
    captured: dict[str, object] = {}

    def _stub_run_query(*args, **kwargs):
        captured["allow_fallback"] = kwargs["allow_fallback"]
        captured["settings"] = kwargs["settings"]
        return state

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "run_query", _stub_run_query)

    result = CliRunner().invoke(cli.app, ["query", state.query, "--json", "--allow-fallback"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "answered"
    assert payload["query_type"] == "policy"
    assert payload["tool_used"] is False
    assert payload["trace_path"] == str(settings.paths.trace_runs_path)
    assert payload["document_evidence"][0]["section_path"] == "Governance, Policies, and Controls"
    assert captured["allow_fallback"] is True
    assert captured["settings"] == settings


def test_phase7_cli_query_environment_blocked_returns_clean_error(tmp_path, monkeypatch) -> None:
    settings = _settings_with_trace(tmp_path)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "run_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProviderConfigurationError("Gemini requires GEMINI_API_KEY to be configured.")
        ),
    )

    result = CliRunner().invoke(cli.app, ["query", "Ambiguous unsupported question"])

    assert result.exit_code == 1
    assert "ENVIRONMENT_BLOCKED" in result.stdout
    assert "GEMINI_API_KEY" in result.stdout
    assert "Hint:" in result.stdout


def test_phase7_cli_eval_json_uses_runner_artifact_and_exit_zero(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    questions_path = tmp_path / "eval_questions.jsonl"
    output_path = tmp_path / "eval_results.json"
    artifact = _completed_eval_artifact(mode="hermetic_verify", output_path=str(output_path))

    def _stub_run_eval(*, mode, questions_path=None, output_path=None, allow_fallback=False):
        captured["mode"] = mode
        captured["questions_path"] = questions_path
        captured["output_path"] = output_path
        captured["allow_fallback"] = allow_fallback
        return artifact

    monkeypatch.setattr(cli, "run_eval", _stub_run_eval)

    result = CliRunner().invoke(
        cli.app,
        [
            "eval",
            "--mode",
            "hermetic_verify",
            "--questions-path",
            str(questions_path),
            "--output-path",
            str(output_path),
            "--allow-fallback",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_metadata"]["status"] == "completed"
    assert captured["mode"] == EvalRunMode.HERMETIC_VERIFY
    assert captured["questions_path"] == questions_path
    assert captured["output_path"] == output_path
    assert captured["allow_fallback"] is True


def test_phase7_cli_eval_output_includes_failure_examples(monkeypatch) -> None:
    artifact = _completed_eval_artifact(output_path="/tmp/eval_results.json")
    artifact["failure_examples"] = [
        {
            "question_id": "mixed-001",
            "question": "Based on SR11-7 and the validation report, what governance action follows from the observed calibration degradation?",
            "failure_tags": ["retrieval_coverage_failure"],
            "run_status": "completed",
            "query_type": "mixed",
            "classification_source": "heuristic",
            "tool_mode": "diagnostics",
            "unsupported_reason": None,
        }
    ]
    monkeypatch.setattr(cli, "run_eval", lambda **kwargs: artifact)

    result = CliRunner().invoke(cli.app, ["eval"])

    assert result.exit_code == 0
    assert "Failure Examples" in result.stdout
    assert "mixed-001 | tags=retrieval_coverage_failure | query_type=mixed | tool=diagnostics" in result.stdout
    assert "Based on SR11-7 and the validation report" in result.stdout


def test_phase7_cli_eval_exit_semantics_are_explicit(monkeypatch) -> None:
    runner = CliRunner()

    for status, expected_exit_code in (
        (EvalRunStatus.COMPLETED.value, 0),
        (EvalRunStatus.ENVIRONMENT_BLOCKED.value, 1),
        (EvalRunStatus.RUNNER_FAILED.value, 1),
    ):
        artifact = _completed_eval_artifact()
        artifact["run_metadata"]["status"] = status
        artifact["run_metadata"]["blocker_reason"] = "Synthetic blocker." if status != "completed" else None
        if status != "completed":
            artifact["summary"]["completed_questions"] = 0
        monkeypatch.setattr(cli, "run_eval", lambda **kwargs: artifact)

        result = runner.invoke(cli.app, ["eval"])

        assert result.exit_code == expected_exit_code
        assert f"EVAL {status.upper()}" in result.stdout
        if status != "completed":
            assert "Reason: Synthetic blocker." in result.stdout
