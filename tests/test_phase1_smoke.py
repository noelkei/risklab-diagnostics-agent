from __future__ import annotations

import json
import os

import yaml

from app import config as config_module
from app.config import load_settings
from schemas import (
    AnswerEvidenceItem,
    ConfidenceLevel,
    DiagnosticsMetrics,
    DiagnosticsOutput,
    EvalCaseSlice,
    EvalChallengeLevel,
    EvalQuestion,
    EvidenceSourceType,
    ExpectedAnswerMode,
    FinalAnswer,
    GraphState,
    NumericCheck,
    QueryClassification,
    QueryType,
    RetrievalChunk,
    StressMetrics,
    StressOutput,
    ToolMode,
    TraceRecord,
)


def test_load_settings_smoke() -> None:
    settings = load_settings()
    expected_model_name = os.getenv("GEMINI_MODEL_NAME", config_module.DEFAULT_GEMINI_MODEL_NAME)

    assert settings.project_name == "risklab-diagnostics-agent"
    assert settings.provider.provider_name == "gemini"
    assert settings.provider.sdk_package == "google-genai"
    assert settings.provider.api_key_env_var == "GEMINI_API_KEY"
    assert settings.provider.model_name_env_var == "GEMINI_MODEL_NAME"
    assert settings.provider.model_name == expected_model_name
    assert settings.provider.allowed_llm_nodes == ("classify_query", "synthesize_answer")
    assert settings.paths.repo_root.exists()
    assert settings.paths.env_file_path.name == ".env"
    assert settings.paths.corpus_manifest_path.exists()
    assert settings.paths.chunks_path.exists()
    assert settings.paths.metric_reference_path.exists()
    assert settings.paths.scenario_config_path.exists()
    assert settings.paths.eval_questions_path.exists()
    assert len(settings.frozen_corpus) == 3
    assert all(document.path.exists() for document in settings.frozen_corpus)


def test_load_settings_supports_local_dotenv(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=test-from-dotenv\nGEMINI_MODEL_NAME=gemini-custom\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)

    settings = config_module.load_settings()

    assert settings.provider.api_key == "test-from-dotenv"
    assert settings.provider.model_name == "gemini-custom"
    assert settings.provider.is_configured is True


def test_load_settings_without_dotenv_or_env_var_is_unconfigured(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)

    settings = config_module.load_settings()

    assert settings.provider.api_key is None
    assert settings.provider.model_name == "gemini-2.5-flash-lite"
    assert settings.provider.is_configured is False


def test_load_settings_ignores_malformed_dotenv_lines(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "NOT_A_PAIR\n"
        " export GEMINI_API_KEY = \"quoted-value\" \n"
        "BROKEN_LINE\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)

    settings = config_module.load_settings()

    assert settings.provider.api_key == "quoted-value"
    assert settings.provider.model_name == "gemini-2.5-flash-lite"
    assert settings.provider.is_configured is True


def test_artifact_scaffolding_is_valid() -> None:
    settings = load_settings()

    corpus_manifest = json.loads(settings.paths.corpus_manifest_path.read_text(encoding="utf-8"))
    data_manifest = json.loads(settings.paths.data_manifest_path.read_text(encoding="utf-8"))
    metric_reference = json.loads(settings.paths.metric_reference_path.read_text(encoding="utf-8"))
    scenario_config = yaml.safe_load(settings.paths.scenario_config_path.read_text(encoding="utf-8"))

    assert corpus_manifest["status"] in {"placeholder", "ready"}
    assert len(corpus_manifest["documents"]) == 3
    assert data_manifest["status"] in {"placeholder", "ready"}
    assert "artifacts" in data_manifest
    assert metric_reference["status"] in {"placeholder", "ready"}
    assert scenario_config["status"] in {"placeholder", "ready"}
    assert settings.paths.eval_questions_path.exists()
    assert settings.paths.chunks_path.exists()


def test_schema_smoke() -> None:
    retrieved_chunk = RetrievalChunk(
        chunk_id="chunk:model_validation_report:p3:1",
        text="Calibration worsened out of time.",
        doc_id="Model_Validation_Report",
        page=3,
        section_path="Validation Summary > Calibration",
        chunk_type="paragraph",
        topic_tags=["calibration"],
        authority_level="internal",
        document_role="internal_case_primary_source",
        sparse_score=1.2,
        dense_score=0.8,
        fused_score=1.0,
    )
    document_evidence = AnswerEvidenceItem(
        source_type=EvidenceSourceType.DOCUMENT,
        source_id=retrieved_chunk.chunk_id,
        doc_id=retrieved_chunk.doc_id,
        page=retrieved_chunk.page,
        support="The validation report shows weaker out-of-time calibration.",
    )
    diagnostics_output = DiagnosticsOutput(
        summary="Calibration degradation surfaced in the diagnostics artifact.",
        metrics=DiagnosticsMetrics(),
    )
    stress_output = StressOutput(
        summary="Stress deltas increase monotonically across scenarios.",
        metrics=StressMetrics(monotonicity_passed=True),
    )
    final_answer = FinalAnswer(
        executive_answer="Calibration weakened and should be monitored.",
        evidence=[document_evidence],
        numeric_summary={"tool_mode": "diagnostics"},
        limitations=["Placeholder artifact content is not yet populated."],
        confidence=ConfidenceLevel.MEDIUM,
        abstained=False,
        review_flag=False,
    )
    graph_state = GraphState(
        query="Summarize calibration performance.",
        run_id="run-001",
        started_at_utc="2026-03-24T12:00:00Z",
        started_at_monotonic_ns=1,
        query_type=QueryType.MIXED,
        classification_source="heuristic",
        retrieved_chunks=[retrieved_chunk],
        retrieved_doc_ids=[retrieved_chunk.doc_id],
        tool_required=True,
        tool_mode=ToolMode.DIAGNOSTICS,
        tool_output=diagnostics_output,
        draft_answer=final_answer,
        final_answer=final_answer,
        confidence=ConfidenceLevel.MEDIUM,
        abstained=False,
        review_flag=False,
    )
    trace = TraceRecord(
        run_id="run-001",
        timestamp_utc="2026-03-24T12:00:00Z",
        query=graph_state.query,
        query_type=QueryType.MIXED,
        retrieved_chunk_ids=[retrieved_chunk.chunk_id],
        retrieved_doc_ids=graph_state.retrieved_doc_ids,
        tool_used=True,
        tool_mode=ToolMode.STRESS,
        confidence=ConfidenceLevel.LOW,
        abstained=False,
        review_flag=True,
        latency_ms=12,
        executive_answer="Calibration weakened and should be monitored.",
    )
    classification = QueryClassification(
        query_type=QueryType.MIXED,
        tool_required=True,
        tool_mode=ToolMode.DIAGNOSTICS,
        reason_code="diagnostics_plus_policy",
    )
    eval_question = EvalQuestion(
        question_id="mixed-001",
        question="What does the report say and what do the diagnostics suggest?",
        question_type=QueryType.MIXED,
        case_slice=EvalCaseSlice.MIXED,
        challenge_level=EvalChallengeLevel.STANDARD,
        expected_sources=["Model_Validation_Report", "risk_diagnostics_tool:diagnostics"],
        expected_answer_mode=ExpectedAnswerMode.DOCUMENT_PLUS_COMPUTED,
        expected_tool_use=True,
        expected_tool_mode=ToolMode.DIAGNOSTICS,
        expected_abstain=False,
        expected_unsupported_reason=None,
        numeric_check=NumericCheck(expected_keys=["numeric_summary.tool_mode"]),
        reference_points=["Diagnostics evidence should be combined with document grounding."],
    )

    assert diagnostics_output.mode == "diagnostics"
    assert diagnostics_output.source == "risk_diagnostics_tool"
    assert stress_output.mode == "stress"
    assert stress_output.source == "risk_diagnostics_tool"
    assert graph_state.tool_output == diagnostics_output
    assert trace.tool_mode == ToolMode.STRESS
    assert classification.tool_mode == ToolMode.DIAGNOSTICS
    assert eval_question.question_type == QueryType.MIXED
