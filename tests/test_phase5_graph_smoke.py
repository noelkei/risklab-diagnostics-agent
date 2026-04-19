from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app import load_settings
from graph import GeminiProvider, run_query
from graph.provider import ProviderError
from schemas import QueryClassification, QueryType, RetrievalChunk, ToolMode
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
        authority_level="internal",
        document_role="internal_case_primary_source",
    )


SR11_POLICY_CHUNKS = [
    _make_chunk(
        chunk_id="chunk:sr11:p4:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=4,
        section_path="Governance, Policies, and Controls",
        text="SR 11-7 expects strong governance, policies, and effective challenge around model use and validation.",
        topic_tags=["governance", "validation"],
    ),
    _make_chunk(
        chunk_id="chunk:sr11:p3:1",
        doc_id="SR11_7_Model_Risk_Management",
        page=3,
        section_path="Model Validation",
        text="Model validation should include ongoing monitoring, outcomes analysis, and independent review.",
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

STRESS_CHUNKS = [
    _make_chunk(
        chunk_id="chunk:model:p10:1",
        doc_id="Model_Validation_Report",
        page=10,
        section_path="Stress Testing Results",
        text="Mild and severe scenarios increase PD and EL proxy relative to baseline in the internal validation report.",
        topic_tags=["stress_testing", "sensitivity", "validation"],
    ),
    _make_chunk(
        chunk_id="chunk:basel:p10:1",
        doc_id="Basel_Stress_Testing_Principles_2018",
        page=10,
        section_path="Stress Testing Principles",
        text="Basel principles expect stress testing to be integrated into governance, scenario design, and risk management.",
        topic_tags=["stress_testing", "governance"],
    ),
]

MISMATCH_CHUNKS = [
    _make_chunk(
        chunk_id="chunk:basel:p5:1",
        doc_id="Basel_Stress_Testing_Principles_2018",
        page=5,
        section_path="Introduction",
        text="Basel stress testing principles discuss scenario governance and bank-wide integration.",
        topic_tags=["stress_testing", "governance"],
    )
]


class StubRetriever:
    def __init__(self, mapping: dict[str, list[RetrievalChunk]]) -> None:
        self._mapping = mapping

    def search(self, query: str) -> list[RetrievalChunk]:
        lowered = query.lower()
        for key, chunks in self._mapping.items():
            if key in lowered:
                return [chunk.model_copy(deep=True) for chunk in chunks]
        return []


class StubProvider:
    def __init__(self, *, classification: QueryClassification | None = None) -> None:
        self._classification = classification or QueryClassification(
            query_type=QueryType.POLICY,
            tool_required=False,
            reason_code="provider_policy",
        )
        self.classification_calls: list[str] = []
        self.synthesis_calls: list[dict[str, object]] = []

    def classify_query(self, query: str) -> QueryClassification:
        self.classification_calls.append(query)
        return self._classification

    def synthesize_executive_answer(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence,
        numeric_summary,
        limitations,
    ) -> str:
        self.synthesis_calls.append(
            {
                "query": query,
                "query_type": query_type,
                "numeric_summary": numeric_summary,
                "limitations": limitations,
            }
        )
        return f"Stub synthesis for {query_type.value}."


def _trace_store(tmp_path: Path) -> JsonlTraceStore:
    return JsonlTraceStore(path=tmp_path / "outputs" / "traces" / "runs.jsonl")


def _load_trace_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase5_policy_query_returns_grounded_answer_and_trace(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"governance": SR11_POLICY_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "What governance expectations does SR11-7 set for model validation?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.POLICY
    assert result.final_answer is not None
    assert result.final_answer.abstained is False
    assert result.final_answer.review_flag is False
    assert result.final_answer.confidence.value == "high"
    assert all(item.source_type.value == "document" for item in result.final_answer.evidence)
    assert provider.classification_calls == []
    assert len(provider.synthesis_calls) == 1

    rows = _load_trace_rows(trace_store.path)
    assert len(rows) == 1
    assert rows[0]["query_type"] == "policy"
    assert rows[0]["abstained"] is False
    assert rows[0]["executive_answer"] == "Stub synthesis for policy."


def test_phase5_blank_query_abstains_and_skips_model_call(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "   ",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.UNSUPPORTED
    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "blank_query"
    assert provider.classification_calls == []
    assert provider.synthesis_calls == []


def test_phase5_numeric_diagnostics_query_uses_tool_and_computed_evidence(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"drift": DIAGNOSTICS_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "What are the top drift features and calibration issues?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.NUMERIC
    assert result.tool_mode == ToolMode.DIAGNOSTICS
    assert result.tool_output is not None
    assert result.final_answer is not None
    assert result.final_answer.abstained is False
    assert result.final_answer.numeric_summary is not None
    assert result.final_answer.numeric_summary["tool_mode"] == "diagnostics"
    assert any(item.source_type.value == "computed" for item in result.final_answer.evidence)
    assert len(provider.synthesis_calls) == 1


def test_phase5_mixed_stress_query_uses_tool_and_document_evidence(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"basel": STRESS_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "Based on Basel principles and the stress scenarios, how do mild and severe cases change PD?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.MIXED
    assert result.tool_mode == ToolMode.STRESS
    assert result.tool_output is not None
    assert result.final_answer is not None
    assert result.final_answer.abstained is False
    assert result.final_answer.numeric_summary is not None
    assert result.final_answer.numeric_summary["tool_mode"] == "stress"
    assert any(item.source_type.value == "computed" for item in result.final_answer.evidence)
    assert any(item.source_type.value == "document" for item in result.final_answer.evidence)


def test_phase5_multi_mode_request_abstains_and_skips_model_call(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "Explain the PSI drift findings and how mild and severe stress scenarios change EL.",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.UNSUPPORTED
    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "multi_mode_request"
    assert provider.classification_calls == []
    assert provider.synthesis_calls == []


def test_phase5_unsupported_query_abstains_without_model_call(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"nvda": MISMATCH_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "Should we buy NVDA stock next week?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.UNSUPPORTED
    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "out_of_scope"
    assert provider.classification_calls == []
    assert provider.synthesis_calls == []


def test_phase5_weak_document_support_skips_model_and_abstains(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "What governance expectations does SR11-7 set for model validation?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "weak_document_support"
    assert provider.synthesis_calls == []


def test_phase5_missing_tool_output_skips_model_and_abstains(tmp_path: Path, monkeypatch) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"psi": SR11_POLICY_CHUNKS})
    trace_store = _trace_store(tmp_path)

    monkeypatch.setattr("graph.workflow.risk_diagnostics_tool", lambda *_args, **_kwargs: None)

    result = run_query(
        "What do the governance findings and PSI diagnostics say?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.query_type == QueryType.MIXED
    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "missing_required_tool_output"
    assert provider.synthesis_calls == []


def test_phase5_document_tool_mismatch_abstains_after_synthesis(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"drift": MISMATCH_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "What are the top drift features and calibration issues?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert result.final_answer is not None
    assert result.final_answer.abstained is True
    assert result.unsupported_reason == "document_tool_mismatch"
    assert len(provider.synthesis_calls) == 1


def test_phase5_classifier_fallback_sets_review_flag(tmp_path: Path) -> None:
    base_settings = load_settings()
    fallback_settings = replace(
        base_settings,
        provider=replace(base_settings.provider, api_key=None),
    )
    provider = StubProvider(
        classification=QueryClassification(
            query_type=QueryType.POLICY,
            tool_required=False,
            reason_code="provider_policy",
        )
    )
    retriever = StubRetriever({"summarize": SR11_POLICY_CHUNKS})
    trace_store = _trace_store(tmp_path)

    result = run_query(
        "Summarize the relevant finding for this case.",
        settings=fallback_settings,
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    assert provider.classification_calls == ["Summarize the relevant finding for this case."]
    assert result.final_answer is not None
    assert result.final_answer.abstained is False
    assert result.final_answer.review_flag is True
    assert result.final_answer.confidence.value == "medium"


def test_phase5_provider_classification_failure_can_fallback_to_unsupported() -> None:
    base_settings = load_settings()
    provider = GeminiProvider(settings=base_settings, allow_fallback=True)
    provider._generate_structured = lambda **_kwargs: (_ for _ in ()).throw(ProviderError("boom"))  # type: ignore[method-assign]

    result = provider.classify_query("Summarize this ambiguous case.")

    assert result.query_type == QueryType.UNSUPPORTED
    assert result.reason_code == "out_of_scope"


def test_phase5_trace_store_is_append_only(tmp_path: Path) -> None:
    provider = StubProvider()
    retriever = StubRetriever({"governance": SR11_POLICY_CHUNKS})
    trace_store = _trace_store(tmp_path)

    run_query(
        "What governance expectations does SR11-7 set for model validation?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )
    run_query(
        "What governance expectations does SR11-7 set for model validation?",
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
    )

    rows = _load_trace_rows(trace_store.path)
    assert len(rows) == 2
    assert rows[0]["run_id"] != rows[1]["run_id"]
    assert all(row["query_type"] == "policy" for row in rows)


def test_phase5_trace_write_failure_propagates(tmp_path: Path) -> None:
    class FailingTraceStore:
        def append(self, record) -> None:
            raise OSError("disk full")

    provider = StubProvider()
    retriever = StubRetriever({"governance": SR11_POLICY_CHUNKS})

    with pytest.raises(OSError, match="disk full"):
        run_query(
            "What governance expectations does SR11-7 set for model validation?",
            retriever=retriever,
            provider=provider,
            trace_store=FailingTraceStore(),
        )


def test_phase5_gemini_provider_uses_configured_model_name() -> None:
    base_settings = load_settings()
    custom_settings = replace(
        base_settings,
        provider=replace(base_settings.provider, model_name="gemini-custom-model"),
    )

    provider = GeminiProvider(settings=custom_settings, allow_fallback=True)

    assert provider.model_name == "gemini-custom-model"
