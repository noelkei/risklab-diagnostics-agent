from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from langgraph.graph import END, START, StateGraph

from app import ProjectSettings, load_settings
from retrieval import collect_retrieved_doc_ids, get_default_retriever
from schemas import (
    AnswerEvidenceItem,
    ConfidenceLevel,
    DiagnosticsOutput,
    EvidenceSourceType,
    FinalAnswer,
    GraphState,
    QueryClassification,
    QueryType,
    RetrievalChunk,
    StressOutput,
    ToolMode,
    TraceRecord,
)
from tools import computed_source_id_for_mode, risk_diagnostics_tool
from traces import JsonlTraceStore

from .provider import GeminiProvider


class RetrieverLike(Protocol):
    def search(self, query: str) -> list[RetrievalChunk]: ...


class ProviderLike(Protocol):
    def classify_query(self, query: str) -> QueryClassification: ...

    def synthesize_executive_answer(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence: list[AnswerEvidenceItem],
        numeric_summary: dict | None,
        limitations: list[str],
    ) -> str: ...


class TraceStoreLike(Protocol):
    def append(self, record: TraceRecord): ...


_WHITESPACE_PATTERN = re.compile(r"\s+")

_OUT_OF_SCOPE_PATTERNS = (
    "buy ",
    "sell ",
    "stock",
    "share price",
    "price target",
    "next week",
    "tomorrow",
    "crypto",
    "bitcoin",
    "ethereum",
    "nvda",
    "tesla",
    "aapl",
)
_POLICY_PATTERNS = (
    "governance",
    "policy",
    "policies",
    "control",
    "controls",
    "documentation",
    "expectation",
    "expectations",
    "requirement",
    "requirements",
    "define",
    "definition",
    "role",
    "responsibility",
    "oversight",
    "board",
    "guidance",
    "sr11",
    "sr 11",
    "sr11-7",
    "principles",
    "basel",
)
_DIAGNOSTICS_PATTERNS = (
    "diagnostic",
    "diagnostics",
    "psi",
    "drift",
    "calibration",
    "score shift",
    "score stability",
    "ks ",
    "auc",
    "brier",
    "challenger",
    "champion",
    "out-of-time",
    "out of time",
    "oot",
    "feature drift",
    "top drift",
)
_STRESS_GENERAL_PATTERNS = (
    "stress testing",
    "stress test",
    "stress governance",
    "scenario design",
    "scenario governance",
)
_STRESS_NUMERIC_PATTERNS = (
    "baseline",
    "mild",
    "severe",
    "mean pd",
    "p95",
    "tail pd",
    "el proxy",
    "delta",
    "monotonic",
    "monotonicity",
    "scenario",
    "scenarios",
)
_CLEAR_UNSUPPORTED_REASONS = {
    "blank_query",
    "out_of_scope",
    "multi_mode_request",
    "weak_document_support",
    "missing_required_tool_output",
}
_EXPECTED_DOC_IDS_BY_MODE = {
    ToolMode.DIAGNOSTICS: {
        "Model_Validation_Report",
        "SR11_7_Model_Risk_Management",
    },
    ToolMode.STRESS: {
        "Model_Validation_Report",
        "Basel_Stress_Testing_Principles_2018",
    },
}
_EXPECTED_TOPIC_TAGS_BY_MODE = {
    ToolMode.DIAGNOSTICS: {"drift", "calibration", "performance", "validation", "challenger"},
    ToolMode.STRESS: {"stress_testing", "sensitivity"},
}


def build_graph(
    *,
    settings: ProjectSettings | None = None,
    retriever: RetrieverLike | None = None,
    provider: ProviderLike | None = None,
    trace_store: TraceStoreLike | None = None,
    allow_fallback: bool = False,
):
    resolved_settings = settings or load_settings()
    resolved_retriever = retriever or get_default_retriever()
    resolved_provider = provider or GeminiProvider(settings=resolved_settings, allow_fallback=allow_fallback)
    resolved_trace_store = trace_store or JsonlTraceStore.from_settings(settings=resolved_settings)

    def intake(state: GraphState) -> dict[str, object]:
        normalized_query = _normalize_query(state.query)
        base_updates: dict[str, object] = {
            "query": normalized_query,
            "run_id": state.run_id or uuid.uuid4().hex,
            "started_at_utc": state.started_at_utc or _utc_now_iso(),
            "started_at_monotonic_ns": state.started_at_monotonic_ns or time.perf_counter_ns(),
        }
        if normalized_query:
            return base_updates
        return {
            **base_updates,
            "query_type": QueryType.UNSUPPORTED,
            "classification_source": "heuristic",
            "unsupported_reason": "blank_query",
            "tool_required": False,
            "tool_mode": None,
        }

    def classify_query(state: GraphState) -> dict[str, object]:
        if state.query_type == QueryType.UNSUPPORTED and state.unsupported_reason == "blank_query":
            return {}

        classification = _classify_query_heuristically(state.query)
        classification_source = "heuristic"
        if classification is None:
            classification = resolved_provider.classify_query(state.query)
            classification_source = "fallback" if not resolved_settings.provider.is_configured else "gemini"

        unsupported_reason = classification.reason_code if classification.query_type == QueryType.UNSUPPORTED else None
        return {
            "query_type": classification.query_type,
            "tool_required": classification.tool_required,
            "tool_mode": classification.tool_mode,
            "unsupported_reason": unsupported_reason,
            "classification_source": classification_source,
        }

    def retrieve_evidence(state: GraphState) -> dict[str, object]:
        if state.query_type == QueryType.UNSUPPORTED:
            return {"retrieved_chunks": [], "retrieved_doc_ids": []}
        retrieved_chunks = resolved_retriever.search(state.query)
        return {
            "retrieved_chunks": retrieved_chunks,
            "retrieved_doc_ids": collect_retrieved_doc_ids(retrieved_chunks),
        }

    def call_risk_tool_if_needed(state: GraphState) -> dict[str, object]:
        if not state.tool_required or state.tool_mode is None:
            return {"tool_output": None}
        tool_output = risk_diagnostics_tool(
            state.tool_mode,
            request={
                "query": state.query,
                "query_type": state.query_type.value if state.query_type is not None else None,
            },
        )
        return {"tool_output": tool_output}

    def synthesize_answer(state: GraphState) -> dict[str, object]:
        evidence = _build_evidence_items(state.retrieved_chunks, state.tool_output, state.tool_mode)
        numeric_summary = _build_numeric_summary(state.tool_output)
        pre_synthesis_reason = _get_pre_synthesis_reason(state)
        limitations = _build_limitations(
            query_type=state.query_type,
            tool_output=state.tool_output,
            unsupported_reason=pre_synthesis_reason,
        )
        if pre_synthesis_reason is not None:
            draft_answer = _build_abstention_answer(
                reason=pre_synthesis_reason,
                evidence=evidence,
                numeric_summary=numeric_summary,
                limitations=limitations,
            )
            return {
                "draft_answer": draft_answer,
                "unsupported_reason": pre_synthesis_reason,
                "confidence": draft_answer.confidence,
                "abstained": draft_answer.abstained,
                "review_flag": draft_answer.review_flag,
            }

        if state.query_type is None:
            raise RuntimeError("query_type must be set before synthesis.")
        executive_answer = resolved_provider.synthesize_executive_answer(
            query=state.query,
            query_type=state.query_type,
            evidence=evidence,
            numeric_summary=numeric_summary,
            limitations=limitations,
        )
        draft_answer = FinalAnswer(
            executive_answer=executive_answer,
            evidence=evidence,
            numeric_summary=numeric_summary,
            limitations=limitations,
            confidence=ConfidenceLevel.LOW,
            abstained=False,
            review_flag=False,
        )
        return {"draft_answer": draft_answer}

    def verify_or_abstain(state: GraphState) -> dict[str, object]:
        if state.draft_answer is None:
            raise RuntimeError("draft_answer must be set before verification.")

        final_reason = _get_final_abstention_reason(state)
        if final_reason is not None:
            final_answer = _build_abstention_answer(
                reason=final_reason,
                evidence=state.draft_answer.evidence,
                numeric_summary=state.draft_answer.numeric_summary,
                limitations=state.draft_answer.limitations,
            )
            return {
                "final_answer": final_answer,
                "confidence": final_answer.confidence,
                "abstained": final_answer.abstained,
                "review_flag": final_answer.review_flag,
                "unsupported_reason": final_reason,
            }

        confidence = _determine_confidence(state)
        review_flag = confidence is ConfidenceLevel.LOW or state.classification_source == "fallback"
        final_answer = state.draft_answer.model_copy(
            update={
                "confidence": confidence,
                "abstained": False,
                "review_flag": review_flag,
            }
        )
        return {
            "final_answer": final_answer,
            "confidence": confidence,
            "abstained": False,
            "review_flag": review_flag,
            "unsupported_reason": None,
        }

    def persist_trace(state: GraphState) -> dict[str, object]:
        if state.final_answer is None or state.query_type is None or state.confidence is None:
            raise RuntimeError("final graph state is incomplete before trace persistence.")

        latency_ms = _compute_latency_ms(state.started_at_monotonic_ns)
        trace_record = TraceRecord(
            run_id=state.run_id or uuid.uuid4().hex,
            timestamp_utc=state.started_at_utc or _utc_now_iso(),
            query=state.query,
            query_type=state.query_type,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in state.retrieved_chunks],
            retrieved_doc_ids=state.retrieved_doc_ids,
            tool_used=state.tool_output is not None,
            tool_mode=state.tool_mode,
            confidence=state.confidence,
            abstained=state.abstained,
            review_flag=state.review_flag,
            latency_ms=latency_ms,
            unsupported_reason=state.unsupported_reason,
            executive_answer=state.final_answer.executive_answer,
        )
        resolved_trace_store.append(trace_record)
        return {"latency_ms": latency_ms}

    builder = StateGraph(GraphState)
    builder.add_node("intake", intake)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("call_risk_tool_if_needed", call_risk_tool_if_needed)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("verify_or_abstain", verify_or_abstain)
    builder.add_node("persist_trace", persist_trace)
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "classify_query")
    builder.add_edge("classify_query", "retrieve_evidence")
    builder.add_edge("retrieve_evidence", "call_risk_tool_if_needed")
    builder.add_edge("call_risk_tool_if_needed", "synthesize_answer")
    builder.add_edge("synthesize_answer", "verify_or_abstain")
    builder.add_edge("verify_or_abstain", "persist_trace")
    builder.add_edge("persist_trace", END)
    return builder.compile()


def run_query(
    query: str,
    *,
    settings: ProjectSettings | None = None,
    retriever: RetrieverLike | None = None,
    provider: ProviderLike | None = None,
    trace_store: TraceStoreLike | None = None,
    allow_fallback: bool = False,
) -> GraphState:
    graph = build_graph(
        settings=settings,
        retriever=retriever,
        provider=provider,
        trace_store=trace_store,
        allow_fallback=allow_fallback,
    )
    result = graph.invoke({"query": query})
    return GraphState.model_validate(result)


def _classify_query_heuristically(query: str) -> QueryClassification | None:
    normalized_query = _normalize_query(query)
    if not normalized_query:
        return QueryClassification(
            query_type=QueryType.UNSUPPORTED,
            tool_required=False,
            reason_code="blank_query",
        )

    lowered = normalized_query.lower()
    if _contains_any(lowered, _OUT_OF_SCOPE_PATTERNS):
        return QueryClassification(
            query_type=QueryType.UNSUPPORTED,
            tool_required=False,
            reason_code="out_of_scope",
        )

    has_policy = _contains_any(lowered, _POLICY_PATTERNS)
    has_diagnostics = _contains_any(lowered, _DIAGNOSTICS_PATTERNS)
    has_stress_general = _contains_any(lowered, _STRESS_GENERAL_PATTERNS)
    has_stress_numeric = _contains_any(lowered, _STRESS_NUMERIC_PATTERNS)

    if has_diagnostics and has_stress_numeric:
        return QueryClassification(
            query_type=QueryType.UNSUPPORTED,
            tool_required=False,
            reason_code="multi_mode_request",
        )
    if has_diagnostics and has_policy:
        return QueryClassification(
            query_type=QueryType.MIXED,
            tool_required=True,
            tool_mode=ToolMode.DIAGNOSTICS,
            reason_code="diagnostics_plus_policy",
        )
    if has_stress_numeric and has_policy:
        return QueryClassification(
            query_type=QueryType.MIXED,
            tool_required=True,
            tool_mode=ToolMode.STRESS,
            reason_code="stress_plus_policy",
        )
    if has_diagnostics:
        return QueryClassification(
            query_type=QueryType.NUMERIC,
            tool_required=True,
            tool_mode=ToolMode.DIAGNOSTICS,
            reason_code="diagnostics_numeric",
        )
    if has_stress_numeric:
        return QueryClassification(
            query_type=QueryType.NUMERIC,
            tool_required=True,
            tool_mode=ToolMode.STRESS,
            reason_code="stress_numeric",
        )
    if has_policy or has_stress_general:
        return QueryClassification(
            query_type=QueryType.POLICY,
            tool_required=False,
            reason_code="policy_document",
        )
    return None


def _build_evidence_items(
    retrieved_chunks: list[RetrievalChunk],
    tool_output: DiagnosticsOutput | StressOutput | None,
    tool_mode: ToolMode | None,
) -> list[AnswerEvidenceItem]:
    evidence_items = [
        AnswerEvidenceItem(
            source_type=EvidenceSourceType.DOCUMENT,
            source_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            page=chunk.page,
            support=_excerpt_text(chunk.text),
        )
        for chunk in retrieved_chunks
    ]
    if tool_output is not None and tool_mode is not None:
        evidence_items.append(
            AnswerEvidenceItem(
                source_type=EvidenceSourceType.COMPUTED,
                source_id=computed_source_id_for_mode(tool_mode),
                support=tool_output.summary,
            )
        )
    return evidence_items


def _build_numeric_summary(tool_output: DiagnosticsOutput | StressOutput | None) -> dict | None:
    if tool_output is None:
        return None
    if isinstance(tool_output, DiagnosticsOutput):
        calibration = tool_output.metrics.calibration
        return {
            "tool_mode": ToolMode.DIAGNOSTICS.value,
            "summary": tool_output.summary,
            "top_drift_features": tool_output.metrics.top_drift_features[:3],
            "champion_oot": calibration.get("champion_oot", {}),
            "challenger_oot": calibration.get("challenger_oot", {}),
            "quality_flags": tool_output.quality_flags,
        }
    return {
        "tool_mode": ToolMode.STRESS.value,
        "summary": tool_output.summary,
        "baseline": tool_output.metrics.baseline,
        "mild": tool_output.metrics.mild,
        "severe": tool_output.metrics.severe,
        "delta_mean_pd": tool_output.metrics.delta_mean_pd.model_dump(mode="json"),
        "delta_tail_pd": tool_output.metrics.delta_tail_pd.model_dump(mode="json"),
        "delta_el_proxy": tool_output.metrics.delta_el_proxy.model_dump(mode="json"),
        "monotonicity_passed": tool_output.metrics.monotonicity_passed,
        "quality_flags": tool_output.quality_flags,
    }


def _build_limitations(
    *,
    query_type: QueryType | None,
    tool_output: DiagnosticsOutput | StressOutput | None,
    unsupported_reason: str | None = None,
) -> list[str]:
    limitations = ["This workflow is limited to the frozen three-document corpus and prepared structured artifacts."]
    if query_type in {QueryType.NUMERIC, QueryType.MIXED} and tool_output is not None:
        limitations.extend(tool_output.limitations)
    if unsupported_reason == "weak_document_support":
        limitations.append("Available document evidence was insufficient for a grounded answer.")
    if unsupported_reason == "missing_required_tool_output":
        limitations.append("Required quantitative evidence was unavailable for this route.")
    if unsupported_reason == "multi_mode_request":
        limitations.append("The MVP can use only one quantitative tool mode per query.")
    if unsupported_reason == "out_of_scope":
        limitations.append("The request falls outside the frozen corpus and quantitative artifact boundary.")
    return _dedupe_preserve_order(limitations)


def _get_pre_synthesis_reason(state: GraphState) -> str | None:
    if state.query_type == QueryType.UNSUPPORTED:
        return state.unsupported_reason or "out_of_scope"
    if state.unsupported_reason in _CLEAR_UNSUPPORTED_REASONS:
        return state.unsupported_reason
    if state.query_type == QueryType.POLICY and not state.retrieved_chunks:
        return "weak_document_support"
    if state.query_type == QueryType.NUMERIC and state.tool_required and state.tool_output is None:
        return "missing_required_tool_output"
    if state.query_type == QueryType.MIXED and not state.retrieved_chunks:
        return "weak_document_support"
    if state.query_type == QueryType.MIXED and state.tool_required and state.tool_output is None:
        return "missing_required_tool_output"
    return None


def _get_final_abstention_reason(state: GraphState) -> str | None:
    pre_reason = _get_pre_synthesis_reason(state)
    if pre_reason is not None:
        return pre_reason
    if _has_document_tool_mismatch(state):
        return "document_tool_mismatch"
    return None


def _has_document_tool_mismatch(state: GraphState) -> bool:
    if state.tool_mode is None or not state.retrieved_doc_ids:
        return False
    expected_doc_ids = _EXPECTED_DOC_IDS_BY_MODE[state.tool_mode]
    if not expected_doc_ids.intersection(state.retrieved_doc_ids):
        return True
    expected_topic_tags = _EXPECTED_TOPIC_TAGS_BY_MODE[state.tool_mode]
    return not any(expected_topic_tags.intersection(set(chunk.topic_tags)) for chunk in state.retrieved_chunks)


def _determine_confidence(state: GraphState) -> ConfidenceLevel:
    if state.query_type == QueryType.POLICY:
        if len(state.retrieved_chunks) >= 2 and state.classification_source == "heuristic":
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    if state.query_type == QueryType.NUMERIC:
        if state.tool_output is None:
            return ConfidenceLevel.LOW
        if not state.tool_output.quality_flags and state.retrieved_chunks:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    if state.query_type == QueryType.MIXED:
        if state.tool_output is None or not state.retrieved_chunks:
            return ConfidenceLevel.LOW
        if not state.tool_output.quality_flags and len(state.retrieved_chunks) >= 2:
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def _build_abstention_answer(
    *,
    reason: str,
    evidence: list[AnswerEvidenceItem],
    numeric_summary: dict | None,
    limitations: list[str],
) -> FinalAnswer:
    return FinalAnswer(
        executive_answer=_unsupported_message_for_reason(reason),
        evidence=evidence,
        numeric_summary=numeric_summary,
        limitations=_dedupe_preserve_order(limitations),
        confidence=ConfidenceLevel.LOW,
        abstained=True,
        review_flag=False,
    )


def _unsupported_message_for_reason(reason: str) -> str:
    if reason == "blank_query":
        return "I can't answer an empty query."
    if reason == "out_of_scope":
        return "I can't support that request with the frozen corpus and quantitative artifacts."
    if reason == "multi_mode_request":
        return "I can't support a single answer because this MVP handles only one quantitative mode per query."
    if reason == "missing_required_tool_output":
        return "I can't support that answer because the required quantitative evidence is unavailable."
    if reason == "document_tool_mismatch":
        return "I can't support a grounded answer because the retrieved document evidence does not align with the requested route."
    return "I can't support a grounded answer because the available evidence is too weak."


def _excerpt_text(text: str, *, limit: int = 180) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _normalize_query(query: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", query).strip()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _compute_latency_ms(started_at_monotonic_ns: int | None) -> int | None:
    if started_at_monotonic_ns is None:
        return None
    return max(0, int((time.perf_counter_ns() - started_at_monotonic_ns) / 1_000_000))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
