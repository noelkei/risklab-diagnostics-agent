from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_ALLOWED_DIRECTION_HINTS = {
    "positive",
    "negative",
    "true",
    "false",
    "non_decreasing",
}


def _validate_string_list(values: list[str], *, field_name: str) -> list[str]:
    cleaned_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} entries must be non-empty strings.")
        if stripped in seen:
            raise ValueError(f"{field_name} entries must be unique.")
        seen.add(stripped)
        cleaned_values.append(stripped)
    return cleaned_values


class QueryType(str, Enum):
    POLICY = "policy"
    NUMERIC = "numeric"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


class ToolMode(str, Enum):
    DIAGNOSTICS = "diagnostics"
    STRESS = "stress"


class EvidenceSourceType(str, Enum):
    DOCUMENT = "document"
    COMPUTED = "computed"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvalRunMode(str, Enum):
    CANONICAL = "canonical"
    HERMETIC_VERIFY = "hermetic_verify"


class EvalRunStatus(str, Enum):
    COMPLETED = "completed"
    ENVIRONMENT_BLOCKED = "environment_blocked"
    RUNNER_FAILED = "runner_failed"


class EvalCaseSlice(str, Enum):
    POLICY = "policy"
    DIAGNOSTICS = "diagnostics"
    STRESS = "stress"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"


class EvalChallengeLevel(str, Enum):
    STANDARD = "standard"
    BORDERLINE_ABSTAIN = "borderline_abstain"
    ADVERSARIAL_LIGHT = "adversarial_light"
    MULTI_MODE_CONFLICT = "multi_mode_conflict"


class ExpectedAnswerMode(str, Enum):
    DOCUMENT_ONLY = "document_only"
    COMPUTED_ONLY = "computed_only"
    DOCUMENT_PLUS_COMPUTED = "document_plus_computed"
    ABSTAIN = "abstain"


class EvalMetricStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class QueryClassification(StrictModel):
    query_type: QueryType
    tool_required: bool
    tool_mode: ToolMode | None = None
    reason_code: str | None = None


class RetrievalChunk(StrictModel):
    chunk_id: str
    text: str
    doc_id: str
    page: int
    section_path: str
    chunk_type: str
    topic_tags: list[str] = Field(default_factory=list)
    authority_level: str
    document_role: str
    sparse_score: float | None = None
    dense_score: float | None = None
    fused_score: float | None = None


class DiagnosticsMetrics(StrictModel):
    top_drift_features: list[dict[str, Any]] = Field(default_factory=list)
    score_shift: dict[str, Any] = Field(default_factory=dict)
    calibration: dict[str, Any] = Field(default_factory=dict)
    reference_metrics: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsOutput(StrictModel):
    source: Literal["risk_diagnostics_tool"] = "risk_diagnostics_tool"
    mode: Literal["diagnostics"] = "diagnostics"
    summary: str
    metrics: DiagnosticsMetrics
    quality_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ScenarioDeltaMap(StrictModel):
    mild: float | None = None
    severe: float | None = None


class StressMetrics(StrictModel):
    baseline: dict[str, Any] = Field(default_factory=dict)
    mild: dict[str, Any] = Field(default_factory=dict)
    severe: dict[str, Any] = Field(default_factory=dict)
    delta_mean_pd: ScenarioDeltaMap = Field(default_factory=ScenarioDeltaMap)
    delta_tail_pd: ScenarioDeltaMap = Field(default_factory=ScenarioDeltaMap)
    delta_el_proxy: ScenarioDeltaMap = Field(default_factory=ScenarioDeltaMap)
    monotonicity_passed: bool


class StressOutput(StrictModel):
    source: Literal["risk_diagnostics_tool"] = "risk_diagnostics_tool"
    mode: Literal["stress"] = "stress"
    summary: str
    metrics: StressMetrics
    quality_flags: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AnswerEvidenceItem(StrictModel):
    source_type: EvidenceSourceType
    source_id: str
    doc_id: str | None = None
    page: int | None = None
    support: str


class FinalAnswer(StrictModel):
    executive_answer: str
    evidence: list[AnswerEvidenceItem] = Field(default_factory=list)
    numeric_summary: dict[str, Any] | None = None
    limitations: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    abstained: bool
    review_flag: bool


class GraphState(StrictModel):
    query: str
    run_id: str | None = None
    started_at_utc: str | None = None
    started_at_monotonic_ns: int | None = None
    query_type: QueryType | None = None
    classification_source: Literal["heuristic", "gemini", "fallback"] | None = None
    retrieved_chunks: list[RetrievalChunk] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    tool_required: bool = False
    tool_mode: ToolMode | None = None
    tool_output: DiagnosticsOutput | StressOutput | None = None
    draft_answer: FinalAnswer | None = None
    final_answer: FinalAnswer | None = None
    confidence: ConfidenceLevel | None = None
    abstained: bool = False
    review_flag: bool = False
    unsupported_reason: str | None = None
    latency_ms: int | None = None


class TraceRecord(StrictModel):
    run_id: str
    timestamp_utc: str
    query: str
    query_type: QueryType
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    tool_used: bool = False
    tool_mode: ToolMode | None = None
    confidence: ConfidenceLevel
    abstained: bool
    review_flag: bool
    latency_ms: int | None = None
    unsupported_reason: str | None = None
    executive_answer: str


class NumericCheck(StrictModel):
    expected_keys: list[str] = Field(default_factory=list)
    expected_values: dict[str, Any] = Field(default_factory=dict)
    expected_direction: dict[str, str] = Field(default_factory=dict)

    @field_validator("expected_keys")
    @classmethod
    def _validate_expected_keys(cls, values: list[str]) -> list[str]:
        return _validate_string_list(values, field_name="expected_keys")

    @field_validator("expected_values")
    @classmethod
    def _validate_expected_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        for dotted_path in values:
            if not dotted_path.strip():
                raise ValueError("expected_values keys must be non-empty dotted paths.")
        return values

    @field_validator("expected_direction")
    @classmethod
    def _validate_expected_direction(cls, values: dict[str, str]) -> dict[str, str]:
        for dotted_path, direction in values.items():
            if not dotted_path.strip():
                raise ValueError("expected_direction keys must be non-empty dotted paths.")
            if direction not in _ALLOWED_DIRECTION_HINTS:
                raise ValueError(
                    f"expected_direction values must be one of: {', '.join(sorted(_ALLOWED_DIRECTION_HINTS))}."
                )
        return values


class EvalQuestion(StrictModel):
    question_id: str
    question: str
    question_type: QueryType
    case_slice: EvalCaseSlice
    challenge_level: EvalChallengeLevel = EvalChallengeLevel.STANDARD
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer_mode: ExpectedAnswerMode
    expected_tool_use: bool
    expected_tool_mode: ToolMode | None = None
    expected_abstain: bool
    expected_unsupported_reason: str | None = None
    numeric_check: NumericCheck | None = None
    reference_points: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("question_id", "question")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Eval question text fields must be non-empty strings.")
        return stripped

    @field_validator("expected_sources", "reference_points")
    @classmethod
    def _validate_string_lists(cls, values: list[str], info: ValidationInfo) -> list[str]:
        return _validate_string_list(values, field_name=info.field_name or "list")

    @model_validator(mode="after")
    def _validate_eval_contract(self) -> "EvalQuestion":
        if self.expected_tool_use and self.expected_tool_mode is None:
            raise ValueError("expected_tool_mode is required when expected_tool_use is true.")
        if not self.expected_tool_use and self.expected_tool_mode is not None:
            raise ValueError("expected_tool_mode must be null when expected_tool_use is false.")
        if self.numeric_check is not None and not self.expected_tool_use:
            raise ValueError("numeric_check requires expected_tool_use to be true.")

        if self.question_type in {QueryType.NUMERIC, QueryType.MIXED} and not self.expected_tool_use:
            raise ValueError("numeric and mixed eval questions must require tool use.")
        if self.question_type in {QueryType.POLICY, QueryType.UNSUPPORTED} and self.expected_tool_use:
            raise ValueError("policy and unsupported eval questions must not require tool use.")

        if self.expected_answer_mode is ExpectedAnswerMode.ABSTAIN and not self.expected_abstain:
            raise ValueError("expected_answer_mode=abstain requires expected_abstain=true.")
        if self.expected_abstain and self.expected_answer_mode is not ExpectedAnswerMode.ABSTAIN:
            raise ValueError("expected_abstain=true requires expected_answer_mode=abstain.")
        if self.expected_unsupported_reason is not None and not self.expected_abstain:
            raise ValueError("expected_unsupported_reason requires expected_abstain=true.")

        if self.expected_answer_mode is ExpectedAnswerMode.DOCUMENT_ONLY and self.expected_tool_use:
            raise ValueError("document_only answers must not require tool use.")
        if (
            self.expected_answer_mode in {ExpectedAnswerMode.COMPUTED_ONLY, ExpectedAnswerMode.DOCUMENT_PLUS_COMPUTED}
            and not self.expected_tool_use
        ):
            raise ValueError("computed answer modes require expected_tool_use=true.")

        if self.case_slice is EvalCaseSlice.POLICY:
            if self.question_type is not QueryType.POLICY:
                raise ValueError("policy slice questions must declare question_type=policy.")
            if self.expected_tool_use:
                raise ValueError("policy slice questions must not require tool use.")
        if self.case_slice is EvalCaseSlice.DIAGNOSTICS:
            if self.question_type not in {QueryType.NUMERIC, QueryType.MIXED}:
                raise ValueError("diagnostics slice questions must be numeric or mixed.")
            if self.expected_tool_mode is not ToolMode.DIAGNOSTICS:
                raise ValueError("diagnostics slice questions must declare expected_tool_mode=diagnostics.")
        if self.case_slice is EvalCaseSlice.STRESS:
            if self.question_type not in {QueryType.NUMERIC, QueryType.MIXED}:
                raise ValueError("stress slice questions must be numeric or mixed.")
            if self.expected_tool_mode is not ToolMode.STRESS:
                raise ValueError("stress slice questions must declare expected_tool_mode=stress.")
        if self.case_slice is EvalCaseSlice.MIXED and self.question_type is not QueryType.MIXED:
            raise ValueError("mixed slice questions must declare question_type=mixed.")
        if self.case_slice is EvalCaseSlice.UNSUPPORTED:
            if self.question_type is not QueryType.UNSUPPORTED:
                raise ValueError("unsupported slice questions must declare question_type=unsupported.")
            if not self.expected_abstain:
                raise ValueError("unsupported slice questions must expect abstention.")
            if self.expected_sources:
                raise ValueError("unsupported slice questions must not declare expected_sources.")
            if self.numeric_check is not None:
                raise ValueError("unsupported slice questions must not declare numeric_check.")

        return self


class EvalMetricResult(StrictModel):
    status: EvalMetricStatus
    reason: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
