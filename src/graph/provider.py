from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app import ProjectSettings, load_settings
from schemas import AnswerEvidenceItem, QueryClassification, QueryType, ToolMode


class ProviderError(RuntimeError):
    """Raised when the narrow Gemini provider boundary cannot complete safely."""


class ProviderConfigurationError(ProviderError):
    """Raised when Gemini is required but not configured."""


class _ExecutiveAnswerResponse(BaseModel):
    executive_answer: str


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class GeminiProvider:
    def __init__(
        self,
        *,
        settings: ProjectSettings | None = None,
        allow_fallback: bool = False,
    ) -> None:
        self._settings = settings or load_settings()
        self._allow_fallback = allow_fallback
        self._client = None

    @property
    def model_name(self) -> str:
        return self._settings.provider.model_name

    def classify_query(self, query: str) -> QueryClassification:
        if not self._settings.provider.is_configured:
            if self._allow_fallback:
                return QueryClassification(
                    query_type=QueryType.UNSUPPORTED,
                    tool_required=False,
                    reason_code="out_of_scope",
                )
            raise ProviderConfigurationError(
                f"Gemini requires {self._settings.provider.api_key_env_var} to be configured."
            )

        prompt = (
            "Classify the query for a small model-risk diagnostics workflow.\n"
            "Return JSON with fields: query_type, tool_required, tool_mode, reason_code.\n"
            "Allowed query_type values: policy, numeric, mixed, unsupported.\n"
            "Allowed tool_mode values: diagnostics, stress, or null.\n"
            "Rules:\n"
            "- policy: document/governance questions with no numeric tool need.\n"
            "- numeric: requires quantitative metrics from the deterministic tool.\n"
            "- mixed: needs both document grounding and quantitative evidence.\n"
            "- unsupported: out of scope, investment advice, future prediction, or a request that needs both tool modes.\n"
            "- tool_required must be true only for numeric or mixed.\n"
            "- tool_mode must be diagnostics or stress only when tool_required is true.\n"
            "- reason_code should be a short snake_case label.\n"
            f"Query: {query}"
        )
        try:
            result = self._generate_structured(
                response_model=QueryClassification,
                prompt=prompt,
                max_output_tokens=128,
            )
        except ProviderError:
            if not self._allow_fallback:
                raise
            return QueryClassification(
                query_type=QueryType.UNSUPPORTED,
                tool_required=False,
                reason_code="out_of_scope",
            )
        return self._normalize_classification(result)

    def synthesize_executive_answer(
        self,
        *,
        query: str,
        query_type: QueryType,
        evidence: list[AnswerEvidenceItem],
        numeric_summary: dict | None,
        limitations: list[str],
    ) -> str:
        if not self._settings.provider.is_configured:
            if self._allow_fallback:
                return self._fallback_executive_answer(
                    query_type=query_type,
                    evidence=evidence,
                    numeric_summary=numeric_summary,
                )
            raise ProviderConfigurationError(
                f"Gemini requires {self._settings.provider.api_key_env_var} to be configured."
            )

        evidence_lines = []
        for item in evidence:
            evidence_lines.append(
                f"- {item.source_type.value} | {item.source_id} | {item.doc_id} | {item.page} | {item.support}"
            )
        prompt = (
            "Write one short executive answer for a bounded model-risk diagnostics workflow.\n"
            "Use only the supplied evidence and numeric summary.\n"
            "Do not invent numbers, sources, caveats, or unsupported claims.\n"
            "Keep it to one or two sentences.\n"
            f"Query type: {query_type.value}\n"
            f"Query: {query}\n"
            "Evidence:\n"
            + "\n".join(evidence_lines)
            + "\n"
            + f"Numeric summary: {numeric_summary}\n"
            + f"Limitations: {limitations}\n"
        )
        try:
            result = self._generate_structured(
                response_model=_ExecutiveAnswerResponse,
                prompt=prompt,
                max_output_tokens=160,
            )
        except ProviderError:
            if not self._allow_fallback:
                raise
            return self._fallback_executive_answer(
                query_type=query_type,
                evidence=evidence,
                numeric_summary=numeric_summary,
            )
        executive_answer = result.executive_answer.strip()
        if not executive_answer:
            raise ProviderError("Gemini synthesis returned an empty executive answer.")
        return executive_answer

    def _generate_structured(
        self,
        *,
        response_model: type[ResponseModelT],
        prompt: str,
        max_output_tokens: int,
    ) -> ResponseModelT:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=self._settings.provider.api_key)

        try:
            response = self._client.models.generate_content(
                model=self._settings.provider.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    candidateCount=1,
                    maxOutputTokens=max_output_tokens,
                    responseMimeType="application/json",
                    responseSchema=response_model,
                    thinkingConfig=types.ThinkingConfig(thinkingBudget=0),
                ),
            )
        except Exception as exc:  # pragma: no cover - network/provider failure
            raise ProviderError("Gemini generate_content call failed.") from exc

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, response_model):
                return parsed
            return response_model.model_validate(parsed)

        response_text = getattr(response, "text", "")
        if not response_text:
            raise ProviderError("Gemini response did not contain JSON text.")
        return response_model.model_validate_json(response_text)

    @staticmethod
    def _normalize_classification(result: QueryClassification) -> QueryClassification:
        if result.query_type in {QueryType.POLICY, QueryType.UNSUPPORTED}:
            return result.model_copy(update={"tool_required": False, "tool_mode": None})
        if result.tool_mode is None:
            raise ProviderError("Gemini classification omitted tool_mode for a supported tool route.")
        return result.model_copy(update={"tool_required": True})

    @staticmethod
    def _fallback_executive_answer(
        *,
        query_type: QueryType,
        evidence: list[AnswerEvidenceItem],
        numeric_summary: dict | None,
    ) -> str:
        if numeric_summary and isinstance(numeric_summary.get("summary"), str):
            return str(numeric_summary["summary"]).strip()
        if evidence:
            return evidence[0].support
        return f"The available {query_type.value} evidence supports only a limited deterministic summary."
