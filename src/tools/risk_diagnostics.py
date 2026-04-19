from __future__ import annotations

from typing import Any

from app import ProjectSettings
from schemas import (
    DiagnosticsMetrics,
    DiagnosticsOutput,
    ScenarioDeltaMap,
    StressMetrics,
    StressOutput,
    ToolMode,
)

from .loader import (
    DiagnosticsArtifactBundle,
    StressArtifactBundle,
    StructuredArtifactLoadError,
    load_diagnostics_artifacts,
    load_stress_artifacts,
)


_DIAGNOSTICS_SOURCE_ID = "risk_diagnostics_tool:diagnostics"
_STRESS_SOURCE_ID = "risk_diagnostics_tool:stress"
_SCENARIO_ORDER = ("baseline", "mild", "severe")
_SCENARIO_METRIC_ORDER = (
    "mean_pd",
    "p95_pd",
    "el_proxy_mean",
    "delta_mean_pd",
    "delta_p95_pd",
    "delta_el_proxy",
    "lgd_avg",
)


def risk_diagnostics_tool(
    mode: ToolMode,
    *,
    request: dict | None = None,
    settings: ProjectSettings | None = None,
) -> DiagnosticsOutput | StressOutput:
    normalized_mode = _normalize_mode(mode)
    _validate_request(request)

    if normalized_mode is ToolMode.DIAGNOSTICS:
        return _build_diagnostics_output(load_diagnostics_artifacts(settings=settings))
    if normalized_mode is ToolMode.STRESS:
        return _build_stress_output(load_stress_artifacts(settings=settings))

    raise StructuredArtifactLoadError(f"Unsupported tool mode: {normalized_mode}")


def computed_source_id_for_mode(mode: ToolMode) -> str:
    normalized_mode = _normalize_mode(mode)
    if normalized_mode is ToolMode.DIAGNOSTICS:
        return _DIAGNOSTICS_SOURCE_ID
    if normalized_mode is ToolMode.STRESS:
        return _STRESS_SOURCE_ID
    raise StructuredArtifactLoadError(f"Unsupported tool mode: {normalized_mode}")


def _build_diagnostics_output(bundle: DiagnosticsArtifactBundle) -> DiagnosticsOutput:
    diagnostics_summary = bundle.diagnostics_summary
    metric_reference = bundle.metric_reference
    if diagnostics_summary.get("mode") != ToolMode.DIAGNOSTICS.value:
        raise StructuredArtifactLoadError("diagnostics_summary mode must be 'diagnostics'.")

    thresholds = _validated_drift_thresholds(metric_reference)
    metrics = diagnostics_summary.get("metrics")
    if not isinstance(metrics, dict):
        raise StructuredArtifactLoadError("diagnostics_summary metrics section is missing or invalid.")

    top_drift_features = _normalize_top_drift_features(metrics.get("top_drift_features"), thresholds)
    score_shift = _normalize_score_shift(metrics.get("score_shift"), metric_reference)
    calibration = _normalize_calibration(metrics.get("calibration"), metric_reference)
    reference_metrics = _normalize_reference_metrics(metrics.get("reference_metrics"), metric_reference)

    return DiagnosticsOutput(
        summary=_require_string(diagnostics_summary.get("summary"), "diagnostics_summary.summary"),
        metrics=DiagnosticsMetrics(
            top_drift_features=top_drift_features,
            score_shift=score_shift,
            calibration=calibration,
            reference_metrics=reference_metrics,
        ),
        quality_flags=_normalize_string_list(diagnostics_summary.get("quality_flags"), "diagnostics_summary.quality_flags"),
        assumptions=_normalize_string_list(diagnostics_summary.get("assumptions"), "diagnostics_summary.assumptions"),
        limitations=_normalize_string_list(diagnostics_summary.get("limitations"), "diagnostics_summary.limitations"),
    )


def _build_stress_output(bundle: StressArtifactBundle) -> StressOutput:
    stress_summary = bundle.stress_summary
    scenario_config = bundle.scenario_config
    if stress_summary.get("mode") != ToolMode.STRESS.value:
        raise StructuredArtifactLoadError("stress_summary mode must be 'stress'.")

    metrics = stress_summary.get("metrics")
    if not isinstance(metrics, dict):
        raise StructuredArtifactLoadError("stress_summary metrics section is missing or invalid.")

    scenario_order = _validated_scenario_order(scenario_config)
    baseline = _normalize_scenario_metrics(metrics.get("baseline"), "stress_summary.metrics.baseline")
    mild = _normalize_scenario_metrics(metrics.get("mild"), "stress_summary.metrics.mild")
    severe = _normalize_scenario_metrics(metrics.get("severe"), "stress_summary.metrics.severe")
    delta_mean_pd = _validate_delta_map("delta_mean_pd", baseline, mild, severe, metrics, scenario_metric="mean_pd")
    delta_tail_pd = _validate_delta_map("delta_tail_pd", baseline, mild, severe, metrics, scenario_metric="p95_pd")
    delta_el_proxy = _validate_delta_map(
        "delta_el_proxy",
        baseline,
        mild,
        severe,
        metrics,
        scenario_metric="el_proxy_mean",
    )
    monotonicity_passed = _recompute_monotonicity(
        scenario_config=scenario_config,
        scenario_order=scenario_order,
        scenario_metrics={"baseline": baseline, "mild": mild, "severe": severe},
    )

    quality_flags = _normalize_string_list(stress_summary.get("quality_flags"), "stress_summary.quality_flags")
    if not monotonicity_passed and "scenario_monotonicity_failed" not in quality_flags:
        quality_flags.append("scenario_monotonicity_failed")

    return StressOutput(
        summary=_require_string(stress_summary.get("summary"), "stress_summary.summary"),
        metrics=StressMetrics(
            baseline=baseline,
            mild=mild,
            severe=severe,
            delta_mean_pd=delta_mean_pd,
            delta_tail_pd=delta_tail_pd,
            delta_el_proxy=delta_el_proxy,
            monotonicity_passed=monotonicity_passed,
        ),
        quality_flags=quality_flags,
        assumptions=_normalize_string_list(stress_summary.get("assumptions"), "stress_summary.assumptions"),
        limitations=_normalize_string_list(stress_summary.get("limitations"), "stress_summary.limitations"),
    )


def _normalize_mode(mode: ToolMode | str) -> ToolMode:
    if isinstance(mode, ToolMode):
        return mode
    if isinstance(mode, str):
        try:
            return ToolMode(mode)
        except ValueError as exc:
            raise StructuredArtifactLoadError(f"Unsupported tool mode: {mode}") from exc
    raise TypeError("mode must be a ToolMode or supported string.")


def _validate_request(request: dict | None) -> None:
    if request is not None and not isinstance(request, dict):
        raise TypeError("request must be a dict or None.")


def _validated_drift_thresholds(metric_reference: dict[str, Any]) -> dict[str, Any]:
    metrics = metric_reference.get("metrics")
    if not isinstance(metrics, dict):
        raise StructuredArtifactLoadError("metric_reference metrics section is missing or invalid.")

    drift_thresholds = metrics.get("drift_thresholds")
    if not isinstance(drift_thresholds, dict):
        raise StructuredArtifactLoadError("metric_reference drift_thresholds section is missing or invalid.")

    stable_lt = _require_float(drift_thresholds.get("stable_lt"), "metric_reference.metrics.drift_thresholds.stable_lt")
    moderate_gte = _require_float(
        drift_thresholds.get("moderate_gte"),
        "metric_reference.metrics.drift_thresholds.moderate_gte",
    )
    moderate_lt = _require_float(
        drift_thresholds.get("moderate_lt"),
        "metric_reference.metrics.drift_thresholds.moderate_lt",
    )
    material_gte = _require_float(
        drift_thresholds.get("material_gte"),
        "metric_reference.metrics.drift_thresholds.material_gte",
    )
    flag_mapping = drift_thresholds.get("flag_mapping")
    if not isinstance(flag_mapping, dict):
        raise StructuredArtifactLoadError("metric_reference drift flag_mapping is missing or invalid.")

    normalized_flag_mapping = {
        "stable": _require_string(flag_mapping.get("stable"), "metric_reference.metrics.drift_thresholds.flag_mapping.stable"),
        "moderate": _require_string(
            flag_mapping.get("moderate"),
            "metric_reference.metrics.drift_thresholds.flag_mapping.moderate",
        ),
        "material": _require_string(
            flag_mapping.get("material"),
            "metric_reference.metrics.drift_thresholds.flag_mapping.material",
        ),
    }
    if not stable_lt == moderate_gte or not moderate_gte < moderate_lt <= material_gte:
        raise StructuredArtifactLoadError("metric_reference drift thresholds are not ordered consistently.")

    return {
        "stable_lt": stable_lt,
        "moderate_gte": moderate_gte,
        "moderate_lt": moderate_lt,
        "material_gte": material_gte,
        "flag_mapping": normalized_flag_mapping,
    }


def _normalize_top_drift_features(raw_value: Any, thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw_value, list) or not raw_value:
        raise StructuredArtifactLoadError("diagnostics_summary.metrics.top_drift_features is missing or invalid.")

    normalized = []
    for index, feature_payload in enumerate(raw_value, start=1):
        if not isinstance(feature_payload, dict):
            raise StructuredArtifactLoadError(f"top_drift_features entry {index} is not a dictionary.")

        feature_name = _require_string(feature_payload.get("feature"), f"top_drift_features[{index}].feature")
        feature_type = _require_string(feature_payload.get("feature_type"), f"top_drift_features[{index}].feature_type")
        psi = _require_float(feature_payload.get("psi"), f"top_drift_features[{index}].psi")
        flag = _require_string(feature_payload.get("flag"), f"top_drift_features[{index}].flag")
        expected_flag = _expected_drift_flag(psi, thresholds)
        if flag != expected_flag:
            raise StructuredArtifactLoadError(
                f"top_drift_features entry {index} flag '{flag}' does not match expected '{expected_flag}'."
            )
        normalized.append(
            {
                "feature": feature_name,
                "feature_type": feature_type,
                "psi": psi,
                "flag": flag,
            }
        )

    normalized.sort(key=lambda item: (-float(item["psi"]), str(item["feature"])))
    return normalized


def _normalize_score_shift(raw_value: Any, metric_reference: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError("diagnostics_summary.metrics.score_shift is missing or invalid.")

    reference_models = _metric_reference_models(metric_reference, "score_stability")
    champion_reference = reference_models["champion_logit"]
    challenger_reference = reference_models["challenger_lgbm"]

    champion_payload = _normalize_score_shift_model(raw_value.get("champion_logit"), "champion_logit")
    challenger_payload = _normalize_score_shift_model(raw_value.get("challenger_lgbm"), "challenger_lgbm")
    _assert_close_dict(champion_payload, champion_reference, ("score_psi", "score_ks_stat"), "champion_logit")
    _assert_close_dict(challenger_payload, challenger_reference, ("score_psi", "score_ks_stat"), "challenger_lgbm")

    return {
        "finding": _require_string(raw_value.get("finding"), "diagnostics_summary.metrics.score_shift.finding"),
        "champion_logit": champion_payload,
        "challenger_lgbm": challenger_payload,
    }


def _normalize_calibration(raw_value: Any, metric_reference: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError("diagnostics_summary.metrics.calibration is missing or invalid.")

    reference_models = _metric_reference_models(metric_reference, "performance_calibration")
    champion_reference = _metric_reference_split(reference_models["champion_logit"], "oot", "champion_logit")
    challenger_reference = _metric_reference_split(reference_models["challenger_lgbm"], "oot", "challenger_lgbm")

    champion_payload = _normalize_calibration_model(raw_value.get("champion_oot"), "champion_oot")
    challenger_payload = _normalize_calibration_model(raw_value.get("challenger_oot"), "challenger_oot")
    calibration_keys = ("mean_pd", "observed_default_rate", "calibration_intercept", "calibration_slope")
    _assert_close_dict(champion_payload, champion_reference, calibration_keys, "champion_oot")
    _assert_close_dict(challenger_payload, challenger_reference, calibration_keys, "challenger_oot")

    return {
        "finding": _require_string(raw_value.get("finding"), "diagnostics_summary.metrics.calibration.finding"),
        "champion_oot": champion_payload,
        "challenger_oot": challenger_payload,
    }


def _normalize_reference_metrics(raw_value: Any, metric_reference: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError("diagnostics_summary.metrics.reference_metrics is missing or invalid.")

    performance_models = _metric_reference_models(metric_reference, "performance_calibration")
    drift_thresholds = _validated_drift_thresholds(metric_reference)
    psi_thresholds = raw_value.get("psi_thresholds")
    if not isinstance(psi_thresholds, dict):
        raise StructuredArtifactLoadError("diagnostics_summary.metrics.reference_metrics.psi_thresholds is missing or invalid.")

    champion_oot_auc = _require_float(raw_value.get("champion_oot_auc"), "reference_metrics.champion_oot_auc")
    champion_oot_brier = _require_float(raw_value.get("champion_oot_brier"), "reference_metrics.champion_oot_brier")
    challenger_oot_auc = _require_float(raw_value.get("challenger_oot_auc"), "reference_metrics.challenger_oot_auc")
    challenger_oot_brier = _require_float(raw_value.get("challenger_oot_brier"), "reference_metrics.challenger_oot_brier")
    _assert_float_close(champion_oot_auc, performance_models["champion_logit"]["oot"]["auc"], "champion_oot_auc")
    _assert_float_close(champion_oot_brier, performance_models["champion_logit"]["oot"]["brier"], "champion_oot_brier")
    _assert_float_close(challenger_oot_auc, performance_models["challenger_lgbm"]["oot"]["auc"], "challenger_oot_auc")
    _assert_float_close(challenger_oot_brier, performance_models["challenger_lgbm"]["oot"]["brier"], "challenger_oot_brier")

    psi_source_pages = psi_thresholds.get("source_pages")
    if not isinstance(psi_source_pages, list) or not psi_source_pages:
        raise StructuredArtifactLoadError("reference_metrics.psi_thresholds.source_pages is missing or invalid.")
    raw_flag_mapping = psi_thresholds.get("flag_mapping")
    if not isinstance(raw_flag_mapping, dict):
        raise StructuredArtifactLoadError("reference_metrics.psi_thresholds.flag_mapping is missing or invalid.")

    psi_stable_lt = _require_float(psi_thresholds.get("stable_lt"), "reference_metrics.psi_thresholds.stable_lt")
    psi_moderate_gte = _require_float(
        psi_thresholds.get("moderate_gte"),
        "reference_metrics.psi_thresholds.moderate_gte",
    )
    psi_moderate_lt = _require_float(psi_thresholds.get("moderate_lt"), "reference_metrics.psi_thresholds.moderate_lt")
    psi_material_gte = _require_float(psi_thresholds.get("material_gte"), "reference_metrics.psi_thresholds.material_gte")
    _assert_float_close(psi_stable_lt, drift_thresholds["stable_lt"], "reference_metrics.psi_thresholds.stable_lt")
    _assert_float_close(
        psi_moderate_gte,
        drift_thresholds["moderate_gte"],
        "reference_metrics.psi_thresholds.moderate_gte",
    )
    _assert_float_close(psi_moderate_lt, drift_thresholds["moderate_lt"], "reference_metrics.psi_thresholds.moderate_lt")
    _assert_float_close(psi_material_gte, drift_thresholds["material_gte"], "reference_metrics.psi_thresholds.material_gte")

    stable_flag = _require_string(raw_flag_mapping.get("stable"), "reference_metrics.psi_thresholds.flag_mapping.stable")
    moderate_flag = _require_string(
        raw_flag_mapping.get("moderate"),
        "reference_metrics.psi_thresholds.flag_mapping.moderate",
    )
    material_flag = _require_string(
        raw_flag_mapping.get("material"),
        "reference_metrics.psi_thresholds.flag_mapping.material",
    )
    if stable_flag != drift_thresholds["flag_mapping"]["stable"]:
        raise StructuredArtifactLoadError("reference_metrics.psi_thresholds.flag_mapping.stable is inconsistent.")
    if moderate_flag != drift_thresholds["flag_mapping"]["moderate"]:
        raise StructuredArtifactLoadError("reference_metrics.psi_thresholds.flag_mapping.moderate is inconsistent.")
    if material_flag != drift_thresholds["flag_mapping"]["material"]:
        raise StructuredArtifactLoadError("reference_metrics.psi_thresholds.flag_mapping.material is inconsistent.")

    return {
        "champion_oot_auc": champion_oot_auc,
        "champion_oot_brier": champion_oot_brier,
        "challenger_oot_auc": challenger_oot_auc,
        "challenger_oot_brier": challenger_oot_brier,
        "psi_thresholds": {
            "source_pages": [int(page) for page in psi_source_pages],
            "stable_lt": psi_stable_lt,
            "moderate_gte": psi_moderate_gte,
            "moderate_lt": psi_moderate_lt,
            "material_gte": psi_material_gte,
            "flag_mapping": {
                "stable": stable_flag,
                "moderate": moderate_flag,
                "material": material_flag,
            },
        },
    }


def _validated_scenario_order(scenario_config: dict[str, Any]) -> tuple[str, ...]:
    raw_order = scenario_config.get("scenario_order")
    if not isinstance(raw_order, list):
        raise StructuredArtifactLoadError("scenario_config.scenario_order is missing or invalid.")

    normalized_order = tuple(_require_string(item, f"scenario_order[{index}]") for index, item in enumerate(raw_order))
    if normalized_order != _SCENARIO_ORDER:
        raise StructuredArtifactLoadError("scenario_config.scenario_order does not match the frozen Phase 4 order.")
    return normalized_order


def _normalize_scenario_metrics(raw_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError(f"{label} is missing or invalid.")

    normalized = {}
    ordered_keys = [key for key in _SCENARIO_METRIC_ORDER if key in raw_value]
    ordered_keys.extend(sorted(key for key in raw_value if key not in _SCENARIO_METRIC_ORDER))
    for key in ordered_keys:
        value = raw_value[key]
        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, (int, float)):
            normalized[key] = float(value)
        else:
            raise StructuredArtifactLoadError(f"{label}.{key} must be numeric or boolean.")
    return normalized


def _validate_delta_map(
    delta_name: str,
    baseline: dict[str, Any],
    mild: dict[str, Any],
    severe: dict[str, Any],
    metrics: dict[str, Any],
    *,
    scenario_metric: str | None = None,
) -> ScenarioDeltaMap:
    metric_key = scenario_metric or delta_name
    baseline_value = _require_numeric_metric(baseline, metric_key, f"stress_summary.metrics.baseline.{metric_key}")
    mild_value = _require_numeric_metric(mild, metric_key, f"stress_summary.metrics.mild.{metric_key}")
    severe_value = _require_numeric_metric(severe, metric_key, f"stress_summary.metrics.severe.{metric_key}")
    recomputed = ScenarioDeltaMap(
        mild=round(mild_value - baseline_value, 6),
        severe=round(severe_value - baseline_value, 6),
    )

    raw_delta_map = metrics.get(delta_name)
    if not isinstance(raw_delta_map, dict):
        raise StructuredArtifactLoadError(f"stress_summary.metrics.{delta_name} is missing or invalid.")

    mild_delta = _require_float(raw_delta_map.get("mild"), f"stress_summary.metrics.{delta_name}.mild")
    severe_delta = _require_float(raw_delta_map.get("severe"), f"stress_summary.metrics.{delta_name}.severe")
    _assert_float_close(mild_delta, recomputed.mild, f"{delta_name}.mild")
    _assert_float_close(severe_delta, recomputed.severe, f"{delta_name}.severe")
    return ScenarioDeltaMap(mild=mild_delta, severe=severe_delta)


def _recompute_monotonicity(
    *,
    scenario_config: dict[str, Any],
    scenario_order: tuple[str, ...],
    scenario_metrics: dict[str, dict[str, Any]],
) -> bool:
    expectation = scenario_config.get("monotonicity_expectation")
    if not isinstance(expectation, dict):
        raise StructuredArtifactLoadError("scenario_config.monotonicity_expectation is missing or invalid.")

    direction = _require_string(expectation.get("direction"), "scenario_config.monotonicity_expectation.direction")
    if direction != "non_decreasing":
        raise StructuredArtifactLoadError("scenario_config.monotonicity_expectation.direction must be 'non_decreasing'.")

    metrics_to_check = expectation.get("metrics")
    if not isinstance(metrics_to_check, list) or not metrics_to_check:
        raise StructuredArtifactLoadError("scenario_config.monotonicity_expectation.metrics is missing or invalid.")

    for metric_name in metrics_to_check:
        metric_label = _require_string(metric_name, "scenario_config.monotonicity_expectation.metrics[]")
        values = [
            _require_numeric_metric(
                scenario_metrics[scenario_name],
                metric_label,
                f"stress_summary.metrics.{scenario_name}.{metric_label}",
            )
            for scenario_name in scenario_order
        ]
        if values != sorted(values):
            return False
    return True


def _metric_reference_models(metric_reference: dict[str, Any], section_name: str) -> dict[str, Any]:
    metrics = metric_reference.get("metrics")
    if not isinstance(metrics, dict):
        raise StructuredArtifactLoadError("metric_reference metrics section is missing or invalid.")

    section = metrics.get(section_name)
    if not isinstance(section, dict):
        raise StructuredArtifactLoadError(f"metric_reference.metrics.{section_name} is missing or invalid.")

    models = section.get("models")
    if not isinstance(models, dict):
        raise StructuredArtifactLoadError(f"metric_reference.metrics.{section_name}.models is missing or invalid.")

    champion_logit = models.get("champion_logit")
    challenger_lgbm = models.get("challenger_lgbm")
    if not isinstance(champion_logit, dict) or not isinstance(challenger_lgbm, dict):
        raise StructuredArtifactLoadError(f"metric_reference.metrics.{section_name}.models is incomplete.")
    return {
        "champion_logit": champion_logit,
        "challenger_lgbm": challenger_lgbm,
    }


def _metric_reference_split(model_section: dict[str, Any], split_name: str, label: str) -> dict[str, Any]:
    split = model_section.get(split_name)
    if not isinstance(split, dict):
        raise StructuredArtifactLoadError(f"metric_reference {label}.{split_name} split is missing or invalid.")
    return split


def _normalize_score_shift_model(raw_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError(f"diagnostics_summary.metrics.score_shift.{label} is missing or invalid.")

    return {
        "score_psi": _require_float(raw_value.get("score_psi"), f"score_shift.{label}.score_psi"),
        "score_ks_stat": _require_float(raw_value.get("score_ks_stat"), f"score_shift.{label}.score_ks_stat"),
    }


def _normalize_calibration_model(raw_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise StructuredArtifactLoadError(f"diagnostics_summary.metrics.calibration.{label} is missing or invalid.")

    return {
        "mean_pd": _require_float(raw_value.get("mean_pd"), f"calibration.{label}.mean_pd"),
        "observed_default_rate": _require_float(
            raw_value.get("observed_default_rate"),
            f"calibration.{label}.observed_default_rate",
        ),
        "calibration_intercept": _require_float(
            raw_value.get("calibration_intercept"),
            f"calibration.{label}.calibration_intercept",
        ),
        "calibration_slope": _require_float(raw_value.get("calibration_slope"), f"calibration.{label}.calibration_slope"),
    }


def _normalize_string_list(raw_value: Any, label: str) -> list[str]:
    if not isinstance(raw_value, list):
        raise StructuredArtifactLoadError(f"{label} is missing or invalid.")

    return [_require_string(item, f"{label}[{index}]") for index, item in enumerate(raw_value, start=1)]


def _expected_drift_flag(psi: float, thresholds: dict[str, Any]) -> str:
    if psi < thresholds["stable_lt"]:
        return thresholds["flag_mapping"]["stable"]
    if psi < thresholds["moderate_lt"]:
        return thresholds["flag_mapping"]["moderate"]
    return thresholds["flag_mapping"]["material"]


def _assert_close_dict(
    payload: dict[str, Any],
    reference: dict[str, Any],
    keys: tuple[str, ...],
    label: str,
) -> None:
    for key in keys:
        _assert_float_close(
            payload[key],
            _require_float(reference.get(key), f"metric_reference.{label}.{key}"),
            f"{label}.{key}",
        )


def _assert_float_close(actual: float | None, expected: float | None, label: str) -> None:
    if actual is None or expected is None:
        raise StructuredArtifactLoadError(f"{label} is missing or invalid.")
    if round(float(actual), 6) != round(float(expected), 6):
        raise StructuredArtifactLoadError(f"{label} does not match the frozen structured artifact reference.")


def _require_numeric_metric(payload: dict[str, Any], key: str, label: str) -> float:
    return _require_float(payload.get(key), label)


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredArtifactLoadError(f"{label} must be numeric.")
    return float(value)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuredArtifactLoadError(f"{label} must be a non-empty string.")
    return value
