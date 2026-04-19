from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from app import load_settings
from app.config import ProjectPaths, ProjectSettings
from schemas import ToolMode
from tools import StructuredArtifactLoadError, computed_source_id_for_mode, risk_diagnostics_tool


def _make_temp_settings(tmp_path: Path) -> ProjectSettings:
    base_settings = load_settings()
    paths = ProjectPaths.from_repo_root(tmp_path)
    for directory in (
        paths.structured_dir,
        paths.corpus_dir,
        paths.eval_dir,
        paths.outputs_dir,
        paths.traces_dir,
        paths.eval_results_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for source_path, target_path in (
        (base_settings.paths.data_manifest_path, paths.data_manifest_path),
        (base_settings.paths.metric_reference_path, paths.metric_reference_path),
        (base_settings.paths.diagnostics_summary_path, paths.diagnostics_summary_path),
        (base_settings.paths.stress_summary_path, paths.stress_summary_path),
        (base_settings.paths.scenario_config_path, paths.scenario_config_path),
    ):
        shutil.copy2(source_path, target_path)

    return ProjectSettings(
        project_name=base_settings.project_name,
        provider=base_settings.provider,
        paths=paths,
        frozen_corpus=base_settings.frozen_corpus,
    )


def test_phase4_diagnostics_mode_is_deterministic_and_provenanced() -> None:
    first_output = risk_diagnostics_tool(ToolMode.DIAGNOSTICS)
    second_output = risk_diagnostics_tool(ToolMode.DIAGNOSTICS)
    ignored_request_output = risk_diagnostics_tool(ToolMode.DIAGNOSTICS, request={"query": "ignored"})

    assert first_output == second_output == ignored_request_output
    assert first_output.source == "risk_diagnostics_tool"
    assert first_output.mode == "diagnostics"
    assert computed_source_id_for_mode(ToolMode.DIAGNOSTICS) == "risk_diagnostics_tool:diagnostics"
    assert first_output.metrics.top_drift_features[0]["feature"] == "revol util"
    assert first_output.metrics.top_drift_features == sorted(
        first_output.metrics.top_drift_features,
        key=lambda item: (-float(item["psi"]), str(item["feature"])),
    )


def test_phase4_stress_mode_is_deterministic_and_recomputes_monotonicity() -> None:
    first_output = risk_diagnostics_tool(ToolMode.STRESS)
    second_output = risk_diagnostics_tool(ToolMode.STRESS, request={"question_type": "numeric"})

    assert first_output == second_output
    assert first_output.source == "risk_diagnostics_tool"
    assert first_output.mode == "stress"
    assert computed_source_id_for_mode("stress") == "risk_diagnostics_tool:stress"
    assert first_output.metrics.monotonicity_passed is True
    assert first_output.metrics.delta_mean_pd.mild == pytest.approx(0.004528)
    assert first_output.metrics.delta_mean_pd.severe == pytest.approx(0.009725)
    assert list(first_output.metrics.model_dump().keys()) == [
        "baseline",
        "mild",
        "severe",
        "delta_mean_pd",
        "delta_tail_pd",
        "delta_el_proxy",
        "monotonicity_passed",
    ]
    assert list(first_output.metrics.baseline.keys()) == ["mean_pd", "p95_pd", "el_proxy_mean"]


def test_phase4_tool_accepts_none_request_and_does_not_mutate_artifacts(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    before_contents = {
        path: path.read_text(encoding="utf-8")
        for path in (
            settings.paths.data_manifest_path,
            settings.paths.metric_reference_path,
            settings.paths.diagnostics_summary_path,
            settings.paths.stress_summary_path,
            settings.paths.scenario_config_path,
        )
    }

    risk_diagnostics_tool(ToolMode.DIAGNOSTICS, request=None, settings=settings)
    risk_diagnostics_tool(ToolMode.STRESS, request=None, settings=settings)

    after_contents = {path: path.read_text(encoding="utf-8") for path in before_contents}
    assert after_contents == before_contents


def test_phase4_tool_rejects_unsupported_mode() -> None:
    with pytest.raises(StructuredArtifactLoadError, match="Unsupported tool mode"):
        risk_diagnostics_tool("unsupported")  # type: ignore[arg-type]


def test_phase4_tool_rejects_non_dict_request() -> None:
    with pytest.raises(TypeError, match="request must be a dict or None"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, request="bad-request")  # type: ignore[arg-type]


def test_phase4_tool_fails_fast_when_artifact_file_is_missing(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    settings.paths.diagnostics_summary_path.unlink()

    with pytest.raises(StructuredArtifactLoadError, match="Missing structured artifact"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, settings=settings)


def test_phase4_tool_fails_fast_when_manifest_is_not_ready(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    data_manifest = json.loads(settings.paths.data_manifest_path.read_text(encoding="utf-8"))
    data_manifest["status"] = "placeholder"
    settings.paths.data_manifest_path.write_text(json.dumps(data_manifest), encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="data_manifest is not ready"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, settings=settings)


def test_phase4_tool_fails_fast_when_json_artifact_is_malformed(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    settings.paths.diagnostics_summary_path.write_text("{not-valid-json}\n", encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="Invalid JSON in diagnostics_summary"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, settings=settings)


def test_phase4_tool_fails_fast_when_yaml_artifact_is_malformed(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    settings.paths.scenario_config_path.write_text("status: ready\nscenario_order: [baseline\n", encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="Invalid YAML in scenario_config"):
        risk_diagnostics_tool(ToolMode.STRESS, settings=settings)


def test_phase4_tool_fails_fast_when_required_metric_key_is_missing(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    diagnostics_summary = json.loads(settings.paths.diagnostics_summary_path.read_text(encoding="utf-8"))
    del diagnostics_summary["metrics"]["calibration"]["champion_oot"]["calibration_slope"]
    settings.paths.diagnostics_summary_path.write_text(json.dumps(diagnostics_summary), encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="calibration.champion_oot.calibration_slope must be numeric"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, settings=settings)


def test_phase4_tool_fails_fast_when_drift_flag_conflicts_with_thresholds(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    diagnostics_summary = json.loads(settings.paths.diagnostics_summary_path.read_text(encoding="utf-8"))
    diagnostics_summary["metrics"]["top_drift_features"][0]["flag"] = "GREEN"
    settings.paths.diagnostics_summary_path.write_text(json.dumps(diagnostics_summary), encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="does not match expected"):
        risk_diagnostics_tool(ToolMode.DIAGNOSTICS, settings=settings)


def test_phase4_tool_fails_fast_when_scenario_order_drifts(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    scenario_config = yaml.safe_load(settings.paths.scenario_config_path.read_text(encoding="utf-8"))
    scenario_config["scenario_order"] = ["mild", "baseline", "severe"]
    settings.paths.scenario_config_path.write_text(yaml.safe_dump(scenario_config, sort_keys=False), encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="scenario_config.scenario_order does not match"):
        risk_diagnostics_tool(ToolMode.STRESS, settings=settings)


def test_phase4_tool_fails_fast_when_stress_delta_map_conflicts_with_values(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    stress_summary = json.loads(settings.paths.stress_summary_path.read_text(encoding="utf-8"))
    stress_summary["metrics"]["delta_mean_pd"]["severe"] = 99.0
    settings.paths.stress_summary_path.write_text(json.dumps(stress_summary), encoding="utf-8")

    with pytest.raises(StructuredArtifactLoadError, match="delta_mean_pd.severe does not match"):
        risk_diagnostics_tool(ToolMode.STRESS, settings=settings)


def test_phase4_tool_returns_false_monotonicity_when_stress_ladder_breaks(tmp_path: Path) -> None:
    settings = _make_temp_settings(tmp_path)
    stress_summary = json.loads(settings.paths.stress_summary_path.read_text(encoding="utf-8"))
    stress_summary["metrics"]["severe"]["mean_pd"] = 0.194
    stress_summary["metrics"]["delta_mean_pd"]["severe"] = 0.003383
    settings.paths.stress_summary_path.write_text(json.dumps(stress_summary), encoding="utf-8")

    output = risk_diagnostics_tool(ToolMode.STRESS, settings=settings)

    assert output.metrics.monotonicity_passed is False
    assert "scenario_monotonicity_failed" in output.quality_flags
