from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app import ProjectSettings, load_settings


READY_STATUS = "ready"


class RiskDiagnosticsToolError(RuntimeError):
    """Raised when the deterministic diagnostics tool cannot complete safely."""


class StructuredArtifactLoadError(RiskDiagnosticsToolError):
    """Raised when frozen structured artifacts are missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class DiagnosticsArtifactBundle:
    data_manifest: dict[str, Any]
    diagnostics_summary: dict[str, Any]
    metric_reference: dict[str, Any]


@dataclass(frozen=True)
class StressArtifactBundle:
    data_manifest: dict[str, Any]
    stress_summary: dict[str, Any]
    scenario_config: dict[str, Any]


def load_diagnostics_artifacts(*, settings: ProjectSettings | None = None) -> DiagnosticsArtifactBundle:
    if settings is None:
        return _load_default_diagnostics_artifacts()

    data_manifest = _load_data_manifest(settings)
    _validate_manifest_artifact(data_manifest, "diagnostics_summary", settings.paths.diagnostics_summary_path)
    _validate_manifest_artifact(data_manifest, "metric_reference", settings.paths.metric_reference_path)
    diagnostics_summary = _read_json_artifact(settings.paths.diagnostics_summary_path, "diagnostics_summary")
    metric_reference = _read_json_artifact(settings.paths.metric_reference_path, "metric_reference")
    _require_ready_artifact(diagnostics_summary, "diagnostics_summary")
    _require_ready_artifact(metric_reference, "metric_reference")
    return DiagnosticsArtifactBundle(
        data_manifest=data_manifest,
        diagnostics_summary=diagnostics_summary,
        metric_reference=metric_reference,
    )


def load_stress_artifacts(*, settings: ProjectSettings | None = None) -> StressArtifactBundle:
    if settings is None:
        return _load_default_stress_artifacts()

    data_manifest = _load_data_manifest(settings)
    _validate_manifest_artifact(data_manifest, "stress_summary", settings.paths.stress_summary_path)
    _validate_manifest_artifact(data_manifest, "scenario_config", settings.paths.scenario_config_path)
    stress_summary = _read_json_artifact(settings.paths.stress_summary_path, "stress_summary")
    scenario_config = _read_yaml_artifact(settings.paths.scenario_config_path, "scenario_config")
    _require_ready_artifact(stress_summary, "stress_summary")
    _require_ready_artifact(scenario_config, "scenario_config")
    return StressArtifactBundle(
        data_manifest=data_manifest,
        stress_summary=stress_summary,
        scenario_config=scenario_config,
    )


@lru_cache(maxsize=1)
def _load_default_diagnostics_artifacts() -> DiagnosticsArtifactBundle:
    return load_diagnostics_artifacts(settings=load_settings())


@lru_cache(maxsize=1)
def _load_default_stress_artifacts() -> StressArtifactBundle:
    return load_stress_artifacts(settings=load_settings())


def _load_data_manifest(settings: ProjectSettings) -> dict[str, Any]:
    data_manifest = _read_json_artifact(settings.paths.data_manifest_path, "data_manifest")
    _require_ready_artifact(data_manifest, "data_manifest")
    return data_manifest


def _validate_manifest_artifact(data_manifest: dict[str, Any], artifact_name: str, expected_path: Path) -> None:
    artifacts = data_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise StructuredArtifactLoadError("data_manifest artifacts section is missing or invalid.")

    artifact_info = artifacts.get(artifact_name)
    if not isinstance(artifact_info, dict):
        raise StructuredArtifactLoadError(f"data_manifest is missing the '{artifact_name}' artifact entry.")

    relative_path = artifact_info.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise StructuredArtifactLoadError(f"data_manifest path for '{artifact_name}' is missing or invalid.")

    resolved_path = expected_path.parents[2] / relative_path
    if resolved_path != expected_path:
        raise StructuredArtifactLoadError(
            f"data_manifest path for '{artifact_name}' does not match the frozen project layout."
        )


def _read_json_artifact(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.exists():
        raise StructuredArtifactLoadError(f"Missing structured artifact: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StructuredArtifactLoadError(f"Invalid JSON in {artifact_name}: {path}") from exc

    if not isinstance(payload, dict):
        raise StructuredArtifactLoadError(f"{artifact_name} must deserialize to a dictionary.")
    return payload


def _read_yaml_artifact(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.exists():
        raise StructuredArtifactLoadError(f"Missing structured artifact: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StructuredArtifactLoadError(f"Invalid YAML in {artifact_name}: {path}") from exc

    if not isinstance(payload, dict):
        raise StructuredArtifactLoadError(f"{artifact_name} must deserialize to a dictionary.")
    return payload


def _require_ready_artifact(payload: dict[str, Any], artifact_name: str) -> None:
    if payload.get("status") != READY_STATUS:
        raise StructuredArtifactLoadError(f"{artifact_name} is not ready.")
