from __future__ import annotations

import json

import pytest
import yaml

from app.config import load_settings
from ingestion import phase2 as phase2_module


FROZEN_DOC_IDS = {
    "Model_Validation_Report",
    "SR11_7_Model_Risk_Management",
    "Basel_Stress_Testing_Principles_2018",
}

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "text",
    "doc_id",
    "page",
    "section_path",
    "chunk_type",
    "topic_tags",
    "authority_level",
    "document_role",
}

ALLOWED_CHUNK_TYPES = {"text", "table_like", "bullet_list"}


def test_phase2_corpus_artifacts_are_ready_and_consistent() -> None:
    settings = load_settings()
    corpus_manifest = json.loads(settings.paths.corpus_manifest_path.read_text(encoding="utf-8"))
    chunks = [
        json.loads(line)
        for line in settings.paths.chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert corpus_manifest["status"] == "ready"
    assert corpus_manifest["document_count"] == 3
    assert corpus_manifest["chunk_count"] == len(chunks)
    assert {document["doc_id"] for document in corpus_manifest["documents"]} == FROZEN_DOC_IDS
    assert sum(document["chunk_count"] for document in corpus_manifest["documents"]) == len(chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert {chunk["doc_id"] for chunk in chunks} == FROZEN_DOC_IDS

    for chunk in chunks:
        assert REQUIRED_CHUNK_FIELDS.issubset(chunk)
        assert chunk["chunk_type"] in ALLOWED_CHUNK_TYPES
        assert chunk["page"] >= 1
        assert chunk["section_path"]
        assert isinstance(chunk["topic_tags"], list)
        assert chunk["text"].strip()


def test_phase2_structured_artifacts_are_ready_and_traceable() -> None:
    settings = load_settings()
    metric_reference = json.loads(settings.paths.metric_reference_path.read_text(encoding="utf-8"))
    diagnostics_summary = json.loads(settings.paths.diagnostics_summary_path.read_text(encoding="utf-8"))
    stress_summary = json.loads(settings.paths.stress_summary_path.read_text(encoding="utf-8"))
    data_manifest = json.loads(settings.paths.data_manifest_path.read_text(encoding="utf-8"))
    scenario_config = yaml.safe_load(settings.paths.scenario_config_path.read_text(encoding="utf-8"))

    assert metric_reference["status"] == "ready"
    assert diagnostics_summary["status"] == "ready"
    assert diagnostics_summary["mode"] == "diagnostics"
    assert stress_summary["status"] == "ready"
    assert stress_summary["mode"] == "stress"
    assert data_manifest["status"] == "ready"
    assert scenario_config["status"] == "ready"
    assert list(scenario_config["scenario_order"]) == ["baseline", "mild", "severe"]
    assert set(scenario_config["scenarios"]) == {"baseline", "mild", "severe"}
    assert stress_summary["metrics"]["monotonicity_passed"] is True

    assert set(data_manifest["artifacts"]) == {
        "metric_reference",
        "scenario_config",
        "diagnostics_summary",
        "stress_summary",
    }

    for artifact_name, artifact_info in data_manifest["artifacts"].items():
        assert artifact_info["path"].startswith("data/structured/")
        assert artifact_info["record_count"] > 0
        assert artifact_info["derivation_policy"] == "direct_extract_or_simple_derivation"
        assert artifact_info["source_pages"]

    assert metric_reference["metrics"]["drift_thresholds"]["material_gte"] == 0.25
    assert diagnostics_summary["metrics"]["top_drift_features"][0]["feature"] == "revol util"
    assert stress_summary["metrics"]["delta_mean_pd"]["severe"] > stress_summary["metrics"]["delta_mean_pd"]["mild"]


def test_phase2_build_fails_fast_when_pdf_open_fails(monkeypatch) -> None:
    settings = load_settings()

    def _raise_pdf_open_error(*_args, **_kwargs):
        raise RuntimeError("cannot open pdf")

    monkeypatch.setattr(phase2_module.fitz, "open", _raise_pdf_open_error)

    with pytest.raises(phase2_module.Phase2BuildError, match="Failed to open PDF for Model_Validation_Report"):
        phase2_module._build_document_chunks(settings, "Model_Validation_Report")


def test_phase2_build_fails_fast_when_no_chunkable_text_extracted(monkeypatch) -> None:
    settings = load_settings()

    monkeypatch.setattr(phase2_module, "_clean_page_lines", lambda *_args, **_kwargs: [])

    with pytest.raises(
        phase2_module.Phase2BuildError,
        match="No chunkable text extracted for Model_Validation_Report",
    ):
        phase2_module._build_document_chunks(settings, "Model_Validation_Report")


def test_phase2_build_fails_fast_on_duplicate_chunk_ids(monkeypatch) -> None:
    settings = load_settings()

    def _duplicate_chunk_result(settings_obj, doc_id):
        document = next(document for document in settings_obj.frozen_corpus if document.doc_id == doc_id)
        return phase2_module.DocumentBuildResult(
            doc_id=document.doc_id,
            path=f"docs/domain/{document.filename}",
            page_count=1,
            toc_used=True,
            chunk_records=[
                {
                    "chunk_id": "chunk:duplicate",
                    "text": f"Duplicate content for {doc_id}",
                    "doc_id": document.doc_id,
                    "page": 1,
                    "section_path": "Test Section",
                    "chunk_type": "text",
                    "topic_tags": [],
                    "authority_level": document.authority_level,
                    "document_role": document.document_role,
                }
            ],
        )

    monkeypatch.setattr(phase2_module, "_build_document_chunks", _duplicate_chunk_result)

    with pytest.raises(phase2_module.Phase2BuildError, match="Duplicate chunk IDs generated"):
        phase2_module.build_phase2_artifacts(settings)
