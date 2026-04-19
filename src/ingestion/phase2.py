from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
import yaml

from app import ProjectSettings, load_settings
from schemas import RetrievalChunk

TARGET_WORD_RANGE = (220, 320)
CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_TABLE = "table_like"
CHUNK_TYPE_BULLETS = "bullet_list"
READY_STATUS = "ready"

MODEL_REPORT_DOC_ID = "Model_Validation_Report"
SR11_DOC_ID = "SR11_7_Model_Risk_Management"
BASEL_DOC_ID = "Basel_Stress_Testing_Principles_2018"

TOPIC_TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "calibration": ("calibration", "reliability", "slope", "intercept", "brier"),
    "drift": ("drift", "psi", "ks", "shift", "stability"),
    "stress_testing": ("stress", "scenario", "expected loss", "el proxy", "monotonic"),
    "performance": ("auc", "pr auc", "gini", "discrimination", "top10pct"),
    "governance": ("governance", "policy", "control", "board", "documentation"),
    "validation": ("validation", "effective challenge", "uncertainty", "limitations"),
    "challenger": ("challenger", "benchmark", "lightgbm", "shap"),
    "sensitivity": ("sensitivity", "spearman", "shock", "perturb"),
}

DOC_PAGE_DEFAULT_PATHS: dict[str, dict[int, list[str]]] = {
    MODEL_REPORT_DOC_ID: {
        1: ["Executive Summary"],
        2: ["Executive Summary"],
    },
    SR11_DOC_ID: {
        1: ["Cover Letter and Guidance Context"],
        2: ["Model Risk Definition and Effective Challenge"],
        3: ["Model Validation"],
        4: ["Governance, Policies, and Controls"],
        5: ["Governance, Policies, and Controls", "Contacts and Cross-Reference"],
    },
    BASEL_DOC_ID: {
        1: ["Front Matter"],
        2: ["Publication Notes"],
        3: ["Contents"],
        4: ["Contents"],
    },
}

LINE_SECTION_OVERRIDES: dict[str, dict[str, list[str]]] = {
    MODEL_REPORT_DOC_ID: {
        "executive summary": ["Executive Summary"],
    },
    SR11_DOC_ID: {
        "model validation": ["Model Validation"],
        "governance, policies, and controls": ["Governance, Policies, and Controls"],
        "contacts": ["Governance, Policies, and Controls", "Contacts"],
        "cross-reference:": ["Governance, Policies, and Controls", "Cross-Reference"],
    },
    BASEL_DOC_ID: {
        "introduction": ["Introduction"],
        "stress testing principles": ["Stress Testing Principles"],
    },
}

LINE_DROP_EXACT: dict[str, tuple[str, ...]] = {
    BASEL_DOC_ID: ("Stress testing principles",),
}

STRUCTURED_SOURCE_DOCUMENTS = {
    "diagnostics": [MODEL_REPORT_DOC_ID, SR11_DOC_ID],
    "stress": [MODEL_REPORT_DOC_ID, BASEL_DOC_ID],
}


class Phase2BuildError(RuntimeError):
    """Raised when the frozen Phase 2 artifact build cannot be completed safely."""


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class DocumentBuildResult:
    doc_id: str
    path: str
    page_count: int
    toc_used: bool
    chunk_records: list[dict[str, Any]]


@dataclass(frozen=True)
class Phase2Artifacts:
    chunks: list[dict[str, Any]]
    corpus_manifest: dict[str, Any]
    metric_reference: dict[str, Any]
    scenario_config: dict[str, Any]
    diagnostics_summary: dict[str, Any]
    stress_summary: dict[str, Any]
    data_manifest: dict[str, Any]


def build_phase2_artifacts(settings: ProjectSettings) -> Phase2Artifacts:
    document_results = [_build_document_chunks(settings, document.doc_id) for document in settings.frozen_corpus]
    chunk_records = [chunk for result in document_results for chunk in result.chunk_records]
    chunk_ids = [chunk["chunk_id"] for chunk in chunk_records]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise Phase2BuildError("Duplicate chunk IDs generated during Phase 2 build.")

    corpus_manifest = {
        "status": READY_STATUS,
        "document_count": len(document_results),
        "chunk_count": len(chunk_records),
        "chunking_policy": {
            "page_bounded": True,
            "target_word_range": list(TARGET_WORD_RANGE),
            "split_on": ["section_heading", "paragraph_boundary"],
            "overlap_words": 0,
            "toc_backed_docs": [MODEL_REPORT_DOC_ID, BASEL_DOC_ID],
            "fallback_section_docs": [SR11_DOC_ID],
        },
        "documents": [
            {
                "doc_id": result.doc_id,
                "path": result.path,
                "page_count": result.page_count,
                "toc_used": result.toc_used,
                "chunk_count": len(result.chunk_records),
                "authority_level": result.chunk_records[0]["authority_level"] if result.chunk_records else "",
                "document_role": result.chunk_records[0]["document_role"] if result.chunk_records else "",
            }
            for result in document_results
        ],
    }

    metric_reference = _build_metric_reference()
    diagnostics_summary = _build_diagnostics_summary(metric_reference)
    scenario_config = _build_scenario_config()
    stress_summary = _build_stress_summary()
    data_manifest = _build_data_manifest(metric_reference, scenario_config, diagnostics_summary, stress_summary)

    return Phase2Artifacts(
        chunks=chunk_records,
        corpus_manifest=corpus_manifest,
        metric_reference=metric_reference,
        scenario_config=scenario_config,
        diagnostics_summary=diagnostics_summary,
        stress_summary=stress_summary,
        data_manifest=data_manifest,
    )


def preview_phase2_build(settings: ProjectSettings, sample_chunks_per_doc: int = 2) -> str:
    document_results = [_build_document_chunks(settings, document.doc_id) for document in settings.frozen_corpus]
    lines = ["Phase 2 preview:"]
    for result in document_results:
        lines.append(
            f"- {result.doc_id}: pages={result.page_count}, toc_used={result.toc_used}, chunks={len(result.chunk_records)}"
        )
        for chunk in result.chunk_records[:sample_chunks_per_doc]:
            preview = chunk["text"][:120].replace("\n", " ")
            lines.append(
                "  "
                + json.dumps(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "page": chunk["page"],
                        "section_path": chunk["section_path"],
                        "chunk_type": chunk["chunk_type"],
                        "topic_tags": chunk["topic_tags"],
                        "preview": preview,
                    },
                    ensure_ascii=True,
                )
            )
    return "\n".join(lines)


def write_phase2_artifacts(settings: ProjectSettings, artifacts: Phase2Artifacts) -> None:
    _write_jsonl(settings.paths.chunks_path, artifacts.chunks)
    _write_json(settings.paths.corpus_manifest_path, artifacts.corpus_manifest)
    _write_json(settings.paths.metric_reference_path, artifacts.metric_reference)
    _write_json(settings.paths.diagnostics_summary_path, artifacts.diagnostics_summary)
    _write_json(settings.paths.stress_summary_path, artifacts.stress_summary)
    _write_json(settings.paths.data_manifest_path, artifacts.data_manifest)
    _write_yaml(settings.paths.scenario_config_path, artifacts.scenario_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 corpus and structured artifacts.")
    parser.add_argument("--preview", action="store_true", help="Print a small ingestion preview without writing files.")
    args = parser.parse_args()

    settings = load_settings()
    if args.preview:
        print(preview_phase2_build(settings))
        return

    artifacts = build_phase2_artifacts(settings)
    write_phase2_artifacts(settings, artifacts)
    print(
        json.dumps(
            {
                "status": READY_STATUS,
                "chunk_count": len(artifacts.chunks),
                "document_count": artifacts.corpus_manifest["document_count"],
            },
            indent=2,
        )
    )


def _build_document_chunks(settings: ProjectSettings, doc_id: str) -> DocumentBuildResult:
    document = next(document for document in settings.frozen_corpus if document.doc_id == doc_id)
    try:
        pdf = fitz.open(document.path)
    except Exception as exc:
        raise Phase2BuildError(f"Failed to open PDF for {doc_id}: {document.path}") from exc

    chunk_records: list[dict[str, Any]] = []
    page_count = pdf.page_count
    try:
        toc_entries = _extract_toc_entries(pdf)

        for page_number in range(1, page_count + 1):
            cleaned_lines = _clean_page_lines(doc_id, pdf.load_page(page_number - 1).get_text("text"), page_number)
            if not cleaned_lines:
                continue

            section_segments = _split_page_into_sections(doc_id, page_number, cleaned_lines, toc_entries)
            page_chunk_index = 0
            for segment in section_segments:
                chunk_texts = _chunk_section_lines(segment["lines"])
                for chunk_text in chunk_texts:
                    page_chunk_index += 1
                    chunk = RetrievalChunk(
                        chunk_id=f"chunk:{_slugify(doc_id)}:p{page_number}:{page_chunk_index}",
                        text=chunk_text,
                        doc_id=document.doc_id,
                        page=page_number,
                        section_path=" > ".join(segment["path"]),
                        chunk_type=_classify_chunk_type(chunk_text),
                        topic_tags=_derive_topic_tags(segment["path"], chunk_text),
                        authority_level=document.authority_level,
                        document_role=document.document_role,
                    )
                    chunk_records.append(chunk.model_dump(exclude_none=True))
    except Phase2BuildError:
        raise
    except Exception as exc:
        raise Phase2BuildError(f"Failed to extract chunkable text for {doc_id}: {document.path}") from exc
    finally:
        pdf.close()

    if not chunk_records:
        raise Phase2BuildError(f"No chunkable text extracted for {doc_id}: {document.path}")

    return DocumentBuildResult(
        doc_id=document.doc_id,
        path=str(document.path.relative_to(settings.paths.repo_root)),
        page_count=page_count,
        toc_used=bool(toc_entries),
        chunk_records=chunk_records,
    )


def _build_metric_reference() -> dict[str, Any]:
    return {
        "status": READY_STATUS,
        "source_documents": [MODEL_REPORT_DOC_ID],
        "metrics": {
            "performance_calibration": {
                "source_pages": [3, 12],
                "models": {
                    "champion_logit": {
                        "train": {
                            "auc": 0.7104,
                            "pr_auc": 0.3824,
                            "gini": 0.4208,
                            "ks": 0.3048,
                            "brier": 0.1485,
                            "top10pct_default_capture": 0.2252,
                            "mean_pd": 0.2067,
                            "observed_default_rate": 0.2067,
                            "calibration_intercept": -0.0005,
                            "calibration_slope": 0.9996,
                        },
                        "oot": {
                            "auc": 0.6973,
                            "pr_auc": 0.2631,
                            "gini": 0.3947,
                            "ks": 0.2939,
                            "brier": 0.1213,
                            "top10pct_default_capture": 0.2092,
                            "mean_pd": 0.1906,
                            "observed_default_rate": 0.1473,
                            "calibration_intercept": -0.5420,
                            "calibration_slope": 0.8370,
                        },
                    },
                    "challenger_lgbm": {
                        "train": {
                            "auc": 0.7318,
                            "pr_auc": 0.4155,
                            "gini": 0.4635,
                            "ks": 0.3368,
                            "brier": 0.1448,
                            "top10pct_default_capture": 0.2421,
                            "mean_pd": 0.2067,
                            "observed_default_rate": 0.2067,
                            "calibration_intercept": 0.0915,
                            "calibration_slope": 1.0790,
                        },
                        "oot": {
                            "auc": 0.7262,
                            "pr_auc": 0.3038,
                            "gini": 0.4525,
                            "ks": 0.3372,
                            "brier": 0.1183,
                            "top10pct_default_capture": 0.2411,
                            "mean_pd": 0.1968,
                            "observed_default_rate": 0.1473,
                            "calibration_intercept": -0.4185,
                            "calibration_slope": 0.9739,
                        },
                    },
                },
            },
            "drift_thresholds": {
                "source_pages": [6],
                "stable_lt": 0.10,
                "moderate_gte": 0.10,
                "moderate_lt": 0.25,
                "material_gte": 0.25,
                "flag_mapping": {
                    "stable": "GREEN",
                    "moderate": "AMBER",
                    "material": "RED",
                },
            },
            "score_stability": {
                "source_pages": [8, 12, 13],
                "models": {
                    "champion_logit": {
                        "score_psi": 0.025877,
                        "score_ks_stat": 0.057605,
                        "score_ks_p": 0.0,
                    },
                    "challenger_lgbm": {
                        "score_psi": 0.015908,
                        "score_ks_stat": 0.044408,
                        "score_ks_p": 0.0,
                    },
                },
            },
            "sensitivity_reference": {
                "source_pages": [9],
                "baseline_mean_pd": 0.190617,
                "baseline_p95_pd": 0.417995,
                "most_sensitive_feature": "dti",
                "ranking_stability_finding": "Spearman correlation remains approximately 1 under tested shocks.",
            },
        },
    }


def _build_diagnostics_summary(metric_reference: dict[str, Any]) -> dict[str, Any]:
    champion_oot = metric_reference["metrics"]["performance_calibration"]["models"]["champion_logit"]["oot"]
    challenger_oot = metric_reference["metrics"]["performance_calibration"]["models"]["challenger_lgbm"]["oot"]
    return {
        "status": READY_STATUS,
        "mode": "diagnostics",
        "source_documents": STRUCTURED_SOURCE_DOCUMENTS["diagnostics"],
        "source_pages": {
            MODEL_REPORT_DOC_ID: [3, 4, 6, 7, 8, 12, 13, 14],
            SR11_DOC_ID: [3, 4],
        },
        "summary": "Champion OOT ranking remains usable, but calibration weakened and material drift requires monitoring.",
        "metrics": {
            "top_drift_features": [
                {"feature": "revol util", "feature_type": "numeric", "psi": 0.2902, "flag": "RED"},
                {"feature": "application type", "feature_type": "categorical", "psi": 0.2457, "flag": "AMBER"},
                {"feature": "int rate", "feature_type": "numeric", "psi": 0.1271, "flag": "AMBER"},
                {"feature": "revol bal", "feature_type": "numeric", "psi": 0.0881, "flag": "GREEN"},
                {"feature": "purpose", "feature_type": "categorical", "psi": 0.0783, "flag": "GREEN"},
                {"feature": "sub grade", "feature_type": "categorical", "psi": 0.0756, "flag": "GREEN"},
                {"feature": "funded amnt", "feature_type": "numeric", "psi": 0.0625, "flag": "GREEN"},
                {"feature": "loan amnt", "feature_type": "numeric", "psi": 0.0625, "flag": "GREEN"},
            ],
            "score_shift": {
                "finding": "Score shift is visible but not extreme, and the challenger remains more stable than the champion.",
                "champion_logit": {
                    "score_psi": 0.025877,
                    "score_ks_stat": 0.057605,
                },
                "challenger_lgbm": {
                    "score_psi": 0.015908,
                    "score_ks_stat": 0.044408,
                },
            },
            "calibration": {
                "finding": "Champion mean PD remains above observed OOT default rate and calibration slope weakens out of time.",
                "champion_oot": {
                    "mean_pd": champion_oot["mean_pd"],
                    "observed_default_rate": champion_oot["observed_default_rate"],
                    "calibration_intercept": champion_oot["calibration_intercept"],
                    "calibration_slope": champion_oot["calibration_slope"],
                },
                "challenger_oot": {
                    "mean_pd": challenger_oot["mean_pd"],
                    "observed_default_rate": challenger_oot["observed_default_rate"],
                    "calibration_intercept": challenger_oot["calibration_intercept"],
                    "calibration_slope": challenger_oot["calibration_slope"],
                },
            },
            "reference_metrics": {
                "champion_oot_auc": champion_oot["auc"],
                "champion_oot_brier": champion_oot["brier"],
                "challenger_oot_auc": challenger_oot["auc"],
                "challenger_oot_brier": challenger_oot["brier"],
                "psi_thresholds": metric_reference["metrics"]["drift_thresholds"],
            },
        },
        "quality_flags": [
            "champion_calibration_degraded_oot",
            "material_input_drift_detected",
            "challenger_outperforms_champion_oot",
        ],
        "assumptions": [
            "Validation findings are interpreted as report-derived snapshots and should be paired with ongoing monitoring.",
            "Independent challenge and documentation remain part of proper model-risk use per SR11-7.",
        ],
        "limitations": [
            "Diagnostics are derived from report-level summaries rather than raw snapshot parquet inputs.",
            "The artifact captures only the frozen three-document MVP evidence boundary and does not replace ongoing validation governance.",
        ],
    }


def _build_scenario_config() -> dict[str, Any]:
    return {
        "status": READY_STATUS,
        "scenario_order": ["baseline", "mild", "severe"],
        "monotonicity_expectation": {
            "direction": "non_decreasing",
            "metrics": ["mean_pd", "p95_pd", "el_proxy_mean"],
        },
        "challenge_requirements": [
            "Document scenario narratives and affected drivers.",
            "Keep stresses severe but plausible and subject assumptions to challenge.",
            "Review results regularly as part of governance and decision support.",
        ],
        "scenarios": {
            "baseline": {
                "label": "Baseline OOT",
                "narrative": "Unstressed out-of-time observations used as the report reference point.",
                "severity_rank": 0,
                "affected_drivers": [],
                "shock_descriptions": [],
                "expected_direction": "reference",
                "source_documents": [MODEL_REPORT_DOC_ID],
                "source_pages": [9, 10],
            },
            "mild": {
                "label": "Mild recession",
                "narrative": "Lower income and higher DTI applied to OOT observations.",
                "severity_rank": 1,
                "affected_drivers": ["annual inc", "dti"],
                "shock_descriptions": ["lower income", "higher DTI"],
                "expected_direction": "increase_risk",
                "source_documents": [MODEL_REPORT_DOC_ID, BASEL_DOC_ID],
                "source_pages": [10, 7, 10, 13, 14],
            },
            "severe": {
                "label": "Severe recession",
                "narrative": "Larger income reduction, larger DTI increase, and higher utilization capped at stressed bounds.",
                "severity_rank": 2,
                "affected_drivers": ["annual inc", "dti", "revol util"],
                "shock_descriptions": ["larger income reduction", "larger DTI increase", "higher utilization capped"],
                "expected_direction": "increase_risk",
                "source_documents": [MODEL_REPORT_DOC_ID, BASEL_DOC_ID],
                "source_pages": [10, 7, 10, 13, 14],
            },
        },
    }


def _build_stress_summary() -> dict[str, Any]:
    baseline_el_proxy = round(647.706794 - 13.718138, 6)
    baseline = {
        "mean_pd": 0.190617,
        "p95_pd": 0.417995,
        "el_proxy_mean": baseline_el_proxy,
    }
    mild = {
        "mean_pd": 0.195145,
        "p95_pd": 0.427175,
        "el_proxy_mean": 647.706794,
        "delta_mean_pd": 0.004528,
        "delta_p95_pd": 0.00918,
        "delta_el_proxy": 13.718138,
        "lgd_avg": 0.979338,
    }
    severe = {
        "mean_pd": 0.200342,
        "p95_pd": 0.436968,
        "el_proxy_mean": 663.375656,
        "delta_mean_pd": 0.009725,
        "delta_p95_pd": 0.018973,
        "delta_el_proxy": 29.387,
        "lgd_avg": 0.979338,
    }
    return {
        "status": READY_STATUS,
        "mode": "stress",
        "source_documents": STRUCTURED_SOURCE_DOCUMENTS["stress"],
        "source_pages": {
            MODEL_REPORT_DOC_ID: [9, 10, 11],
            BASEL_DOC_ID: [5, 7, 10, 13, 14],
        },
        "summary": "Mild and severe scenarios increase mean PD, tail PD, and EL proxy monotonically relative to baseline.",
        "metrics": {
            "baseline": baseline,
            "mild": mild,
            "severe": severe,
            "delta_mean_pd": {
                "mild": mild["delta_mean_pd"],
                "severe": severe["delta_mean_pd"],
            },
            "delta_tail_pd": {
                "mild": mild["delta_p95_pd"],
                "severe": severe["delta_p95_pd"],
            },
            "delta_el_proxy": {
                "mild": mild["delta_el_proxy"],
                "severe": severe["delta_el_proxy"],
            },
            "monotonicity_passed": (
                baseline["mean_pd"] <= mild["mean_pd"] <= severe["mean_pd"]
                and baseline["p95_pd"] <= mild["p95_pd"] <= severe["p95_pd"]
                and baseline["el_proxy_mean"] <= mild["el_proxy_mean"] <= severe["el_proxy_mean"]
            ),
        },
        "quality_flags": [
            "stress_results_directional_only",
            "el_proxy_uses_proxy_lgd_and_ead",
            "scenario_escalation_is_monotonic",
        ],
        "assumptions": [
            "Scenario narratives remain stylized report overlays rather than full macroeconomic paths.",
            "Stress results should be challenged and revisited under governance review in line with Basel principles.",
        ],
        "limitations": [
            "EL proxy uses report-level PD, LGD, and EAD proxies and should not be treated as a production capital model output.",
            "Only baseline, mild, and severe cases from the frozen report are represented in the MVP artifact set.",
        ],
    }


def _build_data_manifest(
    metric_reference: dict[str, Any],
    scenario_config: dict[str, Any],
    diagnostics_summary: dict[str, Any],
    stress_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": READY_STATUS,
        "artifacts": {
            "metric_reference": {
                "path": "data/structured/metric_reference.json",
                "format": "json",
                "record_count": len(metric_reference["metrics"]),
                "source_documents": metric_reference["source_documents"],
                "source_pages": sorted(
                    {
                        page
                        for section in metric_reference["metrics"].values()
                        for page in section["source_pages"]
                    }
                ),
                "derivation_policy": "direct_extract_or_simple_derivation",
            },
            "scenario_config": {
                "path": "data/structured/scenario_config.yaml",
                "format": "yaml",
                "record_count": len(scenario_config["scenarios"]),
                "source_documents": sorted(
                    {
                        document_id
                        for scenario in scenario_config["scenarios"].values()
                        for document_id in scenario["source_documents"]
                    }
                ),
                "source_pages": sorted(
                    {
                        page
                        for scenario in scenario_config["scenarios"].values()
                        for page in scenario["source_pages"]
                    }
                ),
                "derivation_policy": "direct_extract_or_simple_derivation",
            },
            "diagnostics_summary": {
                "path": "data/structured/diagnostics_summary.json",
                "format": "json",
                "record_count": len(diagnostics_summary["metrics"]["top_drift_features"]),
                "source_documents": diagnostics_summary["source_documents"],
                "source_pages": sorted(
                    {
                        page
                        for pages in diagnostics_summary["source_pages"].values()
                        for page in pages
                    }
                ),
                "derivation_policy": "direct_extract_or_simple_derivation",
            },
            "stress_summary": {
                "path": "data/structured/stress_summary.json",
                "format": "json",
                "record_count": 3,
                "source_documents": stress_summary["source_documents"],
                "source_pages": sorted(
                    {
                        page
                        for pages in stress_summary["source_pages"].values()
                        for page in pages
                    }
                ),
                "derivation_policy": "direct_extract_or_simple_derivation",
            },
        },
    }


def _extract_toc_entries(pdf: fitz.Document) -> list[TocEntry]:
    return [
        TocEntry(level=level, title=_normalize_whitespace(title), page=page)
        for level, title, page in pdf.get_toc(simple=True)
        if _normalize_whitespace(title)
    ]


def _clean_page_lines(doc_id: str, raw_text: str, page_number: int) -> list[str]:
    normalized_text = raw_text.replace("\xad", "-")
    raw_lines = [_normalize_whitespace(line) for line in normalized_text.splitlines()]
    cleaned: list[str] = []
    for line in raw_lines:
        if not line:
            continue
        if line == str(page_number):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*\.?", line):
            continue
        if re.fullmatch(r"(?i)[ivxlcdm]+", line):
            continue
        if re.fullmatch(r"Page \d+ of \d+", line):
            continue
        if line in LINE_DROP_EXACT.get(doc_id, ()) and page_number > 1:
            continue
        cleaned.append(line)
    return _merge_hyphenated_lines(cleaned)


def _merge_hyphenated_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if merged and merged[-1].endswith("-") and line and line[0].islower():
            merged[-1] = merged[-1][:-1] + line
            continue
        merged.append(line)
    return merged


def _split_page_into_sections(
    doc_id: str,
    page_number: int,
    lines: list[str],
    toc_entries: list[TocEntry],
) -> list[dict[str, list[str]]]:
    page_entries = [entry for entry in toc_entries if entry.page == page_number]
    active_path = _page_default_path(doc_id, page_number) or _starting_toc_path(toc_entries, page_number)
    sections: list[dict[str, list[str]]] = []
    buffer: list[str] = []
    pending_heading_lines: list[str] = []
    page_entry_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, pending_heading_lines
        if not buffer and not pending_heading_lines:
            return
        payload = buffer or pending_heading_lines
        sections.append(
            {
                "path": active_path or ["Unmapped Section"],
                "lines": payload.copy(),
            }
        )
        buffer = []
        pending_heading_lines = []

    for line in lines:
        override_path = _line_override_path(doc_id, line)
        if override_path is not None:
            if buffer:
                flush_buffer()
            active_path = override_path
            pending_heading_lines.append(line)
            continue

        if page_entry_index < len(page_entries) and _normalize_heading(line) == _normalize_heading(page_entries[page_entry_index].title):
            if buffer:
                flush_buffer()
            active_path = _apply_toc_entry(active_path, page_entries[page_entry_index])
            pending_heading_lines.append(line)
            page_entry_index += 1
            continue

        if pending_heading_lines and not buffer:
            buffer.extend(pending_heading_lines)
            pending_heading_lines = []
        buffer.append(line)

    flush_buffer()
    return [section for section in sections if any(line.strip() for line in section["lines"])]


def _page_default_path(doc_id: str, page_number: int) -> list[str]:
    return DOC_PAGE_DEFAULT_PATHS.get(doc_id, {}).get(page_number, []).copy()


def _starting_toc_path(toc_entries: list[TocEntry], page_number: int) -> list[str]:
    active: list[str] = []
    for entry in toc_entries:
        if entry.page >= page_number:
            break
        active = _apply_toc_entry(active, entry)
    return active


def _apply_toc_entry(active_path: list[str], entry: TocEntry) -> list[str]:
    next_path = active_path[: max(entry.level - 1, 0)]
    next_path.append(entry.title)
    return next_path


def _line_override_path(doc_id: str, line: str) -> list[str] | None:
    overrides = LINE_SECTION_OVERRIDES.get(doc_id, {})
    return overrides.get(_normalize_heading(line), None)


def _chunk_section_lines(lines: list[str]) -> list[str]:
    paragraphs = _build_paragraphs(lines)
    if not paragraphs:
        return []

    expanded: list[str] = []
    for paragraph in paragraphs:
        expanded.extend(_split_long_paragraph(paragraph, TARGET_WORD_RANGE[1]))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for paragraph in expanded:
        paragraph_words = _word_count(paragraph)
        if current and current_words + paragraph_words > TARGET_WORD_RANGE[1]:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
            current_words = paragraph_words
            continue
        current.append(paragraph)
        current_words += paragraph_words

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _build_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current_text: list[str] = []
    table_mode = any(_looks_table_like(line) for line in lines)

    for line in lines:
        if table_mode or _is_list_item(line):
            if current_text:
                paragraphs.append(" ".join(current_text).strip())
                current_text = []
            paragraphs.append(line)
            continue
        current_text.append(line)

    if current_text:
        paragraphs.append(" ".join(current_text).strip())
    return [paragraph for paragraph in paragraphs if paragraph]


def _split_long_paragraph(text: str, max_words: int) -> list[str]:
    if _word_count(text) <= max_words:
        return [text]

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if len(sentences) == 1:
        words = text.split()
        return [
            " ".join(words[index : index + max_words]).strip()
            for index in range(0, len(words), max_words)
        ]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_words = sentence_words
            continue
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _classify_chunk_type(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_count = sum(1 for line in lines if _is_list_item(line))
    if bullet_count >= 2:
        return CHUNK_TYPE_BULLETS
    if any(_looks_table_like(line) for line in lines):
        return CHUNK_TYPE_TABLE
    return CHUNK_TYPE_TEXT


def _derive_topic_tags(section_path: list[str], text: str) -> list[str]:
    haystack = " ".join(section_path + [text]).lower()
    matched = [
        topic
        for topic, patterns in TOPIC_TAG_PATTERNS.items()
        if any(pattern in haystack for pattern in patterns)
    ]
    return sorted(matched)


def _looks_table_like(line: str) -> bool:
    if line.startswith("Table ") or line.startswith("Part "):
        return True
    tokens = line.split()
    if not tokens:
        return False
    numeric_tokens = sum(1 for token in tokens if re.search(r"\d", token))
    return numeric_tokens >= 3 and numeric_tokens >= max(2, len(tokens) // 2)


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^(?:[-*•]|\d+\.)\s*", line))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_heading(text: str) -> str:
    return _normalize_whitespace(text).lower()


def _word_count(text: str) -> int:
    return len(text.split())


def _slugify(value: str) -> str:
    return value.lower()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    serialized = "\n".join(json.dumps(record, sort_keys=False) for record in records)
    path.write_text(serialized + ("\n" if serialized else ""), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
