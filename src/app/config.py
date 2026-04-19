from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    filename: str
    document_role: str
    authority_level: str

    @property
    def path(self) -> Path:
        return _repo_root() / "docs" / "domain" / self.filename


@dataclass(frozen=True)
class ProviderConfig:
    provider_name: str
    sdk_package: str
    api_key_env_var: str
    model_name_env_var: str
    api_key: str | None
    model_name: str
    allowed_llm_nodes: tuple[str, ...]
    disallowed_llm_uses: tuple[str, ...]

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    env_file_path: Path
    data_dir: Path
    corpus_dir: Path
    structured_dir: Path
    eval_dir: Path
    outputs_dir: Path
    traces_dir: Path
    eval_results_dir: Path
    corpus_manifest_path: Path
    chunks_path: Path
    data_manifest_path: Path
    metric_reference_path: Path
    scenario_config_path: Path
    diagnostics_summary_path: Path
    stress_summary_path: Path
    eval_questions_path: Path
    trace_runs_path: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "ProjectPaths":
        data_dir = repo_root / "data"
        corpus_dir = data_dir / "corpus"
        structured_dir = data_dir / "structured"
        eval_dir = data_dir / "eval"
        outputs_dir = repo_root / "outputs"
        traces_dir = outputs_dir / "traces"
        eval_results_dir = outputs_dir / "eval_results"
        return cls(
            repo_root=repo_root,
            env_file_path=repo_root / ".env",
            data_dir=data_dir,
            corpus_dir=corpus_dir,
            structured_dir=structured_dir,
            eval_dir=eval_dir,
            outputs_dir=outputs_dir,
            traces_dir=traces_dir,
            eval_results_dir=eval_results_dir,
            corpus_manifest_path=corpus_dir / "corpus_manifest.json",
            chunks_path=corpus_dir / "chunks.jsonl",
            data_manifest_path=structured_dir / "data_manifest.json",
            metric_reference_path=structured_dir / "metric_reference.json",
            scenario_config_path=structured_dir / "scenario_config.yaml",
            diagnostics_summary_path=structured_dir / "diagnostics_summary.json",
            stress_summary_path=structured_dir / "stress_summary.json",
            eval_questions_path=eval_dir / "eval_questions.jsonl",
            trace_runs_path=traces_dir / "runs.jsonl",
        )


@dataclass(frozen=True)
class ProjectSettings:
    project_name: str
    provider: ProviderConfig
    paths: ProjectPaths
    frozen_corpus: tuple[CorpusDocument, ...]


def load_settings() -> ProjectSettings:
    repo_root = _repo_root()
    paths = ProjectPaths.from_repo_root(repo_root)
    _load_local_env_file(paths.env_file_path)
    provider = ProviderConfig(
        provider_name="gemini",
        sdk_package="google-genai",
        api_key_env_var="GEMINI_API_KEY",
        model_name_env_var="GEMINI_MODEL_NAME",
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name=os.getenv("GEMINI_MODEL_NAME", DEFAULT_GEMINI_MODEL_NAME),
        allowed_llm_nodes=("classify_query", "synthesize_answer"),
        disallowed_llm_uses=(
            "pdf_parsing",
            "retrieval",
            "quantitative_calculations",
            "core_abstention_logic",
            "trace_persistence",
            "eval_scoring",
        ),
    )
    frozen_corpus = (
        CorpusDocument(
            doc_id="Model_Validation_Report",
            filename="Model_Validation_Report.pdf",
            document_role="internal_case_primary_source",
            authority_level="internal",
        ),
        CorpusDocument(
            doc_id="SR11_7_Model_Risk_Management",
            filename="SR11_7_Model_Risk_Management.pdf",
            document_role="supervisory_guidance",
            authority_level="supervisory",
        ),
        CorpusDocument(
            doc_id="Basel_Stress_Testing_Principles_2018",
            filename="Basel_Stress_Testing_Principles_2018.pdf",
            document_role="regulatory_principles",
            authority_level="regulatory",
        ),
    )
    return ProjectSettings(
        project_name="risklab-diagnostics-agent",
        provider=provider,
        paths=paths,
        frozen_corpus=frozen_corpus,
    )
