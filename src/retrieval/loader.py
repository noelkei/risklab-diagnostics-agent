from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from app.config import load_settings
from schemas import RetrievalChunk


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_DOC_ID_SEPARATOR_PATTERN = re.compile(r"[_\-/]+")


class RetrievalLoadError(RuntimeError):
    """Raised when the frozen corpus artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class LoadedRetrievalCorpus:
    chunks: tuple[RetrievalChunk, ...]
    sparse_texts: tuple[str, ...]
    dense_texts: tuple[str, ...]


def normalize_doc_label(doc_id: str) -> str:
    normalized = _DOC_ID_SEPARATOR_PATTERN.sub(" ", doc_id).strip().lower()
    return " ".join(normalized.split())


def tokenize_text(text: str) -> list[str]:
    normalized = text.lower().replace(">", " ")
    return _TOKEN_PATTERN.findall(normalized)


def _build_sparse_text(chunk: RetrievalChunk) -> str:
    parts = [
        normalize_doc_label(chunk.doc_id),
        chunk.section_path,
        " ".join(chunk.topic_tags),
        chunk.text,
    ]
    return " ".join(part for part in parts if part)


def _build_dense_text(chunk: RetrievalChunk) -> str:
    parts = [
        chunk.section_path,
        " ".join(chunk.topic_tags),
        chunk.text,
    ]
    return " ".join(part for part in parts if part)


def load_retrieval_corpus(
    *,
    chunks_path: Path | None = None,
    corpus_manifest_path: Path | None = None,
) -> LoadedRetrievalCorpus:
    settings = load_settings()
    resolved_chunks_path = chunks_path or settings.paths.chunks_path
    resolved_manifest_path = corpus_manifest_path or settings.paths.corpus_manifest_path
    expected_doc_ids = {document.doc_id for document in settings.frozen_corpus}

    if not resolved_chunks_path.exists():
        raise RetrievalLoadError(f"Missing retrieval corpus artifact: {resolved_chunks_path}")
    if not resolved_manifest_path.exists():
        raise RetrievalLoadError(f"Missing corpus manifest artifact: {resolved_manifest_path}")

    try:
        manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RetrievalLoadError(
            f"Invalid corpus manifest JSON: {resolved_manifest_path}"
        ) from exc
    if manifest.get("status") != "ready":
        raise RetrievalLoadError("Corpus manifest is not ready for retrieval.")

    manifest_doc_ids = {document["doc_id"] for document in manifest.get("documents", [])}
    if manifest_doc_ids != expected_doc_ids:
        raise RetrievalLoadError(
            "Corpus manifest documents do not match the frozen corpus configuration."
        )

    raw_lines = [
        line
        for line in resolved_chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parsed_chunks: list[RetrievalChunk] = []
    for line_number, line in enumerate(raw_lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalLoadError(
                f"Invalid chunk JSON on line {line_number} in {resolved_chunks_path}"
            ) from exc

        try:
            parsed_chunks.append(RetrievalChunk.model_validate(payload))
        except ValidationError as exc:
            raise RetrievalLoadError(
                f"Invalid chunk record on line {line_number} in {resolved_chunks_path}"
            ) from exc

    chunks = tuple(parsed_chunks)
    if not chunks:
        raise RetrievalLoadError("Retrieval corpus is empty.")
    if len(chunks) != manifest.get("chunk_count"):
        raise RetrievalLoadError("Chunk count does not match the ready corpus manifest.")

    chunk_doc_ids = {chunk.doc_id for chunk in chunks}
    if chunk_doc_ids != expected_doc_ids:
        raise RetrievalLoadError("Chunk document IDs do not match the frozen corpus configuration.")

    sparse_texts = tuple(_build_sparse_text(chunk) for chunk in chunks)
    dense_texts = tuple(_build_dense_text(chunk) for chunk in chunks)
    return LoadedRetrievalCorpus(
        chunks=chunks,
        sparse_texts=sparse_texts,
        dense_texts=dense_texts,
    )
