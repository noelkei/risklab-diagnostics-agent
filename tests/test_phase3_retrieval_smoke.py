from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from retrieval import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    HybridRetriever,
    RetrievalLoadError,
    collect_retrieved_doc_ids,
    load_retrieval_corpus,
)
from app.config import load_settings


class KeywordEmbeddingBackend:
    def __init__(self) -> None:
        self._features = (
            "sr11",
            "governance",
            "model",
            "validation",
            "psi",
            "drift",
            "feature",
            "basel",
            "stress",
            "testing",
        )

    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        matrix = []
        for sentence in sentences:
            lowered = sentence.lower().replace("-", " ")
            counts = [float(lowered.count(feature)) for feature in self._features]
            counts.append(float(len(lowered.split())))
            matrix.append(counts)

        embeddings = np.asarray(matrix, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            embeddings = embeddings / norms
        return embeddings


def _ready_manifest() -> dict[str, object]:
    settings = load_settings()
    documents = [
        {
            "doc_id": document.doc_id,
            "path": f"docs/domain/{document.filename}",
            "page_count": 1,
            "toc_used": True,
            "chunk_count": 1,
            "authority_level": document.authority_level,
            "document_role": document.document_role,
        }
        for document in settings.frozen_corpus
    ]
    return {
        "status": "ready",
        "document_count": len(documents),
        "chunk_count": 1,
        "documents": documents,
    }


def _write_ready_manifest(path: Path) -> None:
    path.write_text(json.dumps(_ready_manifest()), encoding="utf-8")


def test_phase3_loader_reads_ready_corpus_and_builds_search_views() -> None:
    corpus = load_retrieval_corpus()

    assert len(corpus.chunks) == 72
    assert len(corpus.sparse_texts) == len(corpus.chunks)
    assert len(corpus.dense_texts) == len(corpus.chunks)
    assert "model validation report" in corpus.sparse_texts[0]
    assert corpus.chunks[0].section_path in corpus.dense_texts[0]


def test_phase3_hybrid_retrieval_is_deterministic_and_citation_ready() -> None:
    retriever = HybridRetriever.from_artifacts(embedding_backend=KeywordEmbeddingBackend())

    first_results = retriever.search("PSI drift features")
    second_results = retriever.search("PSI drift features")

    assert len(first_results) <= DEFAULT_FINAL_TOP_K
    assert [chunk.chunk_id for chunk in first_results] == [chunk.chunk_id for chunk in second_results]
    assert len({chunk.chunk_id for chunk in first_results}) == len(first_results)
    assert any(chunk.sparse_score is not None and chunk.dense_score is not None for chunk in first_results)
    assert collect_retrieved_doc_ids(first_results) == collect_retrieved_doc_ids(second_results)

    for chunk in first_results:
        assert chunk.text.strip()
        assert chunk.doc_id
        assert chunk.page >= 1
        assert chunk.section_path
        assert chunk.chunk_type
        assert chunk.authority_level
        assert chunk.document_role
        assert chunk.sparse_score is not None or chunk.dense_score is not None
        assert chunk.fused_score is not None


def test_phase3_blank_query_returns_empty_results() -> None:
    retriever = HybridRetriever.from_artifacts(embedding_backend=KeywordEmbeddingBackend())

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_phase3_low_signal_query_returns_safely_and_deterministically() -> None:
    retriever = HybridRetriever.from_artifacts(embedding_backend=KeywordEmbeddingBackend())

    first_results = retriever.search("qxjzv plasma zebra quartz")
    second_results = retriever.search("qxjzv plasma zebra quartz")

    assert len(first_results) <= DEFAULT_FINAL_TOP_K
    assert [chunk.chunk_id for chunk in first_results] == [chunk.chunk_id for chunk in second_results]
    for chunk in first_results:
        assert chunk.fused_score is not None
        assert chunk.sparse_score is not None or chunk.dense_score is not None


def test_phase3_search_rejects_non_string_query() -> None:
    retriever = HybridRetriever.from_artifacts(embedding_backend=KeywordEmbeddingBackend())

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(None)  # type: ignore[arg-type]


def test_phase3_corpus_backed_queries_surface_expected_documents() -> None:
    retriever = HybridRetriever.from_artifacts(embedding_backend=KeywordEmbeddingBackend())
    scenarios = [
        ("SR11-7 model validation governance", "SR11_7_Model_Risk_Management"),
        ("PSI drift features", "Model_Validation_Report"),
        ("Basel stress testing governance", "Basel_Stress_Testing_Principles_2018"),
    ]

    for query, expected_doc_id in scenarios:
        results = retriever.search(
            query,
            bm25_top_k=DEFAULT_BM25_TOP_K,
            dense_top_k=DEFAULT_DENSE_TOP_K,
            final_top_k=DEFAULT_FINAL_TOP_K,
        )
        assert len(results) <= DEFAULT_FINAL_TOP_K
        assert expected_doc_id in collect_retrieved_doc_ids(results)


def test_phase3_loader_fails_fast_for_missing_chunks_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    _write_ready_manifest(manifest_path)

    with pytest.raises(RetrievalLoadError, match="Missing retrieval corpus artifact"):
        load_retrieval_corpus(
            chunks_path=tmp_path / "missing_chunks.jsonl",
            corpus_manifest_path=manifest_path,
        )


def test_phase3_loader_fails_fast_for_invalid_jsonl(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_ready_manifest(manifest_path)
    chunks_path.write_text("{not-valid-json}\n", encoding="utf-8")

    with pytest.raises(RetrievalLoadError, match="Invalid chunk JSON on line 1"):
        load_retrieval_corpus(
            chunks_path=chunks_path,
            corpus_manifest_path=manifest_path,
        )


def test_phase3_loader_fails_fast_for_missing_required_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_ready_manifest(manifest_path)
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk:test:p1:1",
                "text": "Missing required metadata fields.",
                "doc_id": "Model_Validation_Report",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalLoadError, match="Invalid chunk record on line 1"):
        load_retrieval_corpus(
            chunks_path=chunks_path,
            corpus_manifest_path=manifest_path,
        )


def test_phase3_loader_rejects_empty_corpus(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.json"
    chunks_path = tmp_path / "chunks.jsonl"
    manifest = _ready_manifest()
    manifest["chunk_count"] = 0
    for document in manifest["documents"]:
        document["chunk_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    chunks_path.write_text("", encoding="utf-8")

    with pytest.raises(RetrievalLoadError, match="Retrieval corpus is empty"):
        load_retrieval_corpus(
            chunks_path=chunks_path,
            corpus_manifest_path=manifest_path,
        )
