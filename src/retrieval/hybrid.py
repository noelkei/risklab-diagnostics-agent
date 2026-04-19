from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from schemas import RetrievalChunk

from .loader import LoadedRetrievalCorpus, load_retrieval_corpus, tokenize_text


DEFAULT_DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BM25_TOP_K = 6
DEFAULT_DENSE_TOP_K = 6
DEFAULT_FINAL_TOP_K = 4
_RRF_OFFSET = 10


class EmbeddingBackend(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        normalize_embeddings: bool = False,
    ) -> object: ...


@dataclass(frozen=True)
class _RankedCandidate:
    index: int
    rank: int
    score: float


def collect_retrieved_doc_ids(chunks: Sequence[RetrievalChunk]) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        doc_ids.append(chunk.doc_id)
    return doc_ids


def _load_default_embedding_backend(model_name: str) -> EmbeddingBackend:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class HybridRetriever:
    def __init__(
        self,
        *,
        corpus: LoadedRetrievalCorpus,
        embedding_backend: EmbeddingBackend,
        dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    ) -> None:
        self._chunks = corpus.chunks
        self._dense_texts = corpus.dense_texts
        self._dense_model_name = dense_model_name
        self._embedding_backend = embedding_backend
        self._bm25 = BM25Okapi([tokenize_text(text) for text in corpus.sparse_texts])
        self._dense_embeddings = self._encode_texts(corpus.dense_texts)

    @classmethod
    def from_artifacts(
        cls,
        *,
        chunks_path: Path | None = None,
        corpus_manifest_path: Path | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        dense_model_name: str = DEFAULT_DENSE_MODEL_NAME,
    ) -> "HybridRetriever":
        corpus = load_retrieval_corpus(
            chunks_path=chunks_path,
            corpus_manifest_path=corpus_manifest_path,
        )
        resolved_backend = embedding_backend or _load_default_embedding_backend(dense_model_name)
        return cls(
            corpus=corpus,
            embedding_backend=resolved_backend,
            dense_model_name=dense_model_name,
        )

    @property
    def dense_model_name(self) -> str:
        return self._dense_model_name

    def search(
        self,
        query: str,
        *,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> list[RetrievalChunk]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            return []
        if bm25_top_k < 1 or dense_top_k < 1 or final_top_k < 1:
            raise ValueError("Retrieval top-k values must be positive integers.")

        sparse_candidates = self._rank_sparse(query, bm25_top_k)
        dense_candidates = self._rank_dense(query, dense_top_k)
        return self._fuse_candidates(
            sparse_candidates=sparse_candidates,
            dense_candidates=dense_candidates,
            final_top_k=final_top_k,
        )

    def _rank_sparse(self, query: str, top_k: int) -> list[_RankedCandidate]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self._chunks[index].chunk_id),
        )[:top_k]
        return [
            _RankedCandidate(index=index, rank=rank, score=float(scores[index]))
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    def _rank_dense(self, query: str, top_k: int) -> list[_RankedCandidate]:
        query_embedding = self._encode_texts([query])[0]
        similarities = self._dense_embeddings @ query_embedding
        ranked_indices = sorted(
            range(len(similarities)),
            key=lambda index: (-float(similarities[index]), self._chunks[index].chunk_id),
        )[:top_k]
        return [
            _RankedCandidate(index=index, rank=rank, score=float(similarities[index]))
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    def _fuse_candidates(
        self,
        *,
        sparse_candidates: Sequence[_RankedCandidate],
        dense_candidates: Sequence[_RankedCandidate],
        final_top_k: int,
    ) -> list[RetrievalChunk]:
        fused_scores: dict[int, dict[str, float | None]] = {}

        for candidate in sparse_candidates:
            candidate_scores = fused_scores.setdefault(
                candidate.index,
                {"sparse_score": None, "dense_score": None, "fused_score": 0.0},
            )
            candidate_scores["sparse_score"] = candidate.score
            candidate_scores["fused_score"] = float(candidate_scores["fused_score"] or 0.0) + (
                1.0 / (_RRF_OFFSET + candidate.rank)
            )

        for candidate in dense_candidates:
            candidate_scores = fused_scores.setdefault(
                candidate.index,
                {"sparse_score": None, "dense_score": None, "fused_score": 0.0},
            )
            candidate_scores["dense_score"] = candidate.score
            candidate_scores["fused_score"] = float(candidate_scores["fused_score"] or 0.0) + (
                1.0 / (_RRF_OFFSET + candidate.rank)
            )

        ranked_chunks = [
            self._chunks[index].model_copy(
                update={
                    "sparse_score": candidate_scores["sparse_score"],
                    "dense_score": candidate_scores["dense_score"],
                    "fused_score": float(candidate_scores["fused_score"] or 0.0),
                }
            )
            for index, candidate_scores in fused_scores.items()
        ]
        ranked_chunks.sort(
            key=lambda chunk: (
                -(chunk.fused_score or 0.0),
                -self._sortable_score(chunk.sparse_score),
                -self._sortable_score(chunk.dense_score),
                chunk.chunk_id,
            )
        )
        return ranked_chunks[:final_top_k]

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        encoded: object
        try:
            encoded = self._embedding_backend.encode(
                list(texts),
                normalize_embeddings=True,
            )
        except TypeError:
            encoded = self._embedding_backend.encode(list(texts))

        embeddings = np.asarray(encoded, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms

    @staticmethod
    def _sortable_score(score: float | None) -> float:
        if score is None:
            return float("-inf")
        return float(score)
