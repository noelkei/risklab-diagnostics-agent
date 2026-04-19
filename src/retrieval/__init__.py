"""Hybrid retrieval utilities for the frozen MVP."""

from functools import lru_cache

from .hybrid import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_DENSE_MODEL_NAME,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    HybridRetriever,
    collect_retrieved_doc_ids,
)
from .loader import LoadedRetrievalCorpus, RetrievalLoadError, load_retrieval_corpus


@lru_cache(maxsize=1)
def get_default_retriever() -> HybridRetriever:
    return HybridRetriever.from_artifacts()


__all__ = [
    "DEFAULT_BM25_TOP_K",
    "DEFAULT_DENSE_MODEL_NAME",
    "DEFAULT_DENSE_TOP_K",
    "DEFAULT_FINAL_TOP_K",
    "HybridRetriever",
    "LoadedRetrievalCorpus",
    "RetrievalLoadError",
    "collect_retrieved_doc_ids",
    "get_default_retriever",
    "load_retrieval_corpus",
]
