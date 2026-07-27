"""Exact-first culinary retrieval with a persistent semantic fallback."""

from .rag_pipeline import CulinaryRAG, RetrievalResult, build_vector_index

__all__ = ["CulinaryRAG", "RetrievalResult", "build_vector_index"]
