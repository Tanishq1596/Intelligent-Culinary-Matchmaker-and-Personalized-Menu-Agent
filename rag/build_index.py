"""Rebuild the culinary ChromaDB index."""

from rag.rag_pipeline import build_vector_index


print(build_vector_index(rebuild=True))
