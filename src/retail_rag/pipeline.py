"""Embedding helper shared by the application."""
from __future__ import annotations

from typing import Any


def build_e5_embeddings(model_name: str = "intfloat/multilingual-e5-small") -> Any:
    """HuggingFace embeddings adding the `query: ` / `passage: ` prefixes E5 models expect."""
    from langchain_huggingface import HuggingFaceEmbeddings

    class E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return super().embed_documents([f"passage: {text}" for text in texts])

        def embed_query(self, text: str) -> list[float]:
            return super().embed_query(f"query: {text}")

    return E5Embeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})
