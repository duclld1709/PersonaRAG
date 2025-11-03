"""Embedding client implementations."""

from .huggingface_embedding import HuggingFaceEmbeddingClient
from .openai_embedding import OpenAIEmbeddingClient

__all__ = ["OpenAIEmbeddingClient", "HuggingFaceEmbeddingClient"]