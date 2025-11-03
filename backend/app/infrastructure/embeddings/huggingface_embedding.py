"""HugggingFace embeddings adaper."""


from typing import Iterable, Sequence
from backend.app.domain.interfaces.embedding_client import EmbeddingClient

from langchain_community.embeddings import HuggingFaceEmbeddings


class HuggingFaceEmbeddingClient(EmbeddingClient):
    """Adapter that wraps LangChain's HuggingFaceEmbeddings."""

    def __init__(self, model_name: str, api_token: str | None = None) -> None:
        init_kwargs = {
            "model_name": model_name
        }
        if api_token:
            init_kwargs["huggingfacehub_api_token"] = api_token
        self._client = HuggingFaceEmbeddings(**init_kwargs)
    
    def embed_documents(self, texts: Iterable[str]) -> Sequence[list[float]]:
        return self._client.embed_documents(list(texts))
    
    def embed_query(self, text):
        return self._client.embed_query(text)