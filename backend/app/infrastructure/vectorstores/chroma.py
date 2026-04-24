"""Chroma vector store adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from chromadb.errors import InvalidArgumentError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from app.domain.interfaces.embedding_client import EmbeddingClient

from app.domain.interfaces.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Lightweight wrapper around LangChain's Chroma integration."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: EmbeddingClient,
        *,
        collection_name: str = "rag_documents",
    ) -> None:
        self._persist_directory = persist_directory
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._collection_metadata = {"hnsw:space": "cosine"}
        self._store = self._create_store()

    def _create_store(self) -> Chroma:
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        return Chroma(
            persist_directory=str(self._persist_directory),
            embedding_function=self._embedding_model,
            collection_name=self._collection_name,
            collection_metadata=self._collection_metadata,
        )

    def add_documents(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        ids: list[str] = []
        has_ids = True
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                ids.append(chunk_id)
            else:
                has_ids = False
                break

        def _insert() -> None:
            if has_ids:
                self._store.add_documents(docs, ids=ids)
            else:
                self._store.add_documents(docs)

        try:
            _insert()
        except InvalidArgumentError as exc:
            if "dimension" not in str(exc).lower():
                raise
            logger.warning(
                "Chroma collection dimension mismatch (%s); resetting collection "
                "at %s and retrying. Existing vectors were from a different "
                "embedding model and cannot be reused.",
                exc,
                self._persist_directory,
            )
            self.reset()
            _insert()

    def delete(self, *, ids: list[str] | None = None, filter: dict | None = None) -> None:
        self._store.delete(ids=ids, where=filter)

    def get(self, *, filter: dict | None = None) -> Sequence[Document]:
        raw = self._store._collection.get(  # type: ignore[attr-defined]  # pragma: no cover
            where=filter,
            include=["documents", "metadatas"],
        )
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        ids = raw.get("ids") or []

        results: list[Document] = []
        for chunk_id, metadata, content in zip(ids, metadatas, documents):
            metadata = metadata or {}
            if "chunk_id" not in metadata and chunk_id:
                metadata = {**metadata, "chunk_id": chunk_id}
            results.append(Document(page_content=content, metadata=metadata))
        return results

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        return self._store.as_retriever(**kwargs)

    def reset(self) -> None:
        self._store.delete_collection()
        self._store = self._create_store()

    def count(self) -> int:
        return self._store._collection.count()  # type: ignore[attr-defined]  # pragma: no cover
