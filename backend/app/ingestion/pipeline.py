"""Document ingestion pipeline orchestration."""

from __future__ import annotations

from hashlib import sha1
from logging import getLogger
from typing import Any, Mapping
from uuid import UUID

from langchain_core.documents import Document

from app.core.config import Settings
from app.domain.interfaces.vector_store import VectorStore
from app.domain.interfaces.ingestion import DocumentIndexer

from .splitters import split_documents

logger = getLogger(__name__)

try:
    from langsmith.run_helpers import traceable
except ImportError:  # pragma: no cover - optional dependency
    def traceable(*dargs: Any, **dkwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return decorator


def _metadata_equivalent(existing: Mapping[str, Any] | None, new: Mapping[str, Any]) -> bool:
    if existing is None:
        return False

    excluded_keys = {"chunk_checksum"}

    def _normalize(meta: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in meta.items() if key not in excluded_keys}

    return _normalize(existing) == _normalize(new)


class DocumentIngestionPipeline(DocumentIndexer):
    """Pipeline responsible for loading, splitting, and indexing documents."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @traceable(name="pipeline.index_document", run_type="chain")
    def index_document(
        self,
        *,
        document_id: UUID,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        run_tree: Any | None = None,
    ) -> int:
        """Split the supplied content, replace existing chunks and persist."""

        base_metadata: dict[str, Any] = {
            "document_id": str(document_id),
        }
        if metadata:
            base_metadata.update(metadata)

        base_document = Document(page_content=content, metadata=base_metadata.copy())
        logger.info("Splitting document %s into chunks", document_id)
        raw_chunks = split_documents(
            [base_document],
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        existing_chunks = {
            chunk.metadata.get("chunk_id"): chunk
            for chunk in self._vector_store.get(filter={"document_id": base_metadata["document_id"]})
            if chunk.metadata.get("chunk_id")
        }

        chunks: list[Document] = []
        for index, chunk in enumerate(raw_chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update(chunk.metadata or {})
            chunk_metadata["chunk_index"] = index
            chunk_metadata["chunk_id"] = f"{document_id.hex}:{index}"
            chunk_metadata["chunk_checksum"] = sha1(chunk.page_content.encode("utf-8")).hexdigest()
            chunks.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))

        new_chunks = {chunk.metadata["chunk_id"]: chunk for chunk in chunks}

        ids_to_delete: set[str] = set()
        chunks_to_add: list[Document] = []

        for chunk_id, new_chunk in new_chunks.items():
            existing_chunk = existing_chunks.get(chunk_id)
            if not existing_chunk:
                chunks_to_add.append(new_chunk)
                continue

            existing_checksum = existing_chunk.metadata.get("chunk_checksum")
            new_checksum = new_chunk.metadata.get("chunk_checksum")
            if existing_checksum == new_checksum and _metadata_equivalent(existing_chunk.metadata, new_chunk.metadata):
                continue
            ids_to_delete.add(chunk_id)
            chunks_to_add.append(new_chunk)

        for chunk_id in existing_chunks.keys():
            if chunk_id not in new_chunks:
                ids_to_delete.add(chunk_id)

        if ids_to_delete:
            self._vector_store.delete(ids=list(ids_to_delete))

        if chunks_to_add:
            self._vector_store.add_documents(chunks_to_add)

        logger.info(
            "Indexed document %s: %d added, %d removed, %d unchanged",
            document_id,
            len(chunks_to_add),
            len(ids_to_delete),
            len(new_chunks) - len(chunks_to_add),
        )

        if run_tree and hasattr(run_tree, "add_metadata"):
            try:
                run_tree.add_metadata(
                    {
                        "document_id": str(document_id),
                        "chunk_count": len(new_chunks),
                        "chunks_added": len(chunks_to_add),
                        "chunks_deleted": len(ids_to_delete),
                    }
                )
            except Exception:  # pragma: no cover - best effort
                pass
        return len(new_chunks)

    @traceable(name="pipeline.delete_document", run_type="chain")
    def delete_document(self, document_id: UUID, *, run_tree: Any | None = None) -> None:
        """Remove all chunks associated with the document from the vector store."""

        self._vector_store.delete(filter={"document_id": str(document_id)})
        if run_tree and hasattr(run_tree, "add_metadata"):
            try:
                run_tree.add_metadata({"document_id": str(document_id)})
            except Exception:  # pragma: no cover - best effort
                pass

    def reset(self) -> None:
        """Clear the vector store."""

        self._vector_store.reset()

    def count(self) -> int:
        """Return number of chunks in the vector store."""

        return self._vector_store.count()
