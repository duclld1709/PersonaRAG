"""Document ingestion pipeline orchestration."""

from __future__ import annotations

from hashlib import sha1
from logging import getLogger
from typing import Any, Callable, Mapping
from uuid import UUID

from langchain_core.documents import Document

from app.core.config import Settings
from app.domain.interfaces.ingestion import DocumentIndexer
from app.domain.interfaces.vector_store import VectorStore

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
    """Load, split, and index documents; also notify a BM25 rebuild hook."""

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        *,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._on_change = on_change

    def set_on_change(self, callback: Callable[[], None]) -> None:
        self._on_change = callback

    @traceable(name="pipeline.index_document", run_type="chain")
    def index_document(
        self,
        *,
        document_id: UUID,
        content: str,
        metadata: Mapping[str, Any] | None = None,
        run_tree: Any | None = None,
    ) -> int:
        """Split the supplied content, diff against existing chunks, and persist."""

        base_metadata: dict[str, Any] = {"document_id": str(document_id)}
        if metadata:
            base_metadata.update(metadata)

        base_document = Document(page_content=content, metadata=base_metadata.copy())
        logger.info("Splitting document %s into chunks", document_id)
        raw_chunks = split_documents(
            [base_document],
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
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

        if self._on_change and (chunks_to_add or ids_to_delete):
            try:
                self._on_change()
            except Exception as exc:  # noqa: BLE001 — must not break ingestion
                logger.warning("on_change hook failed: %s", exc)

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
        self._vector_store.delete(filter={"document_id": str(document_id)})
        if self._on_change:
            try:
                self._on_change()
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_change hook failed: %s", exc)
        if run_tree and hasattr(run_tree, "add_metadata"):
            try:
                run_tree.add_metadata({"document_id": str(document_id)})
            except Exception:  # pragma: no cover - best effort
                pass

    def reset(self) -> None:
        self._vector_store.reset()
        if self._on_change:
            try:
                self._on_change()
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_change hook failed: %s", exc)

    def count(self) -> int:
        return self._vector_store.count()
