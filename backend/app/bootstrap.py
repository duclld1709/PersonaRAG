"""Application bootstrap — wires all OpenAI-backed services together."""

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.core.config import Settings
from app.infrastructure.embeddings import OpenAIEmbeddingClient
from app.infrastructure.llm import OpenAIChatClient, OpenAIUtilityClient
from app.infrastructure.moderation import OpenAIModerationClient
from app.infrastructure.persistence import (
    SqliteConversationRepository,
    SqliteDocumentRepository,
    SqliteUsageRepository,
    create_database,
)
from app.infrastructure.storage import LocalFileStorage
from app.infrastructure.vectorstores import ChromaVectorStore
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval import (
    BM25Index,
    HybridRetriever,
    LLMReranker,
    QueryRouter,
    SemanticResponseCache,
)
from app.services import ChatService, ConversationService, DocumentService, UsageService

logger = getLogger(__name__)


def bootstrap_dependencies(app: FastAPI, settings: Settings, *, force_reindex: bool = False) -> None:
    """Build the full dependency graph and attach it to app.state."""

    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set — the backend will start but LLM calls will fail."
        )

    # ---- Persistence layer ----
    db = create_database(settings.database_url, base_dir=Path.cwd())
    conversation_repository = SqliteConversationRepository(db)
    document_repository = SqliteDocumentRepository(db)
    usage_repository = SqliteUsageRepository(db)

    # ---- OpenAI clients ----
    embedding_client = OpenAIEmbeddingClient(
        model_name=settings.embedding_model, api_key=settings.openai_api_key
    )
    chat_client = OpenAIChatClient(
        model_name=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )
    utility_client = OpenAIUtilityClient(
        model_name=settings.utility_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
    moderation_client = OpenAIModerationClient(
        api_key=settings.openai_api_key, model=settings.moderation_model
    )

    # ---- Vector store, BM25 index, ingestion ----
    vector_store = ChromaVectorStore(settings.vectorstore_path, embedding_client)
    bm25_index = BM25Index()

    def _rebuild_bm25() -> None:
        try:
            docs = list(vector_store.get())
            bm25_index.rebuild(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 rebuild failed: %s", exc)

    ingestion_pipeline = DocumentIngestionPipeline(settings=settings, vector_store=vector_store)
    ingestion_pipeline.set_on_change(_rebuild_bm25)

    storage = LocalFileStorage(settings.docs_path)
    document_service = DocumentService(document_repository, storage, ingestion_pipeline)
    conversation_service = ConversationService(conversation_repository, settings)

    # ---- Retrieval components ----
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        vector_weight=settings.hybrid_vector_weight,
        bm25_weight=settings.hybrid_bm25_weight,
        rrf_k=settings.rrf_k,
    )
    reranker = LLMReranker(utility_client)
    router = QueryRouter(utility_client)
    response_cache = SemanticResponseCache(
        embedding_client, threshold=settings.semantic_cache_threshold
    )

    # ---- Warm / reindex, then seed BM25 ----
    if force_reindex:
        _reset_and_reindex(document_service)
    else:
        _warm_vector_store(document_service)
    _rebuild_bm25()

    chat_service = ChatService(
        settings=settings,
        conversation_service=conversation_service,
        vector_store=vector_store,
        chat_client=chat_client,
        utility_client=utility_client,
        embedding_client=embedding_client,
        moderation_client=moderation_client,
        retriever=retriever,
        reranker=reranker,
        router=router,
        cache=response_cache,
        usage_repository=usage_repository,
    )
    usage_service = UsageService(usage_repository)

    app.state.settings = settings
    app.state.database = db
    app.state.conversation_service = conversation_service
    app.state.document_service = document_service
    app.state.chat_service = chat_service
    app.state.usage_service = usage_service
    app.state.chat_client = chat_client
    app.state.utility_client = utility_client
    app.state.embedding_client = embedding_client
    app.state.vector_store = vector_store
    app.state.bm25_index = bm25_index
    app.state.response_cache = response_cache


def refresh_llm_client(app: FastAPI, settings: Settings) -> None:
    """Rebuild the chat & utility clients in place when models change at runtime."""

    existing_chat: Any | None = getattr(app.state, "chat_service", None)
    if existing_chat is None:
        bootstrap_dependencies(app, settings)
        return

    chat_client = OpenAIChatClient(
        model_name=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )
    utility_client = OpenAIUtilityClient(
        model_name=settings.utility_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )

    existing_chat._chat = chat_client  # type: ignore[attr-defined]
    existing_chat._utility = utility_client  # type: ignore[attr-defined]
    existing_chat._reranker._client = utility_client  # type: ignore[attr-defined]
    existing_chat._router._client = utility_client  # type: ignore[attr-defined]

    app.state.chat_client = chat_client
    app.state.utility_client = utility_client


def _warm_vector_store(document_service: DocumentService) -> None:
    try:
        if document_service.vector_count() == 0:
            document_service.reindex_all()
    except FileNotFoundError:
        document_service.reset_index()


def _reset_and_reindex(document_service: DocumentService) -> None:
    try:
        document_service.reset_index()
        document_service.reindex_all()
    except FileNotFoundError:
        document_service.reset_index()


__all__ = ["bootstrap_dependencies", "refresh_llm_client"]
