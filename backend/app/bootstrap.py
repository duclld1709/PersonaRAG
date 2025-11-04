"""Shared application bootstrap utilities."""

import os
from typing import Any

from fastapi import FastAPI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from backend.app.core.config import Settings

from app.infrastructure.persistence import FileDocumentRepository, InMemoryConversationRepository
from backend.app.infrastructure.storage.file_storage import LocalFileStorage
from backend.app.infrastructure.vectorstores.chroma import ChromaVectorStore
from backend.app.ingestion.pipeline import DocumentIngestionPipeline
from backend.app.services.chat_service import ChatService
from backend.app.services.conversation_service import ConversationService
from backend.app.services.document_service import DocumentService
from app.infrastructure.llm import GeminiChatClient, OpenAIChatClient

def bootstrap_dependencies(app: FastAPI, settings: Settings, *, force_reindex:bool = False) -> None:
    embedding_model = _resolve_embedding_model(app, settings)
    vector_store = ChromaVectorStore(settings.vectorstore_path, embedding_model)

    ingestion_pipeline = DocumentIngestionPipeline(settings=settings, vector_store=vector_store)

    conversation_repository = _resolve_conversation_repository(app)
    conversation_service = ConversationService(conversation_repository, settings)

    metadata_index = settings.vectorstore_path.parent / "documents_index.json"
    document_repository = FileDocumentRepository(metadata_index)
    storage = LocalFileStorage(settings.docs_path)
    document_service = DocumentService(document_repository, storage, ingestion_pipeline)

    llm_client = _create_llm_client(settings)
    chat_service = ChatService(
        settings=settings,
        conversation_service=conversation_service,
        vector_store=vector_store,
        llm_client=llm_client,
    )

    if force_reindex:
        _reset_and_reindex(document_service)
    else:
        _warm_vector_store(document_service)

    app.state.settings = settings
    app.state.chat_service = chat_service
    app.state.conversation_service = conversation_service
    app.state.document_service = document_service
    app.state.embedding_model = embedding_model
    app.state.embedding_model_config = _embedding_config_for(settings)


def _resolve_embedding_model(app: FastAPI, settings: Settings) -> Embeddings:
    config = _embedding_config_for(settings)
    cached_model: Any | None = getattr(app.state, "embedding_model", None)
    cached_config = getattr(app.state, "embedding_model_config", None)

    if cached_model is not None and cached_config == config:
        return cached_model
    
    embedding_model = _create_embedding_model(settings)
    app.state.embedding_model = embedding_model
    app.state.embedding_model_config = config
    return embedding_model


def _resolve_conversation_repository(app: FastAPI) -> InMemoryConversationRepository:
    existing_service: Any | None = getattr(app.state, "conversation_service", None)
    if existing_service is not None:
        repository = getattr(existing_service, "_repository", None)
        if isinstance(repository, InMemoryConversationRepository):
            return repository
    return InMemoryConversationRepository()
    
def _embedding_config_for(settings: Settings) -> tuple[str, str]:
    if settings.embedding_provider == "huggingface":
        return (settings.embedding_provider, settings.huggingface_embedding_model)
    return (settings.embedding_provider, settings.embedding_model)

def _create_embedding_model(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "huggingface":
        if settings.huggingface_api_token:
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", settings.huggingface_api_token)
        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

def _create_llm_client(settings: Settings) -> OpenAIChatClient | GeminiChatClient:
    if settings.llm_provider == "gemini":
        return GeminiChatClient(model_name=settings.gemini_chat_model, api_key=settings.gemini_api_key)
    return OpenAIChatClient(model_name=settings.chat_model, api_key=settings.openai_api_key)

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