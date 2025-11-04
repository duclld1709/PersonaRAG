"""Dependency providers for FastAPI routes."""

from fastapi import Request

from app.core.config import Settings
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


def get_document_service(request: Request) -> DocumentService:
    return request.app.state.document_service
