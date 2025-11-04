"""API route registrations."""

from fastapi import APIRouter
from . import health, chat, conversations, documents, settings

api_router = APIRouter()
api_router.include_router(health.router, pre_fix="/health", tags=["health"])
api_router.include_router(chat.router, pre_fix="/chat", tags=["chat"])
api_router.include_router(conversations.router, pre_fix="/conversations", tags=["conversations"])
api_router.include_router(documents.router, pre_fix="/documents", tags=["documents"])
api_router.include_router(settings.router, pre_fix="/settings", tags=["settings"])