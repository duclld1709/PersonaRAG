"""FastAPI application entrypoint: ``uvicorn app.main:app``."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.bootstrap import Providers, bootstrap_dependencies
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging


def create_application(settings: Settings | None = None, *, providers: Providers | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        description="Research Assistant for Students: hybrid RAG on OpenAI.",
        version="2.0.0",
    )
    # Browsers reject `*` together with credentials, so origins are explicit.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bootstrap_dependencies(app, settings, providers=providers)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


def __getattr__(name: str):
    # Built lazily so importing this module (tests, eval) does not connect to OpenAI.
    if name == "app":
        globals()["app"] = application = create_application()
        return application
    raise AttributeError(name)
