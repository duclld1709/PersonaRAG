"""Healthcheck endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Service healthcheck")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
