import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze, health, investigate

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def _check_ollama_reachable() -> None:
    """Warn, don't crash, if LLM_PROVIDER=ollama but nothing is listening -- so devs
    get a clear message instead of a cryptic connection-refused deep in a request."""
    settings = get_settings()
    if settings.llm_provider != "ollama":
        return

    base = settings.ollama_base_url.removesuffix("/v1")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(base)
        logger.info("Ollama reachable at %s (model=%s).", base, settings.ollama_model)
    except httpx.HTTPError:
        logger.warning(
            "LLM_PROVIDER=ollama but %s is not reachable. Is `ollama serve` running? "
            "Requests to /api/v1/analyze-finding will fail until it is.",
            base,
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _check_ollama_reachable()
    yield


def create_app() -> FastAPI:
    _configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="LQMS AI Gateway",
        description=(
            "LLM-only AI intelligence layer (finding analysis) for the existing LQMS "
            "Audit Management System. System of intelligence, not system of record -- "
            "never writes to the production LQMS database."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )

    origins = settings.allowed_origins_list
    if "*" not in origins:
        origins.append("*")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(investigate.router)

    return app


app = create_app()
