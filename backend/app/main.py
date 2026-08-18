import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze, auth, health, investigate

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


async def _check_groq_reachable() -> None:
    settings = get_settings()
    if settings.llm_provider != "groq":
        return
    if not settings.groq_api_key:
        logger.warning("LLM_PROVIDER=groq but GROQ_API_KEY is missing in backend/.env")
    else:
        logger.info("Groq Provider initialized (model=%s).", settings.groq_model)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await _check_groq_reachable()
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins else ["http://localhost:5500", "http://localhost:5510"],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(investigate.router)

    return app


app = create_app()
