"""PhishVision AI REST API Application.

Multi-Modal Phishing Website and Brand Impersonation Detection Engine.
FastAPI Application Entry Point.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import get_pipeline, router as api_v1_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phishvision.main")

# Safe local development CORS defaults
DEFAULT_DEV_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def resolve_cors_configuration() -> tuple[list[str], bool]:
    """Resolve CORS origins and credentials flag from environment or safe local defaults.

    Production Configuration:
        Set `PHISHVISION_CORS_ORIGINS` to a comma-separated string of trusted frontend origins:
        e.g. `PHISHVISION_CORS_ORIGINS=https://portal.domain.com,https://dashboard.domain.com`

    Security Constraint:
        Wildcard `*` combined with `allow_credentials=True` is prohibited by the Fetch/CORS
        specifications and represents a credential leakage vulnerability. If wildcard is
        configured, credentials are automatically disabled.
    """
    raw_env = os.getenv("PHISHVISION_CORS_ORIGINS", "").strip()
    if not raw_env:
        return DEFAULT_DEV_CORS_ORIGINS, True

    origins = [orig.strip() for orig in raw_env.split(",") if orig.strip()]
    if "*" in origins:
        logger.warning(
            "CORS configured with wildcard '*'. Disabling allow_credentials to comply with security standards."
        )
        return ["*"], False
    return origins, True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle management."""
    logger.info("PhishVision AI Server starting up...")
    # Pre-warm unified pipeline in background to accelerate first inference
    try:
        get_pipeline()
        logger.info("Unified Seven-Model Phishing Pipeline successfully pre-warmed.")
    except Exception as e:
        logger.warning(f"Pipeline pre-warm warning (checkpoints will be loaded on demand): {e}")

    yield

    logger.info("PhishVision AI Server shutting down...")


app = FastAPI(
    title="PhishVision AI",
    description=(
        "Production-Grade Multi-Modal Phishing Website and Brand Impersonation Detector. "
        "Orchestrates seven specialized neural and lexical models (M-01 through M-07) "
        "with connection-level Anti-SSRF enforcement and explainable spatial consensus."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS) Middleware
cors_origins, allow_credentials = resolve_cors_configuration()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_v1_router)


@app.get("/", tags=["Root"])
async def root_status() -> dict[str, str]:
    """Root health and discovery endpoint."""
    return {
        "service": "PhishVision AI",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "docs": "/docs",
        "api_v1": "/api/v1",
        "health": "/api/v1/health",
        "brands": "/api/v1/brands",
    }
