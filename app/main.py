"""PhishVision AI REST API Application.

Multi-Modal Phishing Website and Brand Impersonation Detection Engine.
FastAPI Application Entry Point.
"""

from __future__ import annotations

import logging
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
