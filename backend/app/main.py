"""Local FastAPI app for flyer parsing, semantic search, and basket optimization.

Models and the MILP solver run in-process on CPU or CUDA. This process does
not call cloud LLMs or other hosted AI APIs.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .vector_search.matcher import SemanticMatcher
from .vision.flyer_parser import FlyerParser

logger = logging.getLogger(__name__)

# Next.js on :3000, plus the usual local dev servers.
LOCAL_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def faiss_state_dir() -> Path:
    """Directory for the on-disk FAISS index. ``GROCERY_FAISS_DIR`` overrides it."""
    override = os.environ.get("GROCERY_FAISS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "data" / "faiss"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the local parser and matcher, restore a saved index, then release them."""
    logger.info("Loading local FlyerParser and SemanticMatcher.")
    parser = FlyerParser()
    matcher = SemanticMatcher()
    state_dir = faiss_state_dir()
    try:
        loaded = matcher.load_state(state_dir)
    except Exception:
        logger.exception("Ignoring unreadable local FAISS state in %s.", state_dir)
        loaded = False
    if loaded:
        logger.info("Restored local FAISS index (%d vectors).", matcher.index.ntotal)
    else:
        logger.info("No saved FAISS index found in %s.", state_dir)

    app.state.parser = parser
    app.state.matcher = matcher
    app.state.faiss_state_dir = state_dir
    app.state.parser_lock = threading.Lock()
    app.state.index_lock = threading.Lock()
    try:
        yield
    finally:
        lock = getattr(app.state, "index_lock", None)
        try:
            if isinstance(lock, threading.Lock):
                with lock:
                    matcher.save_state(state_dir)
            else:
                matcher.save_state(state_dir)
        except Exception:
            logger.exception("Failed to save the local FAISS index to %s.", state_dir)
        app.state.parser = None
        app.state.matcher = None
        logger.info("Released local parser and matcher.")


app = FastAPI(
    title="Smart Grocery Route Optimizer",
    summary="Local flyer parsing, semantic search, and multi-store basket optimization.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
