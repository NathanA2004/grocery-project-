"""Local REST routes. Parsing, embeddings, search, and the MILP all run in-process."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from ..optimization.solver import OptimizationSolver
from ..schemas import (
    CandidateTile,
    HealthResponse,
    OptimizationRequest,
    OptimizationResult,
    ParseFlyerRequest,
    ParseFlyerResponse,
    ProductTile,
    SearchRequest,
    SearchResponse,
)
from ..vector_search.matcher import MODEL_NAME, SemanticMatcher
from ..vision.flyer_parser import FlyerParser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["grocery"])

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_PDF_SUFFIXES = {".pdf"}
_ALLOWED_SUFFIXES = _IMAGE_SUFFIXES | _PDF_SUFFIXES
_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/tif",
}
_BAD_PDF_ERRORS = {"PDFPageCountError", "PDFSyntaxError", "PDFPopplerTimeoutError"}


def get_parser(request: Request) -> FlyerParser:
    """Return the process-local flyer parser created at startup."""
    parser = getattr(request.app.state, "parser", None)
    if parser is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local flyer parser is not loaded.",
        )
    return parser


def get_matcher(request: Request) -> SemanticMatcher:
    """Return the process-local semantic matcher created at startup."""
    matcher = getattr(request.app.state, "matcher", None)
    if matcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local semantic matcher is not loaded.",
        )
    return matcher


def _index_lock(request: Request) -> threading.Lock:
    lock = getattr(request.app.state, "index_lock", None)
    if not isinstance(lock, threading.Lock):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local search index is not loaded.",
        )
    return lock


def _parser_lock(request: Request) -> threading.Lock:
    lock = getattr(request.app.state, "parser_lock", None)
    if not isinstance(lock, threading.Lock):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local flyer parser is not loaded.",
        )
    return lock


def _checked_upload(file: UploadFile, data: bytes) -> str:
    """Return the lowercase suffix, or raise HTTP 400 for an unusable upload."""
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A flyer file name is required.",
        )
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Flyer upload must be a PDF or an image (png, jpg, jpeg, webp, tiff, bmp).",
        )
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type and content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported flyer content type: {content_type}.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The flyer upload is empty.",
        )
    if suffix in _PDF_SUFFIXES and not data.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a PDF.",
        )
    return suffix


def _flyer_metadata(store_id: str | None) -> ParseFlyerRequest:
    if store_id is not None and not store_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="store_id cannot be empty.",
        )
    try:
        if store_id is None:
            return ParseFlyerRequest()
        return ParseFlyerRequest(store_id=store_id.strip())
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.errors(),
        ) from exc


def _store_id_for_upload(filename: str, requested: str | None) -> str:
    if requested:
        return requested
    stem = Path(filename).stem.strip()
    return stem or "uploaded"


def _parse_saved_flyer(parser: FlyerParser, path: str, suffix: str, store_id: str) -> list[ProductTile]:
    previous = parser.store_id
    parser.store_id = store_id
    try:
        if suffix in _PDF_SUFFIXES:
            return parser.parse_flyer_pdf(path)
        return parser.parse_flyer_page(path)
    finally:
        parser.store_id = previous


def _parse_under_lock(
    parser: FlyerParser,
    lock: threading.Lock,
    path: str,
    suffix: str,
    store_id: str,
) -> list[ProductTile]:
    with lock:
        return _parse_saved_flyer(parser, path, suffix, store_id)


def _index_under_lock(
    matcher: SemanticMatcher,
    lock: threading.Lock,
    tiles: list[ProductTile],
    state_dir: Path | None,
) -> tuple[int, int]:
    with lock:
        before = int(matcher.index.ntotal)
        matcher.index_flyer_items(tiles)
        index_total = int(matcher.index.ntotal)
        if state_dir is not None:
            try:
                matcher.save_state(state_dir)
            except Exception:
                logger.exception("Failed to persist the local FAISS index to %s.", state_dir)
        return before, index_total


def _parsing_http_error(exc: Exception) -> HTTPException:
    """Map parser failures onto 400 (bad flyer) or 500 (local pipeline failure)."""
    if isinstance(exc, FileNotFoundError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded flyer could not be read as an image.",
        )
    if isinstance(exc, HTTPException):
        return exc
    error_name = type(exc).__name__
    if error_name in _BAD_PDF_ERRORS:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF could not be read.",
        )
    if error_name == "PDFInfoNotInstalledError":
        logger.exception("Poppler is required to rasterize flyer PDFs locally.")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Local PDF rendering failed because poppler is not installed.",
        )
    logger.exception("Flyer parsing failed.")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Flyer parsing failed.",
    )


def _execution_device(matcher: SemanticMatcher) -> tuple[str, bool]:
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        logger.debug("PyTorch is not importable while reading the execution device.", exc_info=True)
    device = matcher.device
    kind = "cuda" if device.startswith("cuda") else "cpu"
    return kind, cuda_available


def _model_details(parser: FlyerParser, matcher: SemanticMatcher) -> dict[str, str]:
    yolo_device = "cpu"
    yolo_model = getattr(parser.yolo, "model", None)
    parameters = getattr(yolo_model, "parameters", None)
    if parameters is not None:
        try:
            yolo_device = str(next(parameters()).device)
        except Exception:
            logger.debug("Could not read the YOLO parameter device.", exc_info=True)
    weights = getattr(parser.yolo, "ckpt_path", None) or "yolov8n.pt"
    return {
        "layout_detector": f"ultralytics.YOLO ({Path(str(weights)).name}) on {yolo_device}",
        "ocr": "PaddleOCR (local, lang=en)",
        "embeddings": f"{matcher.model_name or MODEL_NAME} on {matcher.device}",
        "vector_index": "faiss.IndexFlatIP",
        "optimizer": "PuLP MILP (local CBC/HiGHS)",
    }


@router.post("/parse-flyer", response_model=ParseFlyerResponse)
async def parse_flyer(
    request: Request,
    file: UploadFile = File(..., description="Grocery flyer PDF or image."),
    store_id: str | None = Form(default=None),
    parser: FlyerParser = Depends(get_parser),
    matcher: SemanticMatcher = Depends(get_matcher),
) -> ParseFlyerResponse:
    """Run Module 1 on the upload and index the resulting tiles with Module 2."""
    data = await file.read()
    suffix = _checked_upload(file, data)
    meta = _flyer_metadata(store_id)
    resolved_store = _store_id_for_upload(file.filename or "", meta.store_id)
    parser_lock = _parser_lock(request)
    index_lock = _index_lock(request)
    state_dir = getattr(request.app.state, "faiss_state_dir", None)

    fd, raw_path = tempfile.mkstemp(suffix=suffix)
    path = raw_path
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        try:
            # YOLO and PaddleOCR are CPU/GPU-bound. Run them off the event loop.
            tiles = await asyncio.to_thread(
                _parse_under_lock,
                parser,
                parser_lock,
                path,
                suffix,
                resolved_store,
            )
        except Exception as exc:
            raise _parsing_http_error(exc) from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            logger.warning("Could not delete temporary flyer %s.", path)

    try:
        before, index_total = await asyncio.to_thread(
            _index_under_lock,
            matcher,
            index_lock,
            tiles,
            state_dir,
        )
    except Exception as exc:
        logger.exception("Failed to index parsed flyer tiles.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flyer parsing failed.",
        ) from exc

    return ParseFlyerResponse(
        tiles=tiles,
        indexed_count=index_total - before,
        index_total=index_total,
    )


@router.post("/search", response_model=SearchResponse)
def search_items(
    body: SearchRequest,
    request: Request,
    matcher: SemanticMatcher = Depends(get_matcher),
) -> SearchResponse:
    """Embed the shopping list locally and return FAISS candidates above the threshold."""
    try:
        with _index_lock(request):
            raw = matcher.find_item_matches(body.items, top_k=body.top_k, threshold=body.threshold)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    matches = {
        query: [CandidateTile(product=hit["product"], score=float(hit["score"])) for hit in hits]
        for query, hits in raw.items()
    }
    return SearchResponse(matches=matches)


@router.post("/optimize", response_model=OptimizationResult)
def optimize_basket(
    body: OptimizationRequest,
    request: Request,
    matcher: SemanticMatcher = Depends(get_matcher),
) -> OptimizationResult:
    """Match the list locally, then solve the basket with round-trip travel costs.

    ``OptimizationSolver`` charges ``2 * distance * cost_per_km`` for each
    visited store. If ``max_stores`` makes a complete basket infeasible, the
    fallback MILP keeps a high penalty on every unfulfilled line.
    """
    try:
        with _index_lock(request):
            matched = matcher.find_item_matches(list(body.items))
        return OptimizationSolver(matched, body).solve()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Local grocery optimization failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The local optimizer could not solve this basket.",
        ) from exc


@router.get("/health", response_model=HealthResponse)
def health(
    parser: FlyerParser = Depends(get_parser),
    matcher: SemanticMatcher = Depends(get_matcher),
) -> HealthResponse:
    """Report API status, CPU/CUDA execution, local models, and FAISS index.ntotal."""
    device, cuda_available = _execution_device(matcher)
    return HealthResponse(
        status="ok",
        device=device,
        cuda_available=cuda_available,
        models=_model_details(parser, matcher),
        faiss_index_count=int(matcher.index.ntotal),
    )
