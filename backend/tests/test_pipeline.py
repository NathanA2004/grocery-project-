"""Offline end-to-end checks for the local grocery API.

pytest drives the FastAPI app through ``TestClient``. YOLO, PaddleOCR, and
MiniLM are not loaded: a sample flyer image or PDF is parsed by an in-process
stand-in, and semantic search uses that catalog. The HTTP routes and the
PuLP solver run as they do in the app, with no network calls.
"""

from __future__ import annotations

import os
import struct
import sys
import types
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Keep model hubs from being contacted if a library checks the environment
# during import. The tests below never construct the real parsers or encoders.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if "sentence_transformers" not in sys.modules:
    _sentence_transformers = types.ModuleType("sentence_transformers")

    class _OfflineSentenceTransformer:
        """Import stand-in. Tests never call this; it blocks a weight download."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Offline tests do not load sentence-transformers weights.")

    _sentence_transformers.SentenceTransformer = _OfflineSentenceTransformer
    sys.modules["sentence_transformers"] = _sentence_transformers

from app.main import app
from app.schemas import ProductTile, UnitType

# Home-to-store kilometers. Travel is a round trip: 2 * distance * cost_per_km.
_DISTANCES = {"market": 4.0, "corner": 6.0, "plaza": 20.0}
_COST_PER_KM = 0.5
_MAX_STORES = 2
_GROCERY_LIST = ["milk", "eggs", "bread", "juice"]

# Milk and eggs share market, bread is only at corner, juice is only at plaza.
# With max_stores=2 the cheapest feasible basket leaves juice unbought.
_CATALOG = [
    ProductTile(
        id="milk-market",
        store_id="market",
        clean_title="Whole Milk",
        raw_price="$2.00",
        unit_type=UnitType.FLAT_ITEM,
        normalized_price=2.0,
        bbox=[0.0, 0.0, 40.0, 40.0],
    ),
    ProductTile(
        id="eggs-market",
        store_id="market",
        clean_title="Large Eggs",
        raw_price="$3.00",
        unit_type=UnitType.FLAT_ITEM,
        normalized_price=3.0,
        bbox=[40.0, 0.0, 80.0, 40.0],
    ),
    ProductTile(
        id="bread-corner",
        store_id="corner",
        clean_title="Sourdough Bread",
        raw_price="$4.00",
        unit_type=UnitType.FLAT_ITEM,
        normalized_price=4.0,
        bbox=[0.0, 40.0, 40.0, 80.0],
    ),
    ProductTile(
        id="juice-plaza",
        store_id="plaza",
        clean_title="Orange Juice",
        raw_price="$5.00",
        unit_type=UnitType.FLAT_ITEM,
        normalized_price=5.0,
        bbox=[40.0, 40.0, 80.0, 80.0],
    ),
]


class _IndexCount:
    """Stand-in for ``faiss.Index.ntotal`` used by ``GET /api/health``."""

    def __init__(self) -> None:
        self.ntotal = 0


class FakeFlyerParser:
    """Return the fixed catalog for a local sample flyer. No YOLO or OCR."""

    def __init__(self) -> None:
        self.store_id: str | None = None
        self.conf = 0.25
        self.ocr = None
        self.yolo = types.SimpleNamespace(ckpt_path="yolov8n.pt")

    def parse_flyer_page(self, image_path: str) -> list[ProductTile]:
        if not Path(image_path).read_bytes():
            return []
        return list(_CATALOG)

    def parse_flyer_pdf(self, pdf_path: str, dpi: int = 300) -> list[ProductTile]:
        del dpi
        data = Path(pdf_path).read_bytes()
        if not data.startswith(b"%PDF"):
            raise ValueError("The uploaded file is not a PDF.")
        return list(_CATALOG)


class FakeSemanticMatcher:
    """Keyword match over the indexed catalog. No embedding model is loaded."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._device = "cpu"
        self._tiles: list[ProductTile] = []
        self._index = _IndexCount()

    @property
    def index(self) -> _IndexCount:
        return self._index

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_state(self, directory: str | Path) -> bool:
        del directory
        return False

    def save_state(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    def index_flyer_items(self, items: list[ProductTile]) -> None:
        known = {tile.id for tile in self._tiles}
        for tile in items:
            if tile.id in known:
                continue
            known.add(tile.id)
            self._tiles.append(tile)
        self._index.ntotal = len(self._tiles)

    def find_item_matches(
        self,
        user_items: list[str],
        top_k: int = 5,
        threshold: float = 0.60,
    ) -> dict[str, list[dict[str, object]]]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        matches: dict[str, list[dict[str, object]]] = {query: [] for query in user_items}
        if not user_items or not self._tiles:
            return matches
        for query in user_items:
            needle = query.casefold()
            hits: list[dict[str, object]] = []
            for rank, tile in enumerate(self._tiles):
                haystack = f"{tile.clean_title} {tile.id}".casefold()
                if needle not in haystack:
                    continue
                score = round(0.99 - (rank * 0.01), 4)
                if score < threshold:
                    continue
                hits.append({"product": tile, "score": score})
            hits.sort(key=lambda hit: float(hit["score"]), reverse=True)
            matches[query] = hits[:top_k]
        return matches


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def sample_flyer_png() -> bytes:
    """1x1 RGB PNG. The route only needs a non-empty image upload."""
    raw = b"\x00\xff\x00\x00"
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def sample_flyer_pdf() -> bytes:
    """Tiny PDF. The route checks the ``%PDF`` header and never rasterizes it here."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
        b"%%EOF\n"
    )


def _upload_sample_flyer(client: TestClient, *, kind: str = "png") -> dict[str, object]:
    if kind == "pdf":
        filename, payload, content_type = "sample-flyer.pdf", sample_flyer_pdf(), "application/pdf"
    else:
        filename, payload, content_type = "sample-flyer.png", sample_flyer_png(), "image/png"
    response = client.post(
        "/api/parse-flyer",
        files={"file": (filename, payload, content_type)},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["indexed_count"] == len(_CATALOG)
    assert body["index_total"] == len(_CATALOG)
    return body


def _optimize_payload() -> dict[str, object]:
    return {
        "items": _GROCERY_LIST,
        "store_distances": _DISTANCES,
        "max_stores": _MAX_STORES,
        "cost_per_km": _COST_PER_KM,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    """App client whose startup stays on disk and never fetches model weights."""
    monkeypatch.setenv("GROCERY_FAISS_DIR", str(tmp_path / "faiss"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr("app.main.FlyerParser", FakeFlyerParser)
    monkeypatch.setattr("app.main.SemanticMatcher", FakeSemanticMatcher)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_status_device_and_index_count(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["device"] in {"cpu", "cuda"}
    assert isinstance(body["cuda_available"], bool)
    if body["device"] == "cuda":
        assert body["cuda_available"] is True
    assert body["faiss_index_count"] == 0
    assert body["models"]["vector_index"] == "faiss.IndexFlatIP"
    assert "cpu" in body["models"]["embeddings"] or "cuda" in body["models"]["embeddings"]


def test_search_returns_structured_matches_and_handles_empty_queries(client: TestClient) -> None:
    empty = client.post("/api/search", json={"items": []})
    assert empty.status_code == 422
    detail = empty.json()["detail"]
    assert isinstance(detail, list)
    assert any("items" in err.get("loc", []) for err in detail)

    _upload_sample_flyer(client, kind="png")

    found = client.post(
        "/api/search",
        json={"items": ["milk", "eggs"], "top_k": 5, "threshold": 0.6},
    )
    assert found.status_code == 200
    matches = found.json()["matches"]
    assert set(matches) == {"milk", "eggs"}

    milk = matches["milk"][0]
    assert set(milk) == {"product", "score"}
    assert isinstance(milk["score"], float)
    assert 0.6 <= milk["score"] <= 1.0
    assert milk["product"]["id"] == "milk-market"
    assert milk["product"]["store_id"] == "market"
    assert milk["product"]["clean_title"] == "Whole Milk"
    assert milk["product"]["normalized_price"] == 2.0
    assert milk["product"]["unit_type"] == "FLAT_ITEM"
    assert milk["product"]["bbox"] == [0.0, 0.0, 40.0, 40.0]

    eggs = matches["eggs"][0]
    assert eggs["product"]["id"] == "eggs-market"
    assert eggs["product"]["store_id"] == "market"
    assert isinstance(eggs["score"], float)

    missing = client.post("/api/search", json={"items": ["unicorn feed"]})
    assert missing.status_code == 200
    assert missing.json()["matches"] == {"unicorn feed": []}

    # The empty-list rejection must not take down the local process.
    assert client.get("/api/health").status_code == 200


def test_optimize_round_trip_cost_and_store_cap(client: TestClient) -> None:
    _upload_sample_flyer(client, kind="png")

    response = client.post("/api/optimize", json=_optimize_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    itinerary = body["store_itinerary"]
    assert len(itinerary) <= _MAX_STORES
    assert set(itinerary) == {"market", "corner"}

    expected_travel = sum(2 * _DISTANCES[store_id] * _COST_PER_KM for store_id in itinerary)
    expected_grocery = sum(
        tile["normalized_price"] for tiles in itinerary.values() for tile in tiles
    )
    assert body["travel_cost"] == expected_travel
    assert body["grocery_cost"] == expected_grocery
    assert body["total_cost"] == body["grocery_cost"] + body["travel_cost"]

    assert [tile["id"] for tile in itinerary["market"]] == ["milk-market", "eggs-market"]
    assert [tile["id"] for tile in itinerary["corner"]] == ["bread-corner"]
    for store_id, tiles in itinerary.items():
        assert tiles
        assert all(tile["store_id"] == store_id for tile in tiles)
    assert body["unfulfilled_items"] == ["juice"]


def test_end_to_end_health_search_and_optimize(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["faiss_index_count"] == 0

    _upload_sample_flyer(client, kind="pdf")

    search = client.post(
        "/api/search",
        json={"items": _GROCERY_LIST, "top_k": 3, "threshold": 0.6},
    )
    assert search.status_code == 200
    matches = search.json()["matches"]
    assert set(matches) == set(_GROCERY_LIST)
    assert all(hits and "product" in hits[0] and "score" in hits[0] for hits in matches.values())

    optimized = client.post("/api/optimize", json=_optimize_payload())
    assert optimized.status_code == 200, optimized.text
    basket = optimized.json()
    assert basket["total_cost"] == basket["grocery_cost"] + basket["travel_cost"]
    assert len(basket["store_itinerary"]) <= _MAX_STORES
    assert basket["store_itinerary"]
    assert client.get("/api/health").json()["faiss_index_count"] == len(_CATALOG)
