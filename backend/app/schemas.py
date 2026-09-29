"""Pydantic v2 data schemas for the Smart Grocery Route Optimizer (MO-VRP-JSC)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Reject type coercion and unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class UnitType(str, Enum):
    """Price unit derived from flyer OCR text (Module 1)."""

    PER_LB = "PER_LB"
    PER_KG = "PER_KG"
    FLAT_ITEM = "FLAT_ITEM"
    MULTI_BUY = "MULTI_BUY"


class ProductTile(StrictModel):
    """A single flyer product tile after YOLOv8 crop + OCR + regex normalization."""

    id: str = Field(..., min_length=1, description="Unique ProductTile ID mapped in the FAISS index.")
    store_id: str = Field(..., min_length=1, description="Store that published this tile.")
    clean_title: str = Field(..., min_length=1, description="Product title string.")
    raw_price: str = Field(
        ...,
        min_length=1,
        description='Unparsed string (e.g. "2 for $5.00" or "$3.99/lb ($8.79/kg)").',
    )
    unit_type: UnitType = Field(
        ...,
        description="Enum: PER_LB, PER_KG, FLAT_ITEM, or MULTI_BUY.",
    )
    normalized_price: float = Field(
        ...,
        ge=0,
        description="Standardized float value per unit ($/lb or $/unit).",
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="YOLOv8 product_tile bounding box [x1, y1, x2, y2].",
    )


class StoreMetadata(StrictModel):
    """Store identity and map coordinates used for Dist_{j,k} and visit decisions."""

    store_id: str = Field(..., min_length=1, description="Unique store identifier.")
    name: str = Field(..., min_length=1, description="Human-readable store name.")
    latitude: float = Field(..., ge=-90, le=90, description="Store latitude.")
    longitude: float = Field(..., ge=-180, le=180, description="Store longitude.")


class ShoppingListRequest(StrictModel):
    """POST /api/optimize body: shopping list, origin coordinates, and objective weights."""

    items: list[str] = Field(
        ...,
        min_length=1,
        description="User shopping list query terms (e.g. 'eggs', 'chicken breast').",
    )
    origin_latitude: float = Field(..., ge=-90, le=90, description="Trip origin latitude.")
    origin_longitude: float = Field(..., ge=-180, le=180, description="Trip origin longitude.")
    w_dist: float = Field(..., ge=0, description="Weight on total driving distance (w_dist).")
    w_stop: float = Field(..., ge=0, description="Weight on per-store visit overhead (w_stop).")


class ReceiptItem(StrictModel):
    """One fulfilled shopping-list item on the optimized receipt."""

    requested_item: str = Field(..., min_length=1, description="Original user shopping-list term.")
    product: ProductTile = Field(..., description="Matched flyer tile purchased for this item.")
    store_id: str = Field(..., min_length=1, description="Store where the item is bought (X_{i,s}).")


class OptimizationResponse(StrictModel):
    """POST /api/optimize result: store trip sequence, savings, and receipt breakdown."""

    trip_sequence: list[StoreMetadata] = Field(
        ...,
        description="Optimized store trip sequence in visit order.",
    )
    savings: float = Field(..., description="Total savings versus a single-store / baseline basket.")
    receipt_breakdown: list[ReceiptItem] = Field(
        ...,
        description="Per-item assignment of matched products, stores, and prices.",
    )


class OptimizationRequest(StrictModel):
    """POST /api/optimize body: shopping list, home-to-store distances, and visit limits."""

    items: list[str] = Field(..., description="User's grocery items, one entry per unit to buy.")
    store_distances: dict[str, float] = Field(
        ...,
        description="One-way distance from home to each store, keyed by store id.",
    )
    max_stores: int | None = Field(
        default=3,
        ge=0,
        description="Maximum number of distinct stores the trip may visit. None means no cap.",
    )
    cost_per_km: float = Field(
        default=0.20,
        ge=0,
        description=(
            "Cost charged per kilometer of one-way distance. "
            "A visit is a round trip, so the solver charges 2 * distance * cost_per_km."
        ),
    )

    @field_validator("store_distances")
    @classmethod
    def _distances_are_non_negative(cls, distances: dict[str, float]) -> dict[str, float]:
        negative = [store_id for store_id, distance in distances.items() if distance < 0]
        if negative:
            joined = ", ".join(sorted(negative))
            raise ValueError(f"store distances must be >= 0 (negative for {joined}).")
        return distances


class OptimizationResult(StrictModel):
    """Solved basket: money spent, round-trip travel, and the per-store purchases."""

    total_cost: float = Field(..., description="grocery_cost + travel_cost.")
    grocery_cost: float = Field(..., description="Sum of normalized prices of the chosen tiles.")
    travel_cost: float = Field(
        ...,
        description="Sum of round-trip charges (2 * distance * cost_per_km) for visited stores.",
    )
    store_itinerary: dict[str, list[ProductTile]] = Field(
        ...,
        description="Tiles purchased at each visited store, keyed by store id.",
    )
    unfulfilled_items: list[str] = Field(
        ...,
        description="Requested lines with no usable match, or that the store cap left unbought.",
    )


class ParseFlyerRequest(StrictModel):
    """Metadata accepted with a multipart flyer upload (field name ``file``)."""

    store_id: str | None = Field(
        default=None,
        min_length=1,
        description="Store that published this flyer. Defaults to the upload file name when omitted.",
    )


class ParseFlyerResponse(StrictModel):
    """Tiles parsed from the upload and written into the local FAISS index."""

    tiles: list[ProductTile] = Field(..., description="Product tiles extracted from the flyer.")
    indexed_count: int = Field(
        ...,
        ge=0,
        description="Tiles newly added to the FAISS index. Duplicates of an existing id are skipped.",
    )
    index_total: int = Field(..., ge=0, description="FAISS index.ntotal after this upload.")


class SearchRequest(StrictModel):
    """POST /api/search body: shopping-list strings matched against the local index."""

    items: list[str] = Field(..., min_length=1, description="Query strings, one per shopping-list line.")
    top_k: int = Field(default=5, ge=1, description="Maximum candidates kept for each query.")
    threshold: float = Field(
        default=0.60,
        ge=-1,
        le=1,
        description="Minimum cosine similarity returned from the local FAISS index.",
    )

    @field_validator("items")
    @classmethod
    def _items_are_non_blank(cls, items: list[str]) -> list[str]:
        if any(not item.strip() for item in items):
            raise ValueError("search items must be non-blank strings.")
        return items


class CandidateTile(StrictModel):
    """One local FAISS hit: the flyer tile and its cosine similarity."""

    product: ProductTile
    score: float = Field(..., description="Cosine similarity from the L2-normalized FAISS index.")


class SearchResponse(StrictModel):
    """Candidate tiles for each query, best score first."""

    matches: dict[str, list[CandidateTile]] = Field(
        ...,
        description="Query string to candidate tiles that cleared the similarity threshold.",
    )


class HealthResponse(StrictModel):
    """GET /api/health: process status, where local models are running, and index size."""

    status: str = Field(..., description="API liveness. 'ok' when the local models are loaded.")
    device: str = Field(..., description="Execution device of the local embedding model: cpu or cuda.")
    cuda_available: bool = Field(..., description="Whether this machine exposes a CUDA device to PyTorch.")
    models: dict[str, str] = Field(..., description="Local model and solver identifiers. No hosted APIs.")
    faiss_index_count: int = Field(..., ge=0, description="Number of vectors in the FAISS index (index.ntotal).")
