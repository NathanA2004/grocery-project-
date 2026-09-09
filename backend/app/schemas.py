"""Pydantic v2 data schemas for the Smart Grocery Route Optimizer (MO-VRP-JSC)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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
