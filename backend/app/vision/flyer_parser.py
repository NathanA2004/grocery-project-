"""Module 1: YOLOv8 tile cropping, PaddleOCR, and regex price normalization."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..schemas import ProductTile, UnitType

if TYPE_CHECKING:
    from paddleocr import PaddleOCR
    from ultralytics import YOLO

    import numpy as np

logger = logging.getLogger(__name__)

PRODUCT_TILE_ALIASES = frozenset({"product_tile", "product-tile", "producttile", "tile"})
LB_PER_KG = 2.20462262185
_MIN_CROP_PX = 8
_CROP_PAD_PX = 4

# "$3.99/lb ($8.79/kg)" — keep the flyer’s dual-unit phrase as raw_price.
_COMBINED_LB_KG = re.compile(
    r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*|per\s+)?(?:lbs?|pounds?)"
    r"\s*[\(\[]\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*|per\s+)?(?:kgs?|kilos?|kilograms?)\s*[\)\]]",
    re.IGNORECASE,
)
_PER_LB = re.compile(
    r"(?P<raw>\$?\s*(?P<price>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:/\s*|per\s+)?(?:lbs?|pounds?)\b)",
    re.IGNORECASE,
)
_PER_KG = re.compile(
    r"(?P<raw>\$?\s*(?P<price>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:/\s*|per\s+)?(?:kgs?|kilos?|kilograms?)\b)",
    re.IGNORECASE,
)
_MULTI_BUY = re.compile(
    r"(?P<raw>(?P<qty>[2-9]|[1-9]\d+)\s*(?:for|/)\s*\$?\s*(?P<price>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?))",
    re.IGNORECASE,
)
_EACH = re.compile(
    r"(?P<raw>\$?\s*(?P<price>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*(?:/\s*)?(?:ea|each|pc|pcs|ct)\b)",
    re.IGNORECASE,
)
_DOLLAR = re.compile(
    r"(?P<raw>\$\s*(?P<price>\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?))",
)
_CENTS = re.compile(
    r"(?P<raw>(?P<cents>\d{1,3})\s*(?:¢|cents?\b))",
    re.IGNORECASE,
)
_SAVE_PREFIX = re.compile(r"\b(?:save|off|less)\b", re.IGNORECASE)
_PRICE_STRIP = re.compile(
    r"|".join(
        (
            r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*|per\s+)?(?:lbs?|pounds?)"
            r"(?:\s*[\(\[]\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*|per\s+)?"
            r"(?:kgs?|kilos?|kilograms?)\s*[\)\]])?",
            r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*|per\s+)?(?:kgs?|kilos?|kilograms?)\b",
            r"(?:[2-9]|[1-9]\d+)\s*(?:for|/)\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",
            r"\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:/\s*)?(?:ea|each|pc|pcs|ct)\b",
            r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",
            r"\d{1,3}\s*(?:¢|cents?\b)",
        )
    ),
    re.IGNORECASE,
)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_raw_price(raw: str) -> str:
    raw = _collapse_ws(raw)
    raw = re.sub(r"\$\s+", "$", raw)
    raw = re.sub(r"\s*/\s*", "/", raw)
    return raw


def _parse_money(value: str) -> float:
    return float(value.replace(",", ""))


def _round_price(value: float) -> float:
    return round(float(value), 4)


def extract_price_and_unit(ocr_text: str) -> tuple[str, UnitType, float] | None:
    """Extract raw price, UnitType, and normalized $/lb or $/unit from OCR text.

    Handles flyer patterns such as ``$2.99/lb``, ``$4.50/kg``, ``2 for $5``,
    and ``$1.99 ea``. Kilogram prices are converted to dollars-per-pound so
    ``normalized_price`` is comparable across weight units.
    """
    if not ocr_text or not ocr_text.strip():
        return None

    blob = _collapse_ws(ocr_text)

    combined = _COMBINED_LB_KG.search(blob)
    lb_match = _PER_LB.search(blob)
    if lb_match:
        raw = _normalize_raw_price(combined.group(0) if combined else lb_match.group("raw"))
        return raw, UnitType.PER_LB, _round_price(_parse_money(lb_match.group("price")))

    kg_match = _PER_KG.search(blob)
    if kg_match:
        per_kg = _parse_money(kg_match.group("price"))
        return (
            _normalize_raw_price(kg_match.group("raw")),
            UnitType.PER_KG,
            _round_price(per_kg / LB_PER_KG),
        )

    multi = _MULTI_BUY.search(blob)
    if multi:
        qty = int(multi.group("qty"))
        total = _parse_money(multi.group("price"))
        return (
            _normalize_raw_price(multi.group("raw")),
            UnitType.MULTI_BUY,
            _round_price(total / qty),
        )

    each = _EACH.search(blob)
    if each:
        return (
            _normalize_raw_price(each.group("raw")),
            UnitType.FLAT_ITEM,
            _round_price(_parse_money(each.group("price"))),
        )

    for dollar in _DOLLAR.finditer(blob):
        prefix = blob[max(0, dollar.start() - 16) : dollar.start()]
        if _SAVE_PREFIX.search(prefix):
            continue
        return (
            _normalize_raw_price(dollar.group("raw")),
            UnitType.FLAT_ITEM,
            _round_price(_parse_money(dollar.group("price"))),
        )

    cents = _CENTS.search(blob)
    if cents:
        return (
            _normalize_raw_price(cents.group("raw")),
            UnitType.FLAT_ITEM,
            _round_price(int(cents.group("cents")) / 100.0),
        )

    return None


def _extract_clean_title(ocr_text: str) -> str:
    """Product title with price fragments and flyer noise stripped."""
    without_prices = _PRICE_STRIP.sub(" ", ocr_text)
    title = _collapse_ws(without_prices)
    title = re.sub(r"[\(\[\{]\s*[\)\]\}]", " ", title)
    title = re.sub(r"\s{2,}", " ", title)
    return title.strip(" -•|:;,./")


def _build_paddle_ocr() -> PaddleOCR:
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(lang="en", use_textline_orientation=True)
    except TypeError:
        try:
            return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except TypeError:
            return PaddleOCR(use_angle_cls=True, lang="en")


def _load_bgr(image_path: str) -> np.ndarray:
    import cv2
    import numpy as np

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read flyer page image: {image_path}")
    return image


def _crop_bgr(image: np.ndarray, bbox: list[float], pad: int = _CROP_PAD_PX) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(width, int(bbox[2]) + pad)
    y2 = min(height, int(bbox[3]) + pad)
    if (x2 - x1) < _MIN_CROP_PX or (y2 - y1) < _MIN_CROP_PX:
        return None
    return image[y1:y2, x1:x2]


def _is_classic_detection(item: object) -> bool:
    if not (isinstance(item, (list, tuple)) and len(item) >= 2):
        return False
    info = item[1]
    if isinstance(info, str):
        return True
    return isinstance(info, (list, tuple)) and bool(info) and isinstance(info[0], str)


def _ocr_lines_from_payload(raw: object) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, dict):
        for key in ("rec_texts", "rec_text", "texts"):
            value = raw.get(key)
            if value:
                return [str(line) for line in value if line]
        return []

    rec_texts = getattr(raw, "rec_texts", None)
    if rec_texts:
        return [str(line) for line in rec_texts if line]

    if not isinstance(raw, list) or not raw:
        return []

    first = raw[0]
    if isinstance(first, dict) or hasattr(first, "rec_texts"):
        lines: list[str] = []
        for item in raw:
            lines.extend(_ocr_lines_from_payload(item))
        return lines

    # PaddleOCR 2.x: either a page [[box, (text, conf)], ...] or [page].
    page = raw if _is_classic_detection(first) else first

    lines = []
    for item in page or []:
        if item is None:
            continue
        if isinstance(item, dict):
            text = item.get("text") or item.get("rec_text")
            if text:
                lines.append(str(text))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            info = item[1]
            if isinstance(info, (list, tuple)) and info:
                lines.append(str(info[0]))
            elif isinstance(info, str):
                lines.append(info)
    return lines


def _tile_id(store_id: str, bbox: list[float], title: str) -> str:
    payload = (
        f"{store_id}|{bbox[0]:.2f}|{bbox[1]:.2f}|{bbox[2]:.2f}|{bbox[3]:.2f}|{title}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


class FlyerParser:
    """Detect product tiles with YOLOv8, OCR each crop, and normalize prices."""

    def __init__(
        self,
        yolo_model: YOLO | None = None,
        ocr: PaddleOCR | None = None,
        *,
        yolo_weights: str = "yolov8n.pt",
        store_id: str | None = None,
        conf: float = 0.25,
    ) -> None:
        if yolo_model is not None:
            self.yolo = yolo_model
        else:
            from ultralytics import YOLO

            self.yolo = YOLO(yolo_weights)
        self.ocr = ocr if ocr is not None else _build_paddle_ocr()
        self.store_id = store_id
        self.conf = conf

    def parse_flyer_page(self, image_path: str) -> list[ProductTile]:
        """Detect tiles on a flyer page, OCR each crop, and return ProductTiles."""
        image = _load_bgr(image_path)
        store_id = self._resolve_store_id(image_path)
        return self._parse_bgr(image, store_id=store_id, source_key=image_path)

    def parse_flyer_pdf(self, pdf_path: str, dpi: int = 300) -> list[ProductTile]:
        """Rasterize a multi-page flyer PDF at ``dpi`` and parse every page."""
        import cv2
        import numpy as np
        from pdf2image import convert_from_path

        store_id = self._resolve_store_id(pdf_path)
        tiles: list[ProductTile] = []
        for index, page in enumerate(convert_from_path(pdf_path, dpi=dpi), start=1):
            rgb = np.asarray(page.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            tiles.extend(
                self._parse_bgr(
                    bgr,
                    store_id=store_id,
                    source_key=f"{pdf_path}#page{index}",
                )
            )
        return tiles

    def _resolve_store_id(self, source_path: str) -> str:
        if self.store_id:
            return self.store_id
        parent = Path(source_path).resolve().parent.name
        return parent or "unknown_store"

    def _parse_bgr(
        self,
        image: np.ndarray,
        *,
        store_id: str,
        source_key: str,
    ) -> list[ProductTile]:
        tiles: list[ProductTile] = []
        for bbox in self._detect_tile_boxes(image):
            crop = _crop_bgr(image, bbox)
            if crop is None:
                continue
            lines = self._recognize_text_lines(crop)
            if not lines:
                continue
            ocr_text = "\n".join(lines)
            parsed = extract_price_and_unit(ocr_text)
            title = _extract_clean_title(ocr_text)
            if parsed is None or not title:
                logger.debug("Skipping tile without title/price at %s bbox=%s", source_key, bbox)
                continue
            raw_price, unit_type, normalized_price = parsed
            tiles.append(
                ProductTile(
                    id=_tile_id(store_id, bbox, title),
                    store_id=store_id,
                    clean_title=title,
                    raw_price=raw_price,
                    unit_type=unit_type,
                    normalized_price=normalized_price,
                    bbox=bbox,
                )
            )
        return tiles

    def _detect_tile_boxes(self, image: np.ndarray) -> list[list[float]]:
        results = self.yolo.predict(image, conf=self.conf, verbose=False)
        if not results:
            return []
        result = results[0]
        names = result.names or {}
        known_aliases = {str(name).lower() for name in names.values()}
        restrict = bool(known_aliases & PRODUCT_TILE_ALIASES)
        boxes: list[list[float]] = []
        if result.boxes is None:
            return boxes
        for box in result.boxes:
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            label = str(names.get(cls_id, "")).lower()
            if restrict and label not in PRODUCT_TILE_ALIASES:
                continue
            xyxy = box.xyxy[0].tolist()
            boxes.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
        return boxes

    def _recognize_text_lines(self, tile_bgr: np.ndarray) -> list[str]:
        raw = None
        if hasattr(self.ocr, "predict"):
            try:
                raw = self.ocr.predict(tile_bgr)
            except TypeError:
                raw = None
            except Exception:
                logger.debug("PaddleOCR.predict failed; falling back to ocr()", exc_info=True)
                raw = None
        lines = _ocr_lines_from_payload(raw)
        if not lines and hasattr(self.ocr, "ocr"):
            try:
                raw = self.ocr.ocr(tile_bgr, cls=True)
            except TypeError:
                raw = self.ocr.ocr(tile_bgr)
            lines = _ocr_lines_from_payload(raw)
        return [line.strip() for line in lines if str(line).strip()]
