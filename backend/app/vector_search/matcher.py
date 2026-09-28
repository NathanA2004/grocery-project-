"""Module 2: semantic item matching with local MiniLM embeddings and FAISS."""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..schemas import ProductTile, UnitType

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 emits one 384-d vector per string. The index is sized to that.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Natural-language unit phrases so the embedding sees the same kind of words a
# shopper would use, instead of the raw UnitType token.
_UNIT_CONTEXT: dict[UnitType, str] = {
    UnitType.PER_LB: "priced per pound",
    UnitType.PER_KG: "priced per kilogram",
    UnitType.FLAT_ITEM: "priced per item",
    UnitType.MULTI_BUY: "multi-buy deal",
}


def build_search_string(tile: ProductTile) -> str:
    """Build the document string embedded for one flyer tile.

    The title carries the match. Unit and price text are appended so deals
    like ``2 for $5.00`` and ``$3.99/lb`` stay attached to the product, and
    ``store_id`` keeps the tile's store in the same string without a second
    index.
    """
    unit_phrase = _UNIT_CONTEXT[tile.unit_type]
    return f"{tile.clean_title}. {unit_phrase}. {tile.raw_price}. store {tile.store_id}"


class SemanticMatcher:
    """Local cosine search over flyer tiles.

    ``sentence-transformers/all-MiniLM-L6-v2`` runs in-process. Vectors are
    L2-normalized and stored in ``faiss.IndexFlatIP``, whose inner product is
    cosine similarity. Position ``i`` in ``_tiles`` is the ``ProductTile`` that
    produced FAISS label ``i``.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Weights load from the local Hugging Face cache (downloaded once).
        # Queries are never sent to a hosted embedding API.
        self._model = SentenceTransformer(model_name)
        dim = int(self._model.get_sentence_embedding_dimension())
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} reported embedding dimension {dim}, expected {EMBEDDING_DIM}."
            )
        # Inner-product index. Exact search; no training step and no IVF buckets.
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        # Parallel to FAISS labels. _tiles[i] is the tile stored as vector i.
        self._tiles: list[ProductTile] = []
        # ProductTile.id -> FAISS label. Blocks a second insert of the same id
        # from shifting later labels away from the tiles already stored.
        self._position_by_id: dict[str, int] = {}

    def index_flyer_items(self, items: list[ProductTile]) -> None:
        """Embed ``items`` and append them to the FAISS index.

        Already indexed ``ProductTile.id`` values are skipped so a second pass
        over the same flyer does not duplicate vectors or break the id map.
        """
        fresh: list[ProductTile] = []
        seen_in_batch: set[str] = set()
        for tile in items:
            if tile.id in self._position_by_id or tile.id in seen_in_batch:
                logger.debug("Skipping already indexed tile id=%s", tile.id)
                continue
            seen_in_batch.add(tile.id)
            fresh.append(tile)
        if not fresh:
            return

        texts = [build_search_string(tile) for tile in fresh]
        vectors = self._embed_normalized(texts)

        # IndexFlatIP assigns contiguous integer labels in insertion order.
        # The next add() starts at the current ntotal, so vector 0 of this
        # batch becomes label `start`, vector 1 becomes `start + 1`, and so on.
        # Recording that start label *before* add() is what keeps the Python
        # list aligned with those labels even after earlier batches.
        start = int(self._index.ntotal)
        self._index.add(vectors)
        for offset, tile in enumerate(fresh):
            label = start + offset
            self._tiles.append(tile)
            self._position_by_id[tile.id] = label

        # A successful add grows ntotal by exactly len(fresh). If that ever
        # disagrees with the Python map, later searches would return a label
        # that points at the wrong tile (or past the end of _tiles).
        if len(self._tiles) != int(self._index.ntotal) or len(self._position_by_id) != len(self._tiles):
            raise RuntimeError(
                "FAISS index and ProductTile map diverged: "
                f"ntotal={self._index.ntotal} tiles={len(self._tiles)} "
                f"ids={len(self._position_by_id)}."
            )
        logger.info("Indexed %d flyer tiles (index size %d).", len(fresh), self._index.ntotal)

    def find_item_matches(
        self,
        user_items: list[str],
        top_k: int = 5,
        threshold: float = 0.60,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return tiled matches for each shopping-list string.

        Each query maps to up to ``top_k`` hits whose cosine similarity is at
        least ``threshold``. Hits are ordered best-first. A query with no hit
        above the threshold maps to an empty list.

        Each hit is ``{"product": ProductTile, "score": float}``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        matches: dict[str, list[dict[str, Any]]] = {query: [] for query in user_items}
        ntotal = int(self._index.ntotal)
        if not user_items or ntotal == 0:
            return matches

        # Asking for more neighbors than there are vectors makes FAISS pad the
        # row with label -1. Clamp k so every returned label is a real tile.
        k = min(top_k, ntotal)
        queries = self._embed_normalized(list(user_items))
        scores, labels = self._index.search(queries, k)

        for query, row_scores, row_labels in zip(user_items, scores, labels, strict=True):
            hits: list[dict[str, Any]] = []
            for score, label in zip(row_scores, row_labels, strict=True):
                label_i = int(label)
                # -1 is FAISS's "no neighbor" pad. Anything outside _tiles means
                # the index and the map are no longer the same length.
                if label_i < 0 or label_i >= len(self._tiles):
                    continue
                similarity = float(score)
                # IndexFlatIP on unit vectors returns cosine in [-1, 1], already
                # sorted high to low. The first score under the cutoff ends the row.
                if similarity < threshold:
                    break
                hits.append({"product": self._tiles[label_i], "score": similarity})
            matches[query] = hits
        return matches

    def match_items(
        self,
        user_query_list: list[str],
        top_k: int = 5,
        threshold: float = 0.60,
    ) -> dict[str, list[dict[str, Any]]]:
        """Spec name for :meth:`find_item_matches`."""
        return self.find_item_matches(user_query_list, top_k=top_k, threshold=threshold)

    def _embed_normalized(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` and L2-normalize each row.

        Cosine similarity is ``(a · b) / (||a|| ||b||)``. ``IndexFlatIP`` only
        computes the inner product ``a · b``. Dividing every row by its L2 norm
        forces ``||a|| = ||b|| = 1``, so that inner product *is* the cosine.

        ``normalize_embeddings`` is left off on the encoder so the only
        normalization pass is ``faiss.normalize_L2``, which rewrites the
        float32 matrix in place. A zero-length row is left as zeros; its
        cosine with any query is 0 and cannot clear the match threshold.
        """
        encoded = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # FAISS rejects float64 and non-contiguous memory. Copy into a C-order
        # float32 matrix so normalize_L2 cannot alias the encoder's buffer.
        vectors = np.array(encoded, dtype=np.float32, copy=True, order="C")
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Expected embeddings of shape (n, {EMBEDDING_DIM}), got {vectors.shape}."
            )
        faiss.normalize_L2(vectors)
        return vectors
