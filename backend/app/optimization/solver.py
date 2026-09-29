"""Module 3: multi-store grocery selection with a PuLP MILP.

The model chooses one flyer tile per shopping-list line and decides which
stores are worth visiting. Travel is a round trip from home: each visited
store is charged ``2 * distance * cost_per_km``, once, when a chosen tile
is sold there.
"""

from __future__ import annotations

import logging
from typing import Any

import pulp
from pydantic import ValidationError

from ..schemas import OptimizationRequest, OptimizationResult, ProductTile

logger = logging.getLogger(__name__)

# CBC writes 1 as 0.999999. A binary is selected when its value is nearer to 1.
_SELECTED = 0.5

# Fallback solves may leave a line unbought. The penalty weight is at least
# this large, and always larger than the price-plus-travel that line could
# save, so dropping a buyable line never improves the objective.
_UNFULFILLED_PENALTY_FLOOR = 10_000.0


class OptimizationSolver:
    """Assign each shopping-list line to a flyer tile and each used tile to a visit.

    ``matched_items`` is the Module 2 result: each query string maps to hits of
    the form ``{"product": ProductTile, "score": float}``, best score first.
    Similarity scores are not part of the objective. Price and travel cost are.
    """

    def __init__(
        self,
        matched_items: dict[str, list[dict[str, Any]]],
        request: OptimizationRequest,
    ) -> None:
        self._matched_items = matched_items
        self._request = request

    def solve(self) -> OptimizationResult:
        """Minimize grocery price plus store-visit cost and return the basket.

        Lines with at least one usable tile are forced through an exact
        assignment first. If ``max_stores`` makes that model infeasible, a
        second model may leave some of those lines unbought. Lines with no
        usable tile are reported in ``unfulfilled_items`` and never enter the
        MILP.
        """
        options = self._candidate_tiles()
        if not options:
            return self._build_result({})

        chosen = self._optimize(options, allow_unfulfilled=False)
        if chosen is None:
            logger.info(
                "Exact assignment is infeasible with max_stores=%s; solving a partial basket.",
                self._request.max_stores,
            )
            chosen = self._optimize(options, allow_unfulfilled=True)
        if chosen is None:
            raise RuntimeError(
                "The grocery MILP has no feasible basket even after leaving items unfulfilled."
            )
        return self._build_result(chosen)

    def _candidate_tiles(self) -> dict[int, list[ProductTile]]:
        """Usable tiles for each shopping-list index that has at least one.

        The same query string can appear twice. Each index is a separate unit
        to buy, and both indexes read the same Module 2 hit list. A product id
        is kept once per line; the match list is best-score-first, so the first
        copy is the one that stays.
        """
        options: dict[int, list[ProductTile]] = {}
        for index, query in enumerate(self._request.items):
            tiles = self._tiles_for_query(query)
            if tiles:
                options[index] = tiles
        return options

    def _tiles_for_query(self, query: str) -> list[ProductTile]:
        hits = self._matched_items.get(query, [])
        if not isinstance(hits, list):
            logger.warning("Matches for %r are not a list; the line is unmatched.", query)
            return []

        tiles: list[ProductTile] = []
        seen_ids: set[str] = set()
        for hit in hits:
            tile = _tile_from_hit(hit)
            if tile is None or tile.id in seen_ids:
                continue
            if tile.store_id not in self._request.store_distances:
                logger.warning(
                    "No home distance for store %s; skipping tile %s.",
                    tile.store_id,
                    tile.id,
                )
                continue
            seen_ids.add(tile.id)
            tiles.append(tile)
        return tiles

    def _optimize(
        self,
        options: dict[int, list[ProductTile]],
        *,
        allow_unfulfilled: bool,
    ) -> dict[int, ProductTile] | None:
        """Solve one MILP. Return the chosen tile per line, or None if infeasible."""
        store_ids = sorted({tile.store_id for tiles in options.values() for tile in tiles})
        travel_cost = {store_id: self._visit_cost(store_id) for store_id in store_ids}

        # Variables have to be created on the LpProblem. PuLP 4 rejects a
        # standalone LpVariable; earlier releases attach the variable when it
        # first appears in the objective or a constraint.
        problem = pulp.LpProblem("multi_store_grocery", pulp.LpMinimize)

        # x_{j,i} is 1 when shopping-list line j is fulfilled by candidate i.
        # Candidate i is one Module 2 tile for that line (a product at a store,
        # carrying that tile's normalized_price). One line can set only one of
        # its x variables (see the fulfillment rows below). Two lines may both
        # select the same product id: that buys two units of the flyer price.
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        for line, tiles in options.items():
            for candidate, _tile in enumerate(tiles):
                x[line, candidate] = _binary(problem, f"x_{line}_{candidate}")

        # y_s is 1 when the trip visits store s. A visit is charged Travel_s
        # once, regardless of how many tiles are bought there.
        y = {
            store_id: _binary(problem, f"y_{index}")
            for index, store_id in enumerate(store_ids)
        }

        # u_j is 1 when line j is left unbought. It exists only in the fallback
        # model, after an exact assignment has already been proven infeasible.
        u: dict[int, pulp.LpVariable] = {}
        if allow_unfulfilled:
            u = {line: _binary(problem, f"u_{line}") for line in options}

        sold_at: dict[str, list[tuple[int, int]]] = {store_id: [] for store_id in store_ids}
        for line, tiles in options.items():
            for candidate, tile in enumerate(tiles):
                sold_at[tile.store_id].append((line, candidate))

        # Minimize Total Cost =
        #   sum_{j,i} normalized_price_{j,i} * x_{j,i}
        #   + sum_s Travel_s * y_s
        # where Travel_s = 2 * distance_s * cost_per_km (home to the store and back).
        #
        # The fallback model adds a high penalty weight P * u_j for each line.
        # P is larger than the price-plus-travel a single line can possibly save,
        # and at least _UNFULFILLED_PENALTY_FLOOR, so that remains true for any
        # set of lines. A feasible exact basket always beats a basket that drops
        # something. A line is dropped only when max_stores makes buying it
        # impossible. P is not included in the returned costs.
        grocery_terms = [
            tiles[candidate].normalized_price * x[line, candidate]
            for line, tiles in options.items()
            for candidate in range(len(tiles))
        ]
        travel_terms = [travel_cost[store_id] * y[store_id] for store_id in store_ids]
        penalty_terms: list[pulp.LpAffineExpression] = []
        if allow_unfulfilled:
            penalty = _unfulfilled_penalty(options, travel_cost)
            penalty_terms = [penalty * u[line] for line in options]
        problem += pulp.lpSum([*grocery_terms, *travel_terms, *penalty_terms])

        for line, tiles in options.items():
            purchased = pulp.lpSum(x[line, candidate] for candidate in range(len(tiles)))
            if allow_unfulfilled:
                # sum_i x_{j,i} + u_j = 1. The line is bought once, or it is
                # recorded as unfulfilled. It is never bought twice.
                problem += purchased + u[line] == 1, f"fulfill_{line}"
            else:
                # sum_i x_{j,i} = 1. Every line that has a usable tile is bought
                # exactly once. Among those tiles, the objective keeps the one
                # whose price plus the visits it forces is cheapest.
                problem += purchased == 1, f"fulfill_{line}"

        for store_id, pairs in sold_at.items():
            for line, candidate in pairs:
                # y_s >= x_{j,i} for every tile i of line j sold at store s.
                # Selecting that tile forces the visit variable on, which is
                # what adds Travel_s to the objective.
                problem += y[store_id] >= x[line, candidate], f"open_{line}_{candidate}"
            # y_s <= sum of those x_{j,i}. The visit is on only when at least
            # one chosen tile is sold at s, so an unused store cannot take a
            # slot under max_stores (even when its travel cost is zero).
            problem += (
                y[store_id] <= pulp.lpSum(x[line, candidate] for line, candidate in pairs),
                f"close_{store_ids.index(store_id)}",
            )

        if self._request.max_stores is not None:
            # sum_s y_s <= max_stores. The number of stores with a chosen tile
            # cannot exceed the trip limit.
            problem += (
                pulp.lpSum(y[store_id] for store_id in store_ids) <= self._request.max_stores,
                "max_stores",
            )

        outcome = problem.solve(_milp_solver())
        kind = _outcome_kind(outcome)
        if kind == "error":
            status = getattr(outcome, "status", outcome)
            raise RuntimeError(f"Grocery MILP failed with status {status}.")
        if kind != "optimal":
            return None

        chosen: dict[int, ProductTile] = {}
        for line, tiles in options.items():
            for candidate, tile in enumerate(tiles):
                raw = pulp.value(x[line, candidate])
                if raw is not None and float(raw) > _SELECTED:
                    chosen[line] = tile
                    break

        if not allow_unfulfilled and set(chosen) != set(options):
            return None
        return chosen

    def _visit_cost(self, store_id: str) -> float:
        """Round-trip travel charge for store ``store_id``.

        Distance on the request is one way, from home to the store. Driving
        there and back is ``2 * distance * cost_per_km``.
        """
        distance = self._request.store_distances[store_id]
        return 2.0 * distance * self._request.cost_per_km

    def _build_result(self, chosen: dict[int, ProductTile]) -> OptimizationResult:
        """Rebuild money totals from the chosen tiles.

        The solver objective may contain the unfulfilled-line penalty. That
        penalty is a modeling device, so grocery_cost and travel_cost are
        summed again from the assignment the variables actually selected.
        """
        itinerary: dict[str, list[ProductTile]] = {}
        grocery_cost = 0.0
        for line in sorted(chosen):
            tile = chosen[line]
            itinerary.setdefault(tile.store_id, []).append(tile)
            grocery_cost += tile.normalized_price

        travel_cost = sum(self._visit_cost(store_id) for store_id in itinerary)
        unfulfilled = [
            item
            for index, item in enumerate(self._request.items)
            if index not in chosen
        ]
        return OptimizationResult(
            total_cost=grocery_cost + travel_cost,
            grocery_cost=grocery_cost,
            travel_cost=travel_cost,
            store_itinerary=itinerary,
            unfulfilled_items=unfulfilled,
        )


def _unfulfilled_penalty(
    options: dict[int, list[ProductTile]],
    travel_cost: dict[str, float],
) -> float:
    """High penalty weight so dropping a buyable line never improves the objective.

    Dropping one line saves at most its dearest candidate plus the travel of
    every store that candidate list touches. The penalty is one more than the
    largest such saving, and never below ``_UNFULFILLED_PENALTY_FLOOR``.
    Dropping a set of lines saves at most the sum of those per-line savings,
    which is still less than charging the penalty once per dropped line. The
    solver therefore fulfills every line it is allowed to. This weight is
    applied only on the fallback solve, after an exact assignment is infeasible.
    """
    largest_saving = 0.0
    for tiles in options.values():
        dearest = max(tile.normalized_price for tile in tiles)
        visits = sum(travel_cost[store_id] for store_id in {tile.store_id for tile in tiles})
        largest_saving = max(largest_saving, dearest + visits)
    return max(largest_saving + 1.0, _UNFULFILLED_PENALTY_FLOOR)


def _tile_from_hit(hit: object) -> ProductTile | None:
    """Read the ProductTile out of one Module 2 hit, or skip a malformed hit."""
    if not isinstance(hit, dict):
        logger.warning("Ignoring a match entry that is not a dict.")
        return None
    product = hit.get("product")
    if isinstance(product, ProductTile):
        return product
    if isinstance(product, dict):
        try:
            return ProductTile.model_validate(product)
        except ValidationError:
            logger.warning("Ignoring a match entry with an invalid product payload.")
            return None
    logger.warning("Ignoring a match entry with no product.")
    return None


def _binary(problem: pulp.LpProblem, name: str) -> pulp.LpVariable:
    """Create a 0/1 variable named ``name`` on ``problem``."""
    add = getattr(problem, "add_variable", None)
    if add is not None:
        return add(name, lowBound=0, upBound=1, cat=pulp.LpBinary)
    return pulp.LpVariable(name, lowBound=0, upBound=1, cat=pulp.LpBinary)


def _milp_solver() -> pulp.LpSolver:
    """CBC when the CBC binary is installed, otherwise HiGHS or GLPK."""
    for name in ("PULP_CBC_CMD", "COIN_CMD", "HiGHS", "GLPK_CMD"):
        factory = getattr(pulp, name, None)
        if factory is None:
            continue
        try:
            solver = factory(msg=False)
        except TypeError:
            solver = factory()
        available = getattr(solver, "available", None)
        if available is None or available():
            return solver
    raise pulp.PulpError(
        "No MILP solver is available for PuLP. "
        "Install CBC (pip install 'pulp[cbc]') or HiGHS (pip install highspy)."
    )


def _outcome_kind(outcome: object) -> str:
    """Map a PuLP 2 status code or a PuLP 4 ``LpSolveStats`` to a short label."""
    if isinstance(outcome, int):
        if outcome == 1:
            return "optimal"
        return "infeasible" if outcome == -1 else "error"
    status = getattr(outcome, "status", None)
    has_solution = bool(getattr(outcome, "has_solution", False))
    if status == pulp.LpSolveStatus.Optimal and has_solution:
        return "optimal"
    if status == pulp.LpSolveStatus.Infeasible or not has_solution:
        return "infeasible"
    return "error"
