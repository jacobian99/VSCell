"""Candidate-density-aware grids with a 10% crop halo.

Candidate masks are used only to choose crop geometry. They do not select the
final segmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, log, sqrt
from statistics import median
from typing import Mapping

import numpy as np

from .config import GridConfig


class GridPlanningError(ValueError):
    """Raised before workspace generation when a crop plan is unsafe."""


@dataclass(frozen=True)
class CandidateCountSummary:
    """Robust per-image instance-count summary across candidate models."""

    per_model: dict[str, int]
    median_count: float
    minimum_count: int
    maximum_count: int
    disagreement_ratio: float | None
    disagreement_warning: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GridPlan:
    """A fully resolved, preflight-safe plan for one source image."""

    height: int
    width: int
    candidate_counts: CandidateCountSummary
    desired_core_count: int
    quality_rows: int
    quality_cols: int
    rows: int
    cols: int
    patch_count: int
    expected_cells_per_core: float
    core_height: int
    core_width: int
    core_area: int
    aspect_error: float
    limits_adjusted_grid: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["candidate_counts"] = self.candidate_counts.to_dict()
        return value


@dataclass(frozen=True)
class Patch:
    patch_id: str
    row: int
    col: int
    rows: int
    cols: int
    core_y0: int
    core_y1: int
    core_x0: int
    core_x1: int
    halo_y0: int
    halo_y1: int
    halo_x0: int
    halo_x1: int

    @property
    def core_shape(self) -> tuple[int, int]:
        return self.core_y1 - self.core_y0, self.core_x1 - self.core_x0

    @property
    def halo_shape(self) -> tuple[int, int]:
        return self.halo_y1 - self.halo_y0, self.halo_x1 - self.halo_x0

    @property
    def core_in_halo(self) -> tuple[slice, slice]:
        return (
            slice(self.core_y0 - self.halo_y0, self.core_y1 - self.halo_y0),
            slice(self.core_x0 - self.halo_x0, self.core_x1 - self.halo_x0),
        )

    def crop(self, array: np.ndarray) -> np.ndarray:
        return array[self.halo_y0 : self.halo_y1, self.halo_x0 : self.halo_x1]

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def instance_count(mask: np.ndarray) -> int:
    """Count non-background instance labels without assuming contiguous IDs."""
    labels = np.unique(np.asarray(mask))
    return int(np.count_nonzero(labels > 0))


def summarize_candidate_counts(
    candidate_masks: Mapping[str, np.ndarray],
    config: GridConfig = GridConfig(),
) -> CandidateCountSummary:
    if not candidate_masks:
        raise GridPlanningError("at least one candidate mask is required for grid planning")
    counts = {str(model): instance_count(mask) for model, mask in candidate_masks.items()}
    values = list(counts.values())
    low, high = min(values), max(values)
    ratio = None if low == 0 and high == 0 else (float("inf") if low == 0 else high / low)
    warning = None
    if ratio is not None and ratio >= config.count_disagreement_ratio:
        ratio_text = "infinite" if ratio == float("inf") else f"{ratio:.2f}"
        warning = (
            "candidate instance counts materially disagree "
            f"(min={low}, max={high}, ratio={ratio_text}); using the median"
        )
    return CandidateCountSummary(
        per_model=counts,
        median_count=float(median(values)),
        minimum_count=low,
        maximum_count=high,
        disagreement_ratio=ratio,
        disagreement_warning=warning,
    )


def _aspect_error(height: int, width: int, rows: int, cols: int) -> float:
    """Log-scale error between core side lengths; zero means square cores."""
    return abs(log((height * cols) / (width * rows)))


def _nearby_integers(value: float, radius: int = 3) -> set[int]:
    base = {floor(value), ceil(value), round(value)}
    return {
        number + offset
        for number in base
        for offset in range(-radius, radius + 1)
        if number + offset >= 1
    }


def _candidate_shapes(
    height: int,
    width: int,
    count: float,
    config: GridConfig,
) -> set[tuple[int, int]]:
    """Generate a bounded, deterministic set around viable occupancy endpoints."""
    min_cores = max(1, ceil(count / config.maximum_expected_cells_per_core))
    max_cores = max(1, floor(count / config.minimum_expected_cells_per_core))
    target_cores = max(1, ceil(count / config.target_cells_per_core))
    products = {min_cores, max_cores, target_cores}
    candidates: set[tuple[int, int]] = {(1, 1)}

    # An exactly square source image uses a square grid. Occupancy then decides
    # among those aspect-faithful choices.
    if height == width:
        for product in products:
            for side in _nearby_integers(sqrt(product)):
                candidates.add((side, side))
        return {(min(rows, height), min(cols, width)) for rows, cols in candidates}

    ratio = height / width
    for product in products:
        continuous_rows = sqrt(product * ratio)
        continuous_cols = sqrt(product / ratio)
        for rows in _nearby_integers(continuous_rows):
            for cols in _nearby_integers(continuous_cols):
                candidates.add((rows, cols))
    return {(min(rows, height), min(cols, width)) for rows, cols in candidates}


def _occupancy_violation(occupancy: float, config: GridConfig) -> float:
    if occupancy < config.minimum_expected_cells_per_core:
        return config.minimum_expected_cells_per_core - occupancy
    if occupancy > config.maximum_expected_cells_per_core:
        return occupancy - config.maximum_expected_cells_per_core
    return 0.0


def _quality_grid(
    height: int,
    width: int,
    count: float,
    config: GridConfig,
) -> tuple[int, int]:
    candidates = _candidate_shapes(height, width, count, config)
    if not candidates:
        raise GridPlanningError("could not generate a valid grid candidate")

    def score(shape: tuple[int, int]) -> tuple[float, float, float, int, int, int]:
        rows, cols = shape
        occupancy = count / (rows * cols)
        return (
            _occupancy_violation(occupancy, config),
            _aspect_error(height, width, rows, cols),
            abs(occupancy - config.target_cells_per_core),
            abs(rows * cols - ceil(count / config.target_cells_per_core)),
            rows * cols,
            rows,
        )

    return min(candidates, key=score)


def _max_core_shape(height: int, width: int, rows: int, cols: int) -> tuple[int, int, int]:
    core_height = ceil(height / rows)
    core_width = ceil(width / cols)
    return core_height, core_width, core_height * core_width


def _apply_pixel_limits(
    height: int,
    width: int,
    rows: int,
    cols: int,
    config: GridConfig,
) -> tuple[int, int]:
    """Increase axes only until every core fits all configured pixel limits."""
    rows = max(rows, ceil(height / config.max_core_height))
    cols = max(cols, ceil(width / config.max_core_width))
    while True:
        _core_height, _core_width, area = _max_core_shape(height, width, rows, cols)
        if area <= config.max_core_area:
            return rows, cols
        options = []
        if rows < height:
            options.append((rows + 1, cols))
        if cols < width:
            options.append((rows, cols + 1))
        if not options:
            raise GridPlanningError("cannot satisfy max_core_area with one-pixel cores")
        rows, cols = min(
            options,
            key=lambda item: (
                _aspect_error(height, width, *item),
                item[0] * item[1],
                item[0],
            ),
        )


def plan_grid(
    shape: tuple[int, int],
    candidate_masks: Mapping[str, np.ndarray],
    config: GridConfig = GridConfig(),
) -> GridPlan:
    """Plan one image after all full-image candidate masks are available."""
    height, width = map(int, shape)
    if height < 1 or width < 1:
        raise GridPlanningError("image shape must be positive")
    if any(np.asarray(mask).shape != (height, width) for mask in candidate_masks.values()):
        raise GridPlanningError("candidate mask shape differs from source image")
    counts = summarize_candidate_counts(candidate_masks, config)
    quality_rows, quality_cols = _quality_grid(
        height, width, counts.median_count, config
    )
    rows, cols = _apply_pixel_limits(height, width, quality_rows, quality_cols, config)
    patch_count = rows * cols
    if patch_count > config.max_patch_count:
        raise GridPlanningError(
            "planned patch count "
            f"{patch_count} exceeds max_patch_count={config.max_patch_count}; "
            "adjust crop limits/settings or use an out-of-core/whole-slide workflow"
        )
    core_height, core_width, core_area = _max_core_shape(height, width, rows, cols)
    return GridPlan(
        height=height,
        width=width,
        candidate_counts=counts,
        desired_core_count=ceil(counts.median_count / config.target_cells_per_core),
        quality_rows=quality_rows,
        quality_cols=quality_cols,
        rows=rows,
        cols=cols,
        patch_count=patch_count,
        expected_cells_per_core=counts.median_count / patch_count,
        core_height=core_height,
        core_width=core_width,
        core_area=core_area,
        aspect_error=_aspect_error(height, width, rows, cols),
        limits_adjusted_grid=(rows, cols) != (quality_rows, quality_cols),
    )


def grid_shape(
    shape: tuple[int, int],
    candidate_masks: Mapping[str, np.ndarray],
    config: GridConfig = GridConfig(),
) -> tuple[int, int]:
    """Return the candidate-derived row/column counts for one image."""
    plan = plan_grid(shape, candidate_masks, config)
    return plan.rows, plan.cols


def build_grid(
    image_id: str,
    shape: tuple[int, int],
    plan: GridPlan,
    config: GridConfig = GridConfig(),
) -> list[Patch]:
    """Partition a planned image into cores and add the configured crop halo."""
    height, width = map(int, shape)
    if (height, width) != (plan.height, plan.width):
        raise GridPlanningError("grid plan shape differs from source image")
    y_edges = np.rint(np.linspace(0, height, plan.rows + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, width, plan.cols + 1)).astype(int)
    patches: list[Patch] = []
    for row in range(plan.rows):
        for col in range(plan.cols):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            # A 10% total enlargement adds approximately 5% on each side.
            pad_y = (y1 - y0) * config.padding_fraction / 2.0
            pad_x = (x1 - x0) * config.padding_fraction / 2.0
            hy0 = max(0, int(np.floor(y0 - pad_y)))
            hy1 = min(height, int(np.ceil(y1 + pad_y)))
            hx0 = max(0, int(np.floor(x0 - pad_x)))
            hx1 = min(width, int(np.ceil(x1 + pad_x)))
            patches.append(
                Patch(
                    patch_id=f"{image_id}__r{row:03d}_c{col:03d}",
                    row=row,
                    col=col,
                    rows=plan.rows,
                    cols=plan.cols,
                    core_y0=y0,
                    core_y1=y1,
                    core_x0=x0,
                    core_x1=x1,
                    halo_y0=hy0,
                    halo_y1=hy1,
                    halo_x0=hx0,
                    halo_x1=hx1,
                )
            )
    return patches


def validate_partition(patches: list[Patch], shape: tuple[int, int]) -> None:
    coverage = np.zeros(shape, dtype=np.uint16)
    for patch in patches:
        coverage[
            patch.core_y0 : patch.core_y1, patch.core_x0 : patch.core_x1
        ] += 1
    if not np.all(coverage == 1):
        values, counts = np.unique(coverage, return_counts=True)
        raise RuntimeError(
            "grid cores do not partition image exactly: "
            f"{dict(zip(values.tolist(), counts.tolist()))}"
        )
