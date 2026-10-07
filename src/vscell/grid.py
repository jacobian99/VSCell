"""GT-free adaptive grid generation with the historical 10% crop halo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Iterable

import numpy as np

from .config import GridConfig


@dataclass(frozen=True)
class GridReference:
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("reference dimensions must be positive")


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


def infer_grid_reference(shapes: Iterable[tuple[int, int]]) -> GridReference:
    rows = [(int(height), int(width)) for height, width in shapes]
    if not rows:
        raise ValueError("cannot infer a grid reference from an empty dataset")
    if any(height < 1 or width < 1 for height, width in rows):
        raise ValueError("all image dimensions must be positive")
    return GridReference(
        height=max(1, int(round(median(height for height, _ in rows)))),
        width=max(1, int(round(median(width for _, width in rows)))),
    )


def _axis_tiles(
    dimension: int,
    reference: int,
    base_tiles: int,
    lower_deadband: float,
    upper_deadband: float,
) -> int:
    ratio = dimension / reference
    if lower_deadband <= ratio <= upper_deadband:
        return base_tiles
    return max(1, int(round(base_tiles * ratio)))


def grid_shape(
    shape: tuple[int, int],
    reference: GridReference,
    config: GridConfig = GridConfig(),
) -> tuple[int, int]:
    height, width = map(int, shape)
    rows = _axis_tiles(
        height,
        reference.height,
        config.base_rows,
        config.lower_deadband,
        config.upper_deadband,
    )
    cols = _axis_tiles(
        width,
        reference.width,
        config.base_cols,
        config.lower_deadband,
        config.upper_deadband,
    )
    return min(rows, height), min(cols, width)


def build_grid(
    image_id: str,
    shape: tuple[int, int],
    reference: GridReference,
    config: GridConfig = GridConfig(),
) -> list[Patch]:
    """Partition an image into disjoint cores and add a 10%-total halo.

    Historical ``PAD=0.10`` enlarged each crop side by 10% in total, which is
    approximately 5% on each side. Rounding uses floor on the leading side and
    ceil on the trailing side, clipped to the image.
    """
    height, width = map(int, shape)
    if height < 1 or width < 1:
        raise ValueError("image shape must be positive")
    rows, cols = grid_shape((height, width), reference, config)
    y_edges = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    x_edges = np.rint(np.linspace(0, width, cols + 1)).astype(int)
    patches: list[Patch] = []
    for row in range(rows):
        for col in range(cols):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            # An enlargement of p in total adds p/2 to each side.
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
                    rows=rows,
                    cols=cols,
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
            f"grid cores do not partition image exactly: {dict(zip(values.tolist(), counts.tolist()))}"
        )
