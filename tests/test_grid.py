import numpy as np

from vscell.config import GridConfig
from vscell.grid import GridReference, build_grid, grid_shape, validate_partition


def test_adaptive_grid_examples():
    reference = GridReference(40, 40)
    assert grid_shape((50, 50), reference) == (4, 4)
    assert grid_shape((120, 120), reference) == (12, 12)
    assert grid_shape((80, 20), reference) == (8, 2)


def test_historical_padding_and_exact_core_partition():
    config = GridConfig(base_rows=4, base_cols=4, padding_fraction=0.10)
    patches = build_grid("image", (40, 40), GridReference(40, 40), config)
    validate_partition(patches, (40, 40))
    middle = next(patch for patch in patches if patch.row == 1 and patch.col == 1)
    assert (middle.core_y0, middle.core_y1) == (10, 20)
    assert (middle.halo_y0, middle.halo_y1) == (9, 21)
    assert middle.crop(np.zeros((40, 40))).shape == middle.halo_shape
