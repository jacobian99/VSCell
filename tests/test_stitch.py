import numpy as np

from vscell.config import GridConfig
from vscell.grid import GridReference, build_grid
from vscell.stitch import stitch_image


def test_cross_core_cells_use_image_level_best_model():
    shape = (10, 20)
    patches = build_grid(
        "img",
        shape,
        GridReference(*shape),
        GridConfig(base_rows=1, base_cols=2, padding_fraction=0.10),
    )
    best = np.zeros(shape, dtype=np.int32)
    best[3:7, 8:12] = 1  # crosses the two cores
    local_global = np.zeros(shape, dtype=np.int32)
    local_global[2:5, 2:5] = 1
    local_global[3:7, 8:12] = 2
    local_global[5:8, 15:18] = 3
    candidates = {
        "a": best,
        "b": local_global,
        "c": local_global,
        "d": local_global,
    }
    rankings = {patch.patch_id: ["a", "b", "c", "d"] for patch in patches}
    local_masks = {patch.patch_id: patch.crop(local_global) for patch in patches}
    stitched, audit = stitch_image(patches, local_masks, rankings, candidates)
    assert int(stitched.max()) == 3
    assert audit["overlap_model"] == "a"
    assert audit["cross_core_cells_written"] == 1
    assert np.unique(stitched[best > 0]).tolist() == [1]
