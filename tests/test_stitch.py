import numpy as np

from vscell.config import GridConfig
from vscell.grid import build_grid, plan_grid
from vscell.stitch import stitch_image


def test_adjacent_rankings_select_the_seam_candidate():
    shape = (10, 20)
    planning_mask = np.zeros(shape, dtype=np.uint16)
    planning_mask.ravel()[:10] = np.arange(1, 11, dtype=np.uint16)
    plan = plan_grid(shape, {name: planning_mask for name in "abcd"})
    assert (plan.rows, plan.cols) == (1, 2)
    patches = build_grid("img", shape, plan, GridConfig(padding_fraction=0.10))

    candidate_a = np.zeros(shape, dtype=np.int32)
    candidate_a[1:9, 8:12] = 1
    candidate_b = np.zeros(shape, dtype=np.int32)
    candidate_b[3:7, 9:11] = 2
    local_global = np.zeros(shape, dtype=np.int32)
    local_global[2:5, 2:5] = 1
    local_global[3:7, 8:12] = 2
    local_global[5:8, 15:18] = 3
    candidates = {
        "a": candidate_a,
        "b": candidate_b,
        "c": local_global,
        "d": local_global,
    }
    rankings = {
        patches[0].patch_id: ["a", "b", "c", "d"],
        patches[1].patch_id: ["b", "c", "a", "d"],
    }
    local_masks = {patch.patch_id: patch.crop(local_global) for patch in patches}

    stitched, audit = stitch_image(patches, local_masks, rankings, candidates)

    assert audit["stitch_policy"] == "adjacent_anchor_borda_seam_replacement"
    assert audit["seam_decisions"][0]["selected_model"] == "b"
    assert audit["seam_component_count"] == 1
    assert np.unique(stitched[candidate_b > 0]).tolist() == [1]
    assert not np.any(stitched[(candidate_a > 0) & (candidate_b == 0)])
