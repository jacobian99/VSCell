import numpy as np
import pytest

from vscell.config import GridConfig
from vscell.grid import GridPlanningError, build_grid, plan_grid, validate_partition


def _masks(shape, counts):
    result = {}
    for model, count in enumerate(counts):
        mask = np.zeros(shape, dtype=np.uint16)
        mask.ravel()[:count] = np.arange(1, count + 1, dtype=np.uint16)
        result[f"model_{model}"] = mask
    return result


def test_square_examples_use_square_grids_and_target_five_cells_per_core():
    cases = (
        ((44, 48, 63, 39), (3, 3)),
        ((43, 49, 49, 44), (3, 3)),
        ((142, 180, 166, 154), (6, 6)),
        ((52, 61, 69, 50), (3, 3)),
        ((117, 109, 131, 123), (5, 5)),
    )
    for counts, expected in cases:
        plan = plan_grid((256, 256), _masks((256, 256), counts))
        assert (plan.rows, plan.cols) == expected
        assert 3 <= plan.expected_cells_per_core <= 10


def test_rectangular_image_prioritizes_aspect_within_occupancy_range():
    plan = plan_grid((100, 1000), _masks((100, 1000), (90, 90, 90, 90)))
    assert (plan.rows, plan.cols) == (1, 10)
    assert plan.expected_cells_per_core == 9
    assert plan.core_height == plan.core_width == 100


def test_reports_material_candidate_count_disagreement():
    plan = plan_grid((100, 100), _masks((100, 100), (10, 10, 10, 40)))
    assert plan.candidate_counts.median_count == 10
    assert plan.candidate_counts.disagreement_warning is not None


def test_pixel_limits_only_add_splits_and_patch_limit_fails_fast():
    masks = _masks((1000, 1000), (5, 5, 5, 5))
    plan = plan_grid(
        (1000, 1000),
        masks,
        GridConfig(max_core_height=100, max_core_width=100, max_patch_count=100),
    )
    assert (plan.quality_rows, plan.quality_cols) == (1, 1)
    assert (plan.rows, plan.cols) == (10, 10)
    assert plan.limits_adjusted_grid is True
    with pytest.raises(GridPlanningError, match="max_patch_count"):
        plan_grid(
            (1000, 1000),
            masks,
            GridConfig(max_core_height=100, max_core_width=100, max_patch_count=99),
        )


def test_padding_and_exact_core_partition():
    plan = plan_grid((40, 40), _masks((40, 40), (80, 80, 80, 80)))
    assert (plan.rows, plan.cols) == (4, 4)
    patches = build_grid("image", (40, 40), plan, GridConfig(padding_fraction=0.10))
    validate_partition(patches, (40, 40))
    middle = next(patch for patch in patches if patch.row == 1 and patch.col == 1)
    assert (middle.core_y0, middle.core_y1) == (10, 20)
    assert (middle.halo_y0, middle.halo_y1) == (9, 21)
    assert middle.crop(np.zeros((40, 40))).shape == middle.halo_shape
