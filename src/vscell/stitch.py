"""Stitch local VSCell masks into a globally labelled full-image mask."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from .grid import Patch, validate_partition
from .io import relabel_instances


STRUCTURE = np.ones((3, 3), dtype=bool)


def _core_membership(patches: Sequence[Patch], shape: tuple[int, int]) -> np.ndarray:
    membership = np.full(shape, -1, dtype=np.int32)
    for index, patch in enumerate(patches):
        region = membership[
            patch.core_y0 : patch.core_y1, patch.core_x0 : patch.core_x1
        ]
        if np.any(region != -1):
            raise ValueError("patch cores overlap")
        region[:] = index
    if np.any(membership < 0):
        raise ValueError("patch cores do not cover the image")
    return membership


def _rank_points(ranking: Sequence[str], model: str) -> int:
    try:
        return len(ranking) - 1 - list(ranking).index(model)
    except ValueError as error:
        raise ValueError(f"model {model!r} is absent from a patch ranking") from error


def _choose_adjacent_anchor(
    first: Patch,
    second: Patch,
    patch_rankings: dict[str, Sequence[str]],
) -> tuple[str, dict[str, object]]:
    """Choose between the two adjacent patches' top-ranked candidates."""
    first_ranking = list(patch_rankings[first.patch_id])
    second_ranking = list(patch_rankings[second.patch_id])
    if not first_ranking or not second_ranking:
        raise ValueError("patch rankings must be nonempty")
    first_anchor, second_anchor = first_ranking[0], second_ranking[0]
    first_score = _rank_points(first_ranking, first_anchor) + _rank_points(
        second_ranking, first_anchor
    )
    second_score = _rank_points(first_ranking, second_anchor) + _rank_points(
        second_ranking, second_anchor
    )
    if first_score > second_score:
        selected, reason = first_anchor, "higher_adjacent_borda_score"
    elif second_score > first_score:
        selected, reason = second_anchor, "higher_adjacent_borda_score"
    else:
        selected, reason = min(first_anchor, second_anchor), "equal_score_stable_tiebreak"
    return selected, {
        "patches": [first.patch_id, second.patch_id],
        "anchors": {"first": first_anchor, "second": second_anchor},
        "adjacent_borda_scores": {
            first_anchor: first_score,
            second_anchor: second_score,
        },
        "selected_model": selected,
        "reason": reason,
    }


def _spanning_components(
    mask: np.ndarray,
    membership: np.ndarray,
    first_index: int,
    second_index: int,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """Return candidate instances that span one specified adjacent core pair."""
    first_ids = set(np.unique(mask[membership == first_index]).tolist()) - {0}
    second_ids = set(np.unique(mask[membership == second_index]).tolist()) - {0}
    result = []
    for source_id in sorted(first_ids & second_ids):
        components, count = ndi.label(mask == source_id, structure=STRUCTURE)
        for component in range(1, count + 1):
            pixels = components == component
            owners = set(np.unique(membership[pixels]).tolist())
            if first_index in owners and second_index in owners:
                result.append(((int(source_id), component), pixels))
    return result


def stitch_image(
    patches: Sequence[Patch],
    local_vscell_masks: dict[str, np.ndarray],
    patch_rankings: dict[str, Sequence[str]],
    full_candidate_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    """Stitch local consensus masks using adjacent-patch ranking decisions.

    Patch cores tile the full image. Away from core boundaries, the final local
    consensus mask is retained. For each horizontal or vertical adjacent pair,
    the two patches' top-ranked candidates are compared using their ranks in
    both patches. Complete instances from the selected candidate that span the
    shared cores are inserted before the remaining local consensus instances.
    """
    patches = list(patches)
    if not patches:
        raise ValueError("at least one patch is required")
    candidates = list(full_candidate_masks)
    if len(candidates) < 3:
        raise ValueError("at least three candidate masks are required")
    shapes = {np.asarray(mask).shape for mask in full_candidate_masks.values()}
    if len(shapes) != 1:
        raise ValueError("full candidate masks have inconsistent shapes")
    shape = next(iter(shapes))
    validate_partition(patches, shape)
    patch_ids = {patch.patch_id for patch in patches}
    if set(local_vscell_masks) != patch_ids:
        raise ValueError("local VSCell mask coverage does not match patches")
    if set(patch_rankings) != patch_ids:
        raise ValueError("ranking coverage does not match patches")
    for patch_id, ranking in patch_rankings.items():
        if set(ranking) != set(candidates) or len(ranking) != len(candidates):
            raise ValueError(f"{patch_id}: ranking does not cover the candidate models")

    membership = _core_membership(patches, shape)
    index_by_position = {(patch.row, patch.col): index for index, patch in enumerate(patches)}
    proposals: dict[tuple[str, int, int], dict[str, object]] = {}
    seam_decisions: list[dict[str, object]] = []
    for first_index, first in enumerate(patches):
        for row_delta, col_delta, direction in ((0, 1, "right"), (1, 0, "bottom")):
            second_index = index_by_position.get((first.row + row_delta, first.col + col_delta))
            if second_index is None:
                continue
            second = patches[second_index]
            model, decision = _choose_adjacent_anchor(first, second, patch_rankings)
            components = _spanning_components(
                np.asarray(full_candidate_masks[model]), membership, first_index, second_index
            )
            decision["direction"] = direction
            decision["crossing_instances"] = len(components)
            seam_decisions.append(decision)
            score = max(decision["adjacent_borda_scores"].values())
            for (source_id, component), pixels in components:
                key = (model, source_id, component)
                proposal = proposals.setdefault(
                    key,
                    {
                        "model": model,
                        "source_instance_id": source_id,
                        "component": component,
                        "pixels": pixels,
                        "score": score,
                        "edges": [],
                    },
                )
                proposal["score"] = max(int(proposal["score"]), score)
                proposal["edges"].append([first.patch_id, second.patch_id])

    stitched = np.zeros(shape, dtype=np.int32)
    next_id = 1
    proposal_conflicts = 0
    selected_components = []
    for proposal in sorted(
        proposals.values(),
        key=lambda item: (
            -int(item["score"]),
            str(item["model"]),
            int(item["source_instance_id"]),
            int(item["component"]),
        ),
    ):
        pixels = np.asarray(proposal["pixels"])
        free = pixels & (stitched == 0)
        if not free.any():
            proposal_conflicts += 1
            continue
        if np.any(pixels & (stitched > 0)):
            proposal_conflicts += 1
        stitched[free] = next_id
        selected_components.append(
            {
                "model": proposal["model"],
                "source_instance_id": proposal["source_instance_id"],
                "component": proposal["component"],
                "adjacent_borda_score": proposal["score"],
                "edges": proposal["edges"],
                "pixels_written": int(free.sum()),
            }
        )
        next_id += 1

    suppressed_local = 0
    local_written = 0
    patch_counts: dict[str, int] = defaultdict(int)
    for patch in sorted(patches, key=lambda item: (item.row, item.col)):
        local = np.asarray(local_vscell_masks[patch.patch_id])
        if local.shape != patch.halo_shape:
            raise ValueError(
                f"{patch.patch_id}: expected halo mask {patch.halo_shape}, got {local.shape}"
            )
        core_local = local[patch.core_in_halo]
        target = stitched[patch.core_y0 : patch.core_y1, patch.core_x0 : patch.core_x1]
        for instance_id in np.unique(core_local):
            instance_id = int(instance_id)
            if instance_id <= 0:
                continue
            pixels = core_local == instance_id
            if np.any(target[pixels] > 0):
                suppressed_local += 1
                continue
            target[pixels] = next_id
            next_id += 1
            local_written += 1
            patch_counts[patch.patch_id] += 1

    stitched = relabel_instances(stitched)
    audit: dict[str, object] = {
        "stitch_policy": "adjacent_anchor_borda_seam_replacement",
        "seam_decisions": seam_decisions,
        "selected_seam_components": selected_components,
        "seam_component_count": len(selected_components),
        "proposal_conflicts": proposal_conflicts,
        "local_cells_written": local_written,
        "local_cells_suppressed_by_seam": suppressed_local,
        "local_cells_written_by_patch": dict(patch_counts),
        "final_cells": int(stitched.max()),
    }
    return stitched, audit
