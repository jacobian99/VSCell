"""Stitch local VSCell masks into a globally labelled full-image mask."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .grid import Patch, validate_partition
from .io import relabel_instances
from .ranking import borda_aggregate


def aggregate_image_ranking(
    patch_rankings: dict[str, Sequence[str]], candidates: Sequence[str]
) -> tuple[list[str], dict[str, int]]:
    if not patch_rankings:
        raise ValueError("no patch rankings supplied")
    return borda_aggregate(patch_rankings.values(), candidates)


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


def cross_core_instances(mask: np.ndarray, patches: Sequence[Patch]) -> list[int]:
    membership = _core_membership(patches, mask.shape)
    result = []
    for instance_id in np.unique(mask):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        if np.unique(membership[mask == instance_id]).size >= 2:
            result.append(instance_id)
    return result


def stitch_image(
    patches: Sequence[Patch],
    local_vscell_masks: dict[str, np.ndarray],
    patch_rankings: dict[str, Sequence[str]],
    full_candidate_masks: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    """Use image-level best-model cells in overlaps and local VSCell elsewhere.

    The disjoint patch cores tile the image. A cell in the image-level best
    candidate is considered an overlap cell when its pixels intersect at least
    two cores. Those full-image instances are written first. Local Solution-08
    instances are then written from each core unless they intersect an already
    protected cross-core cell. Labels are reassigned globally at the end.
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
    if set(local_vscell_masks) != {patch.patch_id for patch in patches}:
        raise ValueError("local VSCell mask coverage does not match patches")
    if set(patch_rankings) != {patch.patch_id for patch in patches}:
        raise ValueError("ranking coverage does not match patches")

    image_ranking, image_scores = aggregate_image_ranking(patch_rankings, candidates)
    best_model = image_ranking[0]
    best_mask = np.asarray(full_candidate_masks[best_model])
    boundary_ids = cross_core_instances(best_mask, patches)

    stitched = np.zeros(shape, dtype=np.int32)
    next_id = 1
    for instance_id in boundary_ids:
        pixels = best_mask == instance_id
        if pixels.any():
            stitched[pixels] = next_id
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
        target = stitched[
            patch.core_y0 : patch.core_y1, patch.core_x0 : patch.core_x1
        ]
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
        "stitch_policy": "image_borda_best_for_cross_core_cells_local_vscell_elsewhere",
        "image_ranking": image_ranking,
        "image_borda_scores": image_scores,
        "overlap_model": best_model,
        "cross_core_source_instance_ids": boundary_ids,
        "cross_core_cells_written": len(boundary_ids),
        "local_cells_written": local_written,
        "local_cells_suppressed_by_overlap": suppressed_local,
        "local_cells_written_by_patch": dict(patch_counts),
        "final_cells": int(stitched.max()),
    }
    return stitched, audit
