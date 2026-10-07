"""GT-free VSCell Solution 08 anchor-consensus selection.

This is a package-native port of the final paper-facing selector in
``08_anchor_con_selection/evaluate_08_anchor_con_selection.py``. Ground truth
is neither accepted nor consulted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import EnsembleConfig
from .io import relabel_instances


Instance = dict[str, Any]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def _valid_instances(
    prediction: np.ndarray,
    model: str,
    rank_index: int,
    minimum_area: int,
) -> list[Instance]:
    instances: list[Instance] = []
    for instance_id in np.unique(prediction):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        pixels = prediction == instance_id
        area = int(pixels.sum())
        if area < minimum_area:
            continue
        instances.append(
            {
                "model": model,
                "rank_index": rank_index,
                "instance_id": instance_id,
                "pixels": pixels,
                "area": area,
            }
        )
    return instances


def _match_one_peer_model(
    anchors: list[Instance], peers: list[Instance], threshold: float
) -> list[tuple[int, int, float]]:
    if not anchors or not peers:
        return []
    weights = np.zeros((len(anchors), len(peers)), dtype=float)
    for anchor_index, anchor in enumerate(anchors):
        for peer_index, peer in enumerate(peers):
            value = mask_iou(anchor["pixels"], peer["pixels"])
            if value >= threshold:
                weights[anchor_index, peer_index] = value
    anchor_rows, peer_columns = linear_sum_assignment(weights, maximize=True)
    matches = []
    for anchor_index, peer_index in zip(anchor_rows.tolist(), peer_columns.tolist()):
        value = float(weights[anchor_index, peer_index])
        if value >= threshold:
            matches.append((anchor_index, peer_index, value))
    return matches


def _strict_residual_clusters(
    residual_instances: list[Instance],
    complete_link_iou: float,
    support_required: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    clusters: list[dict[str, Any]] = []
    for candidate in residual_instances:
        choices: list[tuple[float, int]] = []
        for cluster_index, cluster in enumerate(clusters):
            if candidate["rank_index"] in cluster["contributors"]:
                continue
            overlaps = [
                mask_iou(candidate["pixels"], member["pixels"])
                for member in cluster["members"]
            ]
            if overlaps and min(overlaps) >= complete_link_iou:
                choices.append((float(np.mean(overlaps)), cluster_index))
        if choices:
            _, best_index = max(choices, key=lambda item: (item[0], -item[1]))
            clusters[best_index]["members"].append(candidate)
            clusters[best_index]["contributors"].add(candidate["rank_index"])
        else:
            clusters.append(
                {"members": [candidate], "contributors": {candidate["rank_index"]}}
            )

    consensus: list[dict[str, Any]] = []
    for cluster_index, cluster in enumerate(clusters):
        members = cluster["members"]
        support = len(cluster["contributors"])
        if support < support_required:
            continue
        if len(members) == 1:
            matrix = np.ones((1, 1), dtype=float)
            pairwise_min = pairwise_mean = 1.0
            medoid_scores = [1.0]
        else:
            matrix = np.eye(len(members), dtype=float)
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    value = mask_iou(members[left]["pixels"], members[right]["pixels"])
                    matrix[left, right] = matrix[right, left] = value
            pairwise = matrix[np.triu_indices(len(members), k=1)]
            pairwise_min = float(pairwise.min())
            pairwise_mean = float(pairwise.mean())
            medoid_scores = [
                float((matrix[index].sum() - 1.0) / (len(members) - 1))
                for index in range(len(members))
            ]
        medoid_index = min(
            range(len(members)),
            key=lambda index: (
                -medoid_scores[index],
                members[index]["rank_index"],
                members[index]["instance_id"],
            ),
        )
        representative = min(
            members,
            key=lambda member: (member["rank_index"], member["instance_id"]),
        )
        consensus.append(
            {
                "cluster_index": cluster_index,
                "members": members,
                "support": support,
                "pairwise_min_iou": pairwise_min,
                "pairwise_mean_iou": pairwise_mean,
                "medoid": members[medoid_index],
                "medoid_mean_iou": medoid_scores[medoid_index],
                "representative": representative,
            }
        )
    consensus.sort(
        key=lambda cluster: (
            -cluster["support"],
            -cluster["pairwise_mean_iou"],
            cluster["medoid"]["rank_index"],
            cluster["medoid"]["instance_id"],
        )
    )
    return consensus, {
        "raw_residual_instances": len(residual_instances),
        "all_residual_clusters": len(clusters),
        "consensus_residual_clusters": len(consensus),
        "unsupported_residual_clusters": len(clusters) - len(consensus),
    }


def select_solution08(
    predictions: dict[str, np.ndarray],
    ranking: Sequence[str],
    config: EnsembleConfig = EnsembleConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select instances using final Solution 08 before boundary refinement."""
    order = list(ranking)
    if len(order) < 3 or set(order) != set(predictions):
        raise ValueError("ranking must contain every prediction model exactly once")
    shapes = {np.asarray(mask).shape for mask in predictions.values()}
    if len(shapes) != 1:
        raise ValueError(f"candidate mask shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("candidate masks must be 2-D")

    anchors = _valid_instances(
        predictions[order[0]], order[0], 0, config.minimum_area
    )
    peer_by_rank = {
        rank_index: _valid_instances(
            predictions[model], model, rank_index, config.minimum_area
        )
        for rank_index, model in enumerate(order[1:], 1)
    }

    matched_keys: set[tuple[int, int]] = set()
    anchor_supporters: list[list[dict[str, Any]]] = [[] for _ in anchors]
    matching_rows: list[dict[str, Any]] = []
    for rank_index in sorted(peer_by_rank):
        peers = peer_by_rank[rank_index]
        for anchor_index, peer_index, value in _match_one_peer_model(
            anchors, peers, config.anchor_support_iou
        ):
            peer = peers[peer_index]
            matched_keys.add((rank_index, int(peer["instance_id"])))
            supporter = {
                "model": str(peer["model"]),
                "rank_index": rank_index,
                "instance_id": int(peer["instance_id"]),
                "iou": float(value),
            }
            anchor_supporters[anchor_index].append(supporter)
            matching_rows.append(
                {
                    "anchor_index": anchor_index,
                    "anchor_instance_id": int(anchors[anchor_index]["instance_id"]),
                    **supporter,
                }
            )

    residual_instances = [
        peer
        for rank_index in sorted(peer_by_rank)
        for peer in peer_by_rank[rank_index]
        if (rank_index, int(peer["instance_id"])) not in matched_keys
    ]
    clusters, cluster_audit = _strict_residual_clusters(
        residual_instances,
        config.residual_complete_link_iou,
        config.support_required,
    )

    accepted: list[dict[str, Any]] = []
    removed_anchor_indices: set[int] = set()
    actions = {
        "new_over_null": 0,
        "replacement": 0,
        "anchor_support_protected": 0,
        "anchor_related_suppressed": 0,
        "no_free_pixels_suppressed": 0,
    }
    proposal_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        representative = cluster["representative"]
        per_anchor_max = [
            max(
                mask_iou(member["pixels"], anchor["pixels"])
                for member in cluster["members"]
            )
            for anchor in anchors
        ]
        best_overlap = max(per_anchor_max, default=0.0)
        row: dict[str, Any] = {
            "support": int(cluster["support"]),
            "pairwise_min_iou": float(cluster["pairwise_min_iou"]),
            "pairwise_mean_iou": float(cluster["pairwise_mean_iou"]),
            "representative_model": str(representative["model"]),
            "representative_instance_id": int(representative["instance_id"]),
            "representative_rule": "highest_llm_rank",
            "per_anchor_max_iou": per_anchor_max,
            "anchor_max_iou": best_overlap,
            "member_models": [str(member["model"]) for member in cluster["members"]],
        }
        if best_overlap >= config.anchor_null_iou:
            actions["anchor_related_suppressed"] += 1
            row["action"] = "anchor_related_suppressed"
            proposal_rows.append(row)
            continue

        best_anchor: int | None = None
        intended_action = "new_over_null"
        if best_overlap > 0.0:
            best_anchor = max(
                range(len(anchors)), key=lambda index: (per_anchor_max[index], -index)
            )
            intended_action = "replacement"
            if anchor_supporters[best_anchor]:
                actions["anchor_support_protected"] += 1
                row.update(
                    {
                        "action": "anchor_support_protected",
                        "protected_anchor_index": best_anchor,
                        "protected_anchor_instance_id": int(
                            anchors[best_anchor]["instance_id"]
                        ),
                        "protected_anchor_support": len(anchor_supporters[best_anchor]),
                    }
                )
                proposal_rows.append(row)
                continue

        accepted.append(
            {
                "pixels": representative["pixels"].copy(),
                "action": intended_action,
            }
        )
        actions[intended_action] += 1
        if intended_action == "replacement":
            assert best_anchor is not None
            removed_anchor_indices.add(best_anchor)
        row.update(
            {
                "action": intended_action,
                "replaced_anchor_index": best_anchor,
                "replaced_anchor_instance_id": (
                    int(anchors[best_anchor]["instance_id"])
                    if best_anchor is not None
                    else None
                ),
            }
        )
        proposal_rows.append(row)

    selected = np.zeros(shape, dtype=np.int32)
    next_id = 1
    retained_anchor_count = 0
    for anchor_index, anchor in enumerate(anchors):
        if anchor_index in removed_anchor_indices:
            continue
        free = anchor["pixels"] & (selected == 0)
        if free.any():
            selected[free] = next_id
            next_id += 1
            retained_anchor_count += 1

    written = 0
    written_areas: list[int] = []
    for proposal in accepted:
        free = proposal["pixels"] & (selected == 0)
        if not free.any():
            actions["no_free_pixels_suppressed"] += 1
            continue
        selected[free] = next_id
        next_id += 1
        written += 1
        written_areas.append(int(free.sum()))

    support_histogram = Counter(len(items) for items in anchor_supporters)
    audit = {
        "policy": "08_anchor_con_selection",
        "anchor_model": order[0],
        "config": asdict(config),
        "valid_anchor_cells": len(anchors),
        "removed_anchor_cells": len(removed_anchor_indices),
        "retained_anchor_cells": retained_anchor_count,
        "anchors_with_no_peer_support": int(support_histogram.get(0, 0)),
        "anchors_with_one_peer_supporter": int(support_histogram.get(1, 0)),
        "anchors_with_two_or_more_peer_supporters": int(
            sum(count for support, count in support_histogram.items() if support >= 2)
        ),
        "anchor_support_histogram": {
            str(support): int(count)
            for support, count in sorted(support_histogram.items())
        },
        "valid_peer_instances": int(sum(len(items) for items in peer_by_rank.values())),
        "peer_instances_consumed_as_anchor_support": len(matched_keys),
        "residual_peer_instances": len(residual_instances),
        "matching": matching_rows,
        "cluster_audit": cluster_audit,
        "accepted_peer_proposals": len(accepted),
        "added_peer_cells_written": written,
        "written_peer_areas": written_areas,
        "selected_cells_before_refinement": int(selected.max()),
        "actions": actions,
        "proposals": proposal_rows,
    }
    return selected, audit


def refine_boundaries(
    selected: np.ndarray,
    candidate_masks: Sequence[np.ndarray],
    config: EnsembleConfig = EnsembleConfig(),
) -> np.ndarray:
    """Apply the historical corroborating-peer boundary refinement."""
    output = np.zeros_like(selected, dtype=np.int32)
    next_id = 1
    for instance_id in np.unique(selected):
        instance_id = int(instance_id)
        if instance_id <= 0:
            continue
        anchor = selected == instance_id
        anchor_area = int(anchor.sum())
        matches: list[tuple[float, np.ndarray]] = []
        for prediction in candidate_masks:
            ids = np.unique(prediction[anchor])
            ids = ids[ids > 0]
            best_iou, best_pixels = 0.0, None
            for candidate_id in ids:
                pixels = prediction == candidate_id
                value = mask_iou(anchor, pixels)
                if value > best_iou:
                    best_iou, best_pixels = value, pixels
            if best_pixels is not None:
                matches.append((best_iou, best_pixels))
        matches.sort(key=lambda item: -item[0])
        # The best match is the source instance and is deliberately excluded.
        corroborators = [
            pixels
            for value, pixels in matches[1:]
            if value >= config.refinement_iou
        ]
        refined = anchor.copy()
        if len(corroborators) >= config.refinement_other_peers:
            stack = np.stack(corroborators)
            refined |= stack.all(axis=0)
            refined &= ~(~stack).all(axis=0)
            if int(refined.sum()) < config.minimum_area:
                refined = anchor
        free = refined & (output == 0)
        if free.any():
            output[free] = next_id
            next_id += 1
    return output


def vscell_consensus(
    predictions: dict[str, np.ndarray],
    ranking: Sequence[str],
    config: EnsembleConfig = EnsembleConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    selected, audit = select_solution08(predictions, ranking, config)
    refined = refine_boundaries(
        selected, [predictions[model] for model in ranking], config
    )
    refined = relabel_instances(refined)
    audit["selected_cells_after_refinement"] = int(refined.max())
    return refined, audit
