"""Filesystem workflow connecting full-image masks to ranking and stitching."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_MODELS, EnsembleConfig, GridConfig
from .ensemble import vscell_consensus
from .grid import GridReference, Patch, build_grid, infer_grid_reference
from .io import list_images, read_image, read_mask, write_mask
from .stitch import stitch_image
from .visualization import save_panel_set


MANIFEST_VERSION = 1


def _mask_path(candidate_root: Path, model: str, image_stem: str) -> Path:
    choices = (
        candidate_root / model / f"{image_stem}_masks.tif",
        candidate_root / model / f"{image_stem}.tif",
        candidate_root / f"{image_stem}__{model}.tif",
    )
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"no full-image instance mask found for model {model!r} and image "
        f"{image_stem!r}. Generate the mask externally or with `vscell backend`, "
        "then place it in one of the supported locations: "
        + ", ".join(str(path) for path in choices)
    )


def analyze_inputs(input_path: str | Path, grid: GridConfig = GridConfig()) -> dict[str, Any]:
    images = list_images(input_path)
    rows = []
    shapes = []
    for path in images:
        shape = read_image(path).shape[:2]
        shapes.append(shape)
        rows.append({"image": str(path.resolve()), "image_id": path.stem, "shape": list(shape)})
    reference = infer_grid_reference(shapes)
    total = 0
    for row in rows:
        patches = build_grid(row["image_id"], tuple(row["shape"]), reference, grid)
        row["grid"] = [patches[0].rows, patches[0].cols]
        row["patch_count"] = len(patches)
        total += len(patches)
    return {
        "image_count": len(images),
        "grid_reference": {"height": reference.height, "width": reference.width},
        "patch_count": total,
        "images": rows,
    }


def prepare_workspace(
    input_path: str | Path,
    candidate_root: str | Path,
    output_dir: str | Path,
    *,
    target: str,
    modality: str,
    models: Iterable[str] = DEFAULT_MODELS,
    grid: GridConfig = GridConfig(),
) -> Path:
    """Crop full-image predictions, render VLM panels, and write a manifest."""
    models = list(models)
    if len(models) < 3 or len(set(models)) != len(models):
        raise ValueError("models must contain at least three unique names")
    output_dir = Path(output_dir).resolve()
    candidate_root = Path(candidate_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_path)
    arrays = {path.stem: read_image(path) for path in images}
    reference = infer_grid_reference(array.shape[:2] for array in arrays.values())
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "target": target,
        "modality": modality,
        "models": models,
        "grid_reference": {"height": reference.height, "width": reference.width},
        "grid_config": grid.__dict__,
        "images": [],
        "patches": [],
    }
    panels_root = output_dir / "panels"
    cropped_masks_root = output_dir / "candidate_crops"
    for image_path in images:
        image_id = image_path.stem
        image = arrays[image_id]
        mask_paths = {
            model: _mask_path(candidate_root, model, image_id) for model in models
        }
        full_masks = {model: read_mask(path) for model, path in mask_paths.items()}
        if any(mask.shape != image.shape[:2] for mask in full_masks.values()):
            raise ValueError(f"{image_id}: candidate mask shape differs from raw image")
        patches = build_grid(image_id, image.shape[:2], reference, grid)
        manifest["images"].append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "shape": list(image.shape[:2]),
                "full_candidate_masks": {
                    model: str(path) for model, path in mask_paths.items()
                },
                "patch_ids": [patch.patch_id for patch in patches],
            }
        )
        for patch in patches:
            reference_crop = patch.crop(image)
            crop_masks = {model: patch.crop(mask) for model, mask in full_masks.items()}
            patch_mask_paths = {}
            for model, mask in crop_masks.items():
                path = cropped_masks_root / model / f"{patch.patch_id}.tif"
                write_mask(path, mask)
                patch_mask_paths[model] = str(path)
            reference_path, candidate_paths = save_panel_set(
                panels_root / image_id,
                patch.patch_id,
                reference_crop,
                crop_masks,
            )
            manifest["patches"].append(
                {
                    **patch.to_dict(),
                    "image_id": image_id,
                    "reference_panel": str(reference_path),
                    "candidate_panels": {
                        model: str(path) for model, path in candidate_paths.items()
                    },
                    "candidate_masks": patch_mask_paths,
                }
            )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("unsupported workspace manifest version")
    return manifest


def select_teacher_patches(
    manifest: dict[str, Any], fraction: float, seed: int = 120912
) -> tuple[list[str], list[str]]:
    if not 0 <= fraction <= 1:
        raise ValueError("teacher fraction must be in [0, 1]")
    patch_ids = sorted(row["patch_id"] for row in manifest["patches"])
    shuffled = patch_ids.copy()
    random.Random(seed).shuffle(shuffled)
    count = round(len(shuffled) * fraction)
    teacher = sorted(shuffled[:count])
    student = sorted(shuffled[count:])
    return teacher, student


def write_teacher_split(
    manifest_path: str | Path, fraction: float, seed: int = 120912
) -> Path:
    manifest = load_manifest(manifest_path)
    teacher, student = select_teacher_patches(manifest, fraction, seed)
    output = Path(manifest_path).with_name("teacher_student_split.json")
    output.write_text(
        json.dumps(
            {
                "seed": seed,
                "teacher_fraction_requested": fraction,
                "teacher_count": len(teacher),
                "student_count": len(student),
                "teacher_patch_ids": teacher,
                "student_patch_ids": student,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def load_rankings(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        rows = data["rankings"] if isinstance(data, dict) and "rankings" in data else data
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [
            {"patch_id": patch_id, "ranking": ranking}
            for patch_id, ranking in rows.items()
        ]
    result = {}
    for row in rows:
        patch_id = row["patch_id"]
        ranking = row.get("ranking", row.get("final_order"))
        if not ranking:
            raise ValueError(f"{patch_id}: missing final ranking")
        result[patch_id] = {**row, "ranking": list(ranking)}
    return result


def finalize_workspace(
    manifest_path: str | Path,
    rankings_path: str | Path,
    *,
    ensemble: EnsembleConfig = EnsembleConfig(),
) -> dict[str, Any]:
    """Run Solution 08 per patch, then stitch and globally relabel images."""
    manifest_path = Path(manifest_path).resolve()
    output_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    rankings = load_rankings(rankings_path)
    patch_rows = {row["patch_id"]: row for row in manifest["patches"]}
    if set(rankings) != set(patch_rows):
        missing = sorted(set(patch_rows) - set(rankings))
        extra = sorted(set(rankings) - set(patch_rows))
        raise ValueError(f"ranking coverage mismatch; missing={missing[:5]} extra={extra[:5]}")

    local_masks: dict[str, Any] = {}
    patch_audits = []
    local_root = output_dir / "vscell_patch_masks"
    for patch_id, row in patch_rows.items():
        predictions = {
            model: read_mask(path) for model, path in row["candidate_masks"].items()
        }
        final_mask, audit = vscell_consensus(
            predictions, rankings[patch_id]["ranking"], ensemble
        )
        destination = local_root / f"{patch_id}.tif"
        write_mask(destination, final_mask)
        local_masks[patch_id] = final_mask
        patch_audits.append(
            {
                "patch_id": patch_id,
                "ranking_source": rankings[patch_id].get("source", "unspecified"),
                "ranking": rankings[patch_id]["ranking"],
                "mask_path": str(destination),
                "ensemble": audit,
            }
        )

    final_root = output_dir / "final_masks"
    image_audits = []
    for image_row in manifest["images"]:
        patch_ids = image_row["patch_ids"]
        patches = [Patch(**{key: patch_rows[patch_id][key] for key in Patch.__dataclass_fields__}) for patch_id in patch_ids]
        full_candidates = {
            model: read_mask(path)
            for model, path in image_row["full_candidate_masks"].items()
        }
        stitched, stitch_audit = stitch_image(
            patches,
            {patch_id: local_masks[patch_id] for patch_id in patch_ids},
            {patch_id: rankings[patch_id]["ranking"] for patch_id in patch_ids},
            full_candidates,
        )
        destination = final_root / f"{image_row['image_id']}_vscell.tif"
        write_mask(destination, stitched)
        image_audits.append(
            {
                "image_id": image_row["image_id"],
                "final_mask_path": str(destination),
                "stitch": stitch_audit,
            }
        )
    audit = {
        "status": "complete",
        "ground_truth_used": False,
        "manifest": str(manifest_path),
        "rankings": str(Path(rankings_path).resolve()),
        "ensemble_config": ensemble.__dict__,
        "patches": patch_audits,
        "images": image_audits,
    }
    audit_path = output_dir / "final_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit
