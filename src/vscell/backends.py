"""Isolated full-image inference adapters for the four VSCell candidates.

Each adapter imports its heavy dependency lazily so it can be executed inside
the matching environment. The core VSCell environment does not need Cellpose,
TensorFlow, CellSAM, or their weights.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Target, normalize_modality
from .io import gray_original, gray_u8, list_images, read_image, rgb_u8, write_mask


BackendFunction = Callable[[np.ndarray, Target, str, str], np.ndarray]


@dataclass(frozen=True)
class BackendResult:
    backend: str
    image: Path
    mask: Path
    target: Target
    modality: str


def _announce_default(backend: str, target: Target, detail: str) -> None:
    warnings.warn(
        f"VSCell is using default {backend} ({detail}) for {target.replace('_', '-')} "
        "segmentation because this backend has no target-specific switch; proceeding.",
        UserWarning,
        stacklevel=2,
    )


def _use_gpu(device: str) -> bool:
    if device == "cuda":
        return True
    if device == "cpu":
        return False
    if device != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def predict_cellpose(
    image: np.ndarray, target: Target, modality: str, device: str = "auto"
) -> np.ndarray:
    """Classic Cellpose v3 adapter using nuclei/cyto and paper preprocessing."""
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError(
            "Cellpose backend requires the classic Cellpose environment "
            "(validated with cellpose 3.1.1.3)."
        ) from error
    if not hasattr(models, "Cellpose"):
        raise RuntimeError(
            "classic Cellpose API is unavailable; run this backend in the "
            "Cellpose-v3 environment, not the Cellpose-SAM v4 environment"
        )
    canonical = normalize_modality(modality)
    if canonical == "H&E":
        prepared = 255 - gray_u8(image)
    else:
        prepared = gray_original(image).astype(np.float32)
    model_type = "nuclei" if target == "nuclear" else "cyto"
    model = models.Cellpose(gpu=_use_gpu(device), model_type=model_type)
    masks = model.eval(prepared, diameter=None, channels=[0, 0])[0]
    return np.asarray(masks, dtype=np.int32)


def predict_cellpose_sam(
    image: np.ndarray, target: Target, modality: str, device: str = "auto"
) -> np.ndarray:
    """Original Cellpose-SAM ``cpsam`` adapter validated with cellpose 4.1.1."""
    _announce_default("Cellpose-SAM", target, "cpsam")
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError(
            "Cellpose-SAM backend requires cellpose 4.1.1 in its isolated environment"
        ) from error
    canonical = normalize_modality(modality)
    prepared = rgb_u8(image) if canonical == "H&E" else gray_original(image).astype(np.float32)
    model = models.CellposeModel(gpu=_use_gpu(device), pretrained_model="cpsam")
    masks = model.eval(prepared)[0]
    return np.asarray(masks, dtype=np.int32)


def predict_stardist(
    image: np.ndarray, target: Target, modality: str, device: str = "auto"
) -> np.ndarray:
    """StarDist adapter using the published versatile HE/fluorescence models."""
    del device  # TensorFlow controls its own device.
    try:
        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
    except ImportError as error:
        raise RuntimeError(
            "StarDist backend requires the isolated TensorFlow/StarDist environment"
        ) from error
    canonical = normalize_modality(modality)
    if target == "whole_cell":
        _announce_default(
            "StarDist", target, "available pretrained models are nuclear-oriented"
        )
    if canonical == "H&E":
        prepared = rgb_u8(image).astype(np.float32)
        normalized = normalize(prepared, 1, 99.8, axis=(0, 1))
        variant = "2D_versatile_he"
    else:
        prepared = gray_original(image).astype(np.float32)
        # Reproduce the corrected CB mIF experiment: invert only StarDist input.
        if canonical == "mIF":
            prepared = float(np.nanmax(prepared)) - prepared
        normalized = normalize(prepared, 1, 99.8)
        variant = "2D_versatile_fluo"
    model = StarDist2D.from_pretrained(variant)
    labels, _ = model.predict_instances(normalized)
    return np.asarray(labels, dtype=np.int32)


def predict_cellsam(
    image: np.ndarray, target: Target, modality: str, device: str = "auto"
) -> np.ndarray:
    """CellSAM v1.2 general-model adapter.

    CellSAM has no nuclear/whole-cell switch. Users obtain official weights
    under the upstream terms and provide ``DEEPCELL_ACCESS_TOKEN`` as needed.
    """
    _announce_default("CellSAM", target, "cellsam_general")
    try:
        import cellSAM
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CellSAM backend requires its isolated environment and official v1.2 access"
        ) from error
    canonical = normalize_modality(modality)
    prepared = rgb_u8(image) if canonical == "H&E" else gray_u8(image)
    resolved_device = (
        "cuda" if _use_gpu(device) and torch.cuda.is_available() else "cpu"
    )
    ensure_cellsam_credentials()
    local_model = os.environ.get("VSCELL_CELLSAM_MODEL_PATH", "").strip()
    if local_model:
        model = cellSAM.get_local_model(local_model)
    else:
        model = cellSAM.get_model(model="cellsam_general")
    model = model.to(resolved_device).eval()
    output = cellSAM.segment_cellular_image(
        prepared, model=model, device=resolved_device
    )
    return np.asarray(output[0], dtype=np.int32)


BACKENDS: dict[str, BackendFunction] = {
    "cellpose": predict_cellpose,
    "stardist": predict_stardist,
    "cellsam": predict_cellsam,
    "cellpose_sam": predict_cellpose_sam,
}


def run_backend(
    backend: str,
    input_path: str | Path,
    output_dir: str | Path,
    target: Target,
    modality: str,
    device: str = "auto",
    overwrite: bool = False,
) -> list[BackendResult]:
    key = backend.strip().lower().replace("-", "_")
    if key not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {sorted(BACKENDS)}")
    if target not in ("nuclear", "whole_cell"):
        raise ValueError("target must be nuclear or whole_cell")
    output_dir = Path(output_dir) / key
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for image_path in list_images(input_path):
        destination = output_dir / f"{image_path.stem}_masks.tif"
        if destination.exists() and not overwrite:
            results.append(
                BackendResult(key, image_path, destination, target, modality)
            )
            continue
        image = read_image(image_path)
        mask = BACKENDS[key](image, target, modality, device)
        if mask.shape != image.shape[:2]:
            raise RuntimeError(
                f"{key}/{image_path.name}: output {mask.shape} != input {image.shape[:2]}"
            )
        write_mask(destination, mask)
        results.append(BackendResult(key, image_path, destination, target, modality))
    return results


def backend_environment_hint(backend: str) -> str:
    hints = {
        "cellpose": "cellpose==3.1.1.3",
        "cellpose_sam": "cellpose==4.1.1",
        "stardist": "stardist==0.9.2, csbdeep==0.8.2, TensorFlow",
        "cellsam": (
            "cellSAM commit 35bc6e0d and DEEPCELL_ACCESS_TOKEN for official v1.2 weights"
        ),
    }
    return hints[backend]


def ensure_cellsam_credentials() -> None:
    explicit_model = os.environ.get("VSCELL_CELLSAM_MODEL_PATH", "").strip()
    if explicit_model:
        path = Path(explicit_model).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"VSCELL_CELLSAM_MODEL_PATH does not point to a file: {path}"
            )
        return
    cached_model = (
        Path.home()
        / ".deepcell"
        / "models"
        / "cellsam_v1.2"
        / "cellsam_general.pt"
    )
    if cached_model.is_file():
        return
    if not os.environ.get("DEEPCELL_ACCESS_TOKEN", "").strip():
        raise RuntimeError(
            "CellSAM v1.2 weights are not in the default DeepCell cache and "
            "DEEPCELL_ACCESS_TOKEN is unset. Obtain official access from the "
            "DeepCell portal, or point VSCELL_CELLSAM_MODEL_PATH to an authorized "
            "local cellsam_general.pt file. VSCell does not redistribute weights."
        )
