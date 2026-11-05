"""Reference and anonymized candidate-overlay rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .io import display_rgb


def instance_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    foreground = mask > 0
    eroded = ndi.binary_erosion(foreground, structure=np.ones((3, 3)), border_value=0)
    outer = foreground & ~eroded
    different = np.zeros(mask.shape, dtype=bool)
    different[:-1] |= mask[:-1] != mask[1:]
    different[1:] |= mask[:-1] != mask[1:]
    different[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    different[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return outer | (different & foreground)


def render_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    fill_alpha: float = 0.32,
    fill_color: tuple[int, int, int] = (255, 215, 0),
    boundary_color: tuple[int, int, int] = (0, 255, 80),
) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError(f"image and mask shape mismatch: {image.shape[:2]} vs {mask.shape}")
    base = display_rgb(image).astype(np.float32)
    foreground = np.asarray(mask) > 0
    color = np.asarray(fill_color, dtype=np.float32)
    base[foreground] = (1.0 - fill_alpha) * base[foreground] + fill_alpha * color
    base[instance_boundary(mask)] = np.asarray(boundary_color, dtype=np.float32)
    return np.rint(base).clip(0, 255).astype(np.uint8)


def render_vlm_panel(panel: np.ndarray, panel_size: int) -> np.ndarray:
    """Fit an RGB panel into a square VLM canvas without distorting geometry.

    This is presentation-only. Full-resolution reference crops and label masks
    remain unchanged for consensus and stitching. Nearest-neighbor keeps
    overlay boundaries crisp.
    """
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    panel = np.asarray(panel)
    if panel.ndim != 3 or panel.shape[2] != 3:
        raise ValueError("VLM panel must be an RGB array")
    height, width = panel.shape[:2]
    scale = min(panel_size / height, panel_size / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = np.asarray(
        Image.fromarray(panel, mode="RGB").resize(
            (resized_width, resized_height), Image.Resampling.NEAREST
        )
    )
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    y0 = (panel_size - resized_height) // 2
    x0 = (panel_size - resized_width) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return canvas


def save_panel_set(
    output_dir: str | Path,
    patch_id: str,
    reference: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    panel_size: int = 256,
) -> tuple[Path, dict[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / f"{patch_id}__ref.png"
    Image.fromarray(
        render_vlm_panel(display_rgb(reference), panel_size), mode="RGB"
    ).save(reference_path)
    candidates: dict[str, Path] = {}
    for model, mask in candidate_masks.items():
        path = output_dir / f"{patch_id}__{model}.png"
        Image.fromarray(
            render_vlm_panel(render_overlay(reference, mask), panel_size), mode="RGB"
        ).save(path)
        candidates[model] = path
    return reference_path, candidates
