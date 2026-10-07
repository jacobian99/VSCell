"""Image and instance-mask I/O shared by core and backend runners."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(
        item for item in root.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        array = np.asarray(tifffile.imread(path))
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or color image, got {array.shape} from {path}")
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        raise ValueError(f"expected channels last RGB/RGBA, got {array.shape} from {path}")
    return array[..., :3] if array.ndim == 3 else array


def read_mask(path: str | Path) -> np.ndarray:
    mask = np.squeeze(np.asarray(tifffile.imread(path)))
    if mask.ndim != 2:
        raise ValueError(f"instance mask must be 2-D, got {mask.shape}")
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.all(np.isfinite(mask)) or not np.all(mask == np.rint(mask)):
            raise ValueError("instance mask contains non-integral labels")
    mask = mask.astype(np.int64, copy=False)
    if np.any(mask < 0):
        raise ValueError("instance mask labels must be nonnegative")
    return mask


def write_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = relabel_instances(mask)
    dtype = np.uint16 if int(mask.max()) <= np.iinfo(np.uint16).max else np.uint32
    tifffile.imwrite(path, mask.astype(dtype), compression="zlib")


def relabel_instances(mask: np.ndarray) -> np.ndarray:
    """Return contiguous labels, splitting disconnected islands into instances.

    Some upstream segmenters can emit one numeric label for spatially separate
    components.  Keeping that label would make a single VSCell instance refer
    to multiple cells, so every source label is split with 8-connectivity before
    global IDs are assigned.
    """
    mask = np.asarray(mask)
    output = np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for value in np.unique(mask):
        if value <= 0:
            continue
        pixels = mask == value
        components, count = ndi.label(pixels, structure=np.ones((3, 3), dtype=bool))
        for component_id in range(1, count + 1):
            output[components == component_id] = next_id
            next_id += 1
    return output


def gray_original(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    return image[..., :3].astype(np.float32).mean(axis=-1)


def minmax_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.copy()
    work = image.astype(np.float32)
    low = float(np.nanmin(work))
    high = float(np.nanmax(work))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("image contains non-finite values")
    if high <= low:
        return np.zeros(work.shape, dtype=np.uint8)
    return np.rint((work - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)


def gray_u8(image: np.ndarray) -> np.ndarray:
    return minmax_u8(gray_original(image))


def rgb_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return minmax_u8(image[..., :3])


def display_rgb(image: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> np.ndarray:
    image = np.asarray(image)
    work = image.astype(np.float32)
    if work.ndim == 2:
        lo, hi = np.percentile(work, (lower, upper))
        normalized = np.zeros_like(work) if hi <= lo else np.clip((work - lo) / (hi - lo), 0, 1)
        return np.repeat(np.rint(normalized * 255).astype(np.uint8)[..., None], 3, axis=-1)
    output = np.empty(work.shape[:2] + (3,), dtype=np.uint8)
    for channel in range(3):
        plane = work[..., channel]
        lo, hi = np.percentile(plane, (lower, upper))
        normalized = np.zeros_like(plane) if hi <= lo else np.clip((plane - lo) / (hi - lo), 0, 1)
        output[..., channel] = np.rint(normalized * 255).astype(np.uint8)
    return output
