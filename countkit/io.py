"""Finding and loading images."""

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(path):
    """Return image files in a folder (sorted), or the file itself."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def to_uint8(img):
    """Stretch any bit depth to 0-255 so 8-bit and 16-bit images behave the same."""
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, (0.5, 99.5))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def read_image(path):
    """Load an image as (color BGR uint8, grayscale uint8)."""
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = to_uint8(img)
    if img.ndim == 2:
        gray = img
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color = img[:, :, :3]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray
