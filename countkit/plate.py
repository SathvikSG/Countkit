"""Colony counting on agar plate photos. Classic image processing, no AI.

Steps:
  1. Find the plate (largest circle) and keep only the inner part, so the
     rim and anything written on the edge don't get counted.
  2. Subtract the background so uneven lighting doesn't matter.
  3. Threshold what's left. Works for colonies lighter or darker than the agar.
  4. Split touching colonies with a watershed.
  5. Drop specks and giant blobs (glare, labels, smears).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .measure import relabel_sequential
from .split import split_touching


@dataclass
class PlateSettings:
    rim_fraction: float = 0.90   # keep this fraction of the plate radius
    min_area: int = 12           # px; smaller = noise
    max_area: int = 20000        # px; bigger = glare/label/smear
    min_distance: int = 3        # px between colony centers when splitting
    background_size: int = 51    # px; median filter for background (odd)


def find_plate(gray):
    """Return (cx, cy, r) of the plate, or None if no clear circle shows up."""
    h, w = gray.shape
    scale = 600 / max(h, w) if max(h, w) > 600 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale) if scale != 1.0 else gray
    small = cv2.medianBlur(small, 5)
    sh, sw = small.shape
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(sh, sw),
        param1=100, param2=40,
        minRadius=int(min(sh, sw) * 0.30), maxRadius=int(min(sh, sw) * 0.55),
    )
    if circles is None:
        return None
    cx, cy, r = max(circles[0], key=lambda c: c[2])
    return float(cx / scale), float(cy / scale), float(r / scale)


def plate_mask(shape, circle, rim_fraction):
    mask = np.zeros(shape, dtype=np.uint8)
    cx, cy, r = circle
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(r * rim_fraction), 1, -1)
    return mask.astype(bool)


def count_colonies(gray, settings=None):
    """Return (labels, info). info holds the plate circle and any flags."""
    s = settings or PlateSettings()
    flags = []

    circle = find_plate(gray)
    if circle is None:
        flags.append("no_plate_found")
        h, w = gray.shape
        circle = (w / 2, h / 2, min(h, w) / 2)
    inside = plate_mask(gray.shape, circle, s.rim_fraction)

    k = s.background_size | 1
    background = cv2.medianBlur(gray, k)
    diff = np.abs(gray.astype(np.int16) - background.astype(np.int16)).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    values = diff[inside]
    if values.size == 0 or values.max() == values.min():
        return np.zeros(gray.shape, dtype=np.int32), {"circle": circle, "flags": flags + ["empty"]}
    thresh = max(threshold_otsu(values), 12)  # floor stops noise-only plates lighting up
    mask = (diff > thresh) & inside
    mask = remove_small_objects(mask, max_size=s.min_area - 1)
    mask = remove_small_holes(mask, max_size=s.min_area * 4)

    labels = split_touching(mask, min_distance=s.min_distance)

    areas = np.bincount(labels.ravel())
    too_small = np.nonzero(areas < s.min_area)[0]
    too_big = np.nonzero(areas > s.max_area)[0]
    labels[np.isin(labels, too_small[too_small > 0])] = 0
    if np.any(too_big > 0):
        labels[np.isin(labels, too_big[too_big > 0])] = 0
        flags.append("large_blob_removed")

    return relabel_sequential(labels), {"circle": circle, "flags": flags}
