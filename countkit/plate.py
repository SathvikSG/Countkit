"""Colony counting on agar plate photos. Classic image processing, no AI.

Steps:
  0. Find the plate (largest circle) and shrink big photos so the plate is
     about 400 px across in radius. Phone and scanner photos are often
     3000+ px wide, and all the size settings below assume this scale.
  1. Keep only the inner part of the plate, so the rim and anything written
     on the edge don't get counted.
  2. Blur a little, then subtract the background so uneven lighting doesn't
     matter. The background filter is wide (60% of the plate radius) so big
     soft colonies stay whole instead of turning into rings.
  3. Threshold what's left. Works for colonies lighter or darker than the agar.
  4. Split touching colonies with a watershed.
  5. Drop specks, giant blobs (glare, labels, smears) and shapes that aren't
     colony-like: thin arcs from the dish wall and most printed characters.
  6. Flag plates where growth covers most of the agar, so single colonies
     can't be told apart (a lawn, or "too many to count").
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects

from .measure import relabel_sequential
from .split import split_touching


WORKING_RADIUS = 400  # px; plates bigger than this get shrunk to it first


@dataclass
class PlateSettings:
    # Sizes are in working pixels, i.e. after the plate is scaled to a
    # 400 px radius. Output areas are converted back to original pixels.
    rim_fraction: float = 0.85   # keep this fraction of the plate radius
    min_area: int = 30           # px; smaller = noise
    max_area: int = 40000        # px; bigger = glare/label/smear
    min_distance: int | None = None  # px between colony centers when splitting; None = from colony size
    blur_sigma: float = 1.0      # px; light blur before background subtraction
    background_fraction: float = 0.6  # background filter width, as a fraction of the plate radius
    min_contrast: int = 6        # gray levels; threshold never goes below this
    max_hole_area: int = 2000    # px; fill holes up to this size (ring-shaped colonies)
    min_solidity: float = 0.8    # area / convex hull area; text and clumps of junk score low
    min_width: float = 8.0       # px; thinner than this = an arc of the dish wall or a pen stroke
                                 # (lowered automatically on plates with tiny colonies)
    max_eccentricity: float = 0.95  # 0 = circle, 1 = line
    crowded_texture: float = 8.0  # flag the plate if the median local contrast is above this


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


def local_contrast(gray, size=15):
    """Standard deviation of brightness in a size x size window around each pixel."""
    g = gray.astype(np.float32)
    mean = cv2.blur(g, (size, size))
    return np.sqrt(np.maximum(cv2.blur(g * g, (size, size)) - mean * mean, 0))


def count_colonies(gray, settings=None):
    """Return (labels, info). Labels are at the original image size.

    info holds the plate circle (original pixels) and any flags.
    """
    s = settings or PlateSettings()
    flags = []
    full_shape = gray.shape

    circle = find_plate(gray)
    if circle is None:
        flags.append("no_plate_found")
        h, w = gray.shape
        circle = (w / 2, h / 2, min(h, w) / 2)

    scale = min(1.0, WORKING_RADIUS / circle[2])
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    work_circle = (circle[0] * scale, circle[1] * scale, circle[2] * scale)
    inside = plate_mask(gray.shape, work_circle, s.rim_fraction)

    smooth = cv2.GaussianBlur(gray, (0, 0), s.blur_sigma) if s.blur_sigma else gray
    k = int(WORKING_RADIUS * s.background_fraction) | 1
    background = cv2.medianBlur(smooth, k)
    diff = np.abs(smooth.astype(np.int16) - background.astype(np.int16)).astype(np.uint8)

    values = diff[inside]
    if values.size == 0 or values.max() == values.min():
        return np.zeros(full_shape, dtype=np.int32), {"circle": circle, "flags": flags + ["empty"]}
    thresh = max(threshold_otsu(values), s.min_contrast)  # floor stops noise-only plates lighting up
    mask = (diff > thresh) & inside
    mask = remove_small_objects(mask, max_size=s.min_area - 1)
    # Some colonies are rings (pale edge, center close to the agar color).
    # Fill them in, or the watershed cuts each ring into a dozen pieces.
    mask = remove_small_holes(mask, max_size=s.max_hole_area)

    # On a normal plate most of the agar is flat, so the median local contrast
    # is low. When colonies (or a lawn) cover most of the plate it jumps.
    if np.median(local_contrast(gray)[inside]) > s.crowded_texture:
        flags.append("crowded_plate")

    # Colony size varies 10x between plates, so the splitting distance follows
    # the typical colony radius on this plate: the median round, solid blob
    # before splitting (so specks of printed text don't drag it down).
    blobs = [r for r in regionprops(label(mask)) if r.area >= s.min_area
             and r.solidity >= 0.9 and r.eccentricity <= 0.8]
    typical_r = float(np.median([np.sqrt(r.area / np.pi) for r in blobs])) if blobs else 5.0
    min_distance = s.min_distance or int(np.clip(round(0.35 * typical_r), 3, 20))
    min_width = min(s.min_width, typical_r)
    # A fixed min_distance gets a bit of
    # smoothing too, so lumpy edges don't each become a colony.
    split_smooth = 2.0 if s.min_distance else 0.1 * typical_r
    labels = split_touching(mask, min_distance=min_distance, smooth=split_smooth)

    areas = np.bincount(labels.ravel())
    too_small = np.nonzero(areas < s.min_area)[0]
    too_big = np.nonzero(areas > s.max_area)[0]
    labels[np.isin(labels, too_small[too_small > 0])] = 0
    if np.any(too_big > 0):
        labels[np.isin(labels, too_big[too_big > 0])] = 0
        flags.append("large_blob_removed")

    for r in regionprops(labels):
        if (r.solidity < s.min_solidity or r.axis_minor_length < min_width
                or r.eccentricity > s.max_eccentricity):
            labels[labels == r.label] = 0

    labels = relabel_sequential(labels)
    if scale < 1.0:
        labels = cv2.resize(labels, (full_shape[1], full_shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    return labels, {"circle": circle, "flags": flags}
