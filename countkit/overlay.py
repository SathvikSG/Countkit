"""Draw every counted object outlined and numbered, so counts can be checked by eye."""

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

GREEN = (60, 220, 60)     # normal object (BGR)
ORANGE = (0, 150, 255)    # flagged clump
BLUE = (255, 160, 0)      # plate edge used for counting


def draw_overlay(color, labels, objects, circle=None, rim_fraction=None):
    out = color.copy()
    h, w = labels.shape
    clumps = {o["object_id"] for o in objects if o["is_clump"]}

    edges = find_boundaries(labels, mode="inner")
    width = min(h, w) // 800  # thicker outlines on big photos so they show when zoomed out
    if width > 0:
        edges = cv2.dilate(edges.astype(np.uint8), np.ones((width + 1, width + 1), np.uint8)) > 0
        edges &= labels > 0
    clump_edges = edges & np.isin(labels, list(clumps)) if clumps else np.zeros_like(edges)
    out[edges & ~clump_edges] = GREEN
    out[clump_edges] = ORANGE

    if circle is not None:
        cx, cy, r = circle
        r = r * rim_fraction if rim_fraction else r
        cv2.circle(out, (int(cx), int(cy)), int(r), BLUE, max(1, w // 400))

    # Size numbers to the objects, not the image, so they don't cover what they label.
    if objects:
        typical = float(np.median([o["diameter_px"] for o in objects]))
        font_scale = float(np.clip(0.3 * typical / 22, 0.3, 1.5))
    else:
        font_scale = 0.5
    thickness = max(1, int(font_scale * 2))
    for o in objects:
        text = str(o["object_id"])
        pos = (int(o["x"]) + 3, int(o["y"]) - 3)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    ORANGE if o["is_clump"] else (255, 255, 255), thickness, cv2.LINE_AA)

    banner = f"count: {len(objects)}"
    if clumps:
        banner += f"  (clumps: {len(clumps)})"
    banner_scale = max(0.6, min(h, w) / 900)
    for color_, width in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(out, banner, (10, int(30 * banner_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    banner_scale, color_, max(width, int(width * banner_scale)), cv2.LINE_AA)
    return out
