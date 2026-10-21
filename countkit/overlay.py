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
    clump_edges = edges & np.isin(labels, list(clumps)) if clumps else np.zeros_like(edges)
    out[edges & ~clump_edges] = GREEN
    out[clump_edges] = ORANGE

    if circle is not None:
        cx, cy, r = circle
        r = r * rim_fraction if rim_fraction else r
        cv2.circle(out, (int(cx), int(cy)), int(r), BLUE, max(1, w // 400))

    font_scale = max(0.3, min(h, w) / 1600)
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
    cv2.putText(out, banner, (10, int(30 * max(font_scale, 0.8))), cv2.FONT_HERSHEY_SIMPLEX,
                max(font_scale, 0.8), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, banner, (10, int(30 * max(font_scale, 0.8))), cv2.FONT_HERSHEY_SIMPLEX,
                max(font_scale, 0.8), (255, 255, 255), 2, cv2.LINE_AA)
    return out
