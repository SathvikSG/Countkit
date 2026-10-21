"""Colony counting with a trained object detector (YOLO).

The classic plate engine is hand-written rules, and on real AGAR plates it
tops out around 30 to 50% error: printed text gets counted, soft colonies
merge. A detector trained on hand-boxed colonies learns those cases instead.

How it runs on one photo:
  1. Find the plate and scale the photo so the plate radius is PLATE_RADIUS
     px. Colonies then have about the same size in every photo, and the same
     scale is used for training (scripts/train_plate_detector.py).
  2. Cut the plate into overlapping TILE x TILE tiles. Colonies are tiny
     compared to a whole plate, so the model looks at tiles, not the whole
     photo shrunk down.
  3. Run the model on every tile, put the boxes back in plate coordinates,
     and drop duplicates where tiles overlap.
  4. Keep boxes whose center is inside the plate.

The rest of CountKit works on label images, so each box is drawn as a
filled ellipse. Areas from this engine are therefore ellipse estimates.

Needs: pip install -e ".[detect]"
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .plate import find_plate

PLATE_RADIUS = 800   # px; plates are scaled to this radius before tiling
TILE = 640           # px; tile size, also the YOLO training image size
OVERLAP = 128        # px; tiles overlap so colonies on a seam are seen whole


@dataclass
class DetectorSettings:
    model: str = "models/plate_detector.pt"
    confidence: float = 0.25     # drop boxes the model is less sure about
    iou: float = 0.5             # boxes overlapping more than this are one colony
    rim_fraction: float = 0.98   # keep box centers inside this fraction of the radius
    device: str | None = None    # None = let ultralytics pick (GPU/MPS if there is one)


_models = {}


def _load(path):
    if path not in _models:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("The detector needs ultralytics. Run: pip install -e '.[detect]'") from e
        if not Path(path).exists():
            raise FileNotFoundError(
                f"No detector model at {path}. Train one with scripts/train_plate_detector.py "
                "or pass --model path/to/model.pt")
        _models[path] = YOLO(path)
    return _models[path]


def plate_crop(color):
    """Scale the photo so the plate has PLATE_RADIUS, and crop a square around it.

    Returns (crop, scale, (x0, y0), circle, flags). A pixel (u, v) in the crop
    is ((u + x0) / scale, (v + y0) / scale) in the original photo.
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    flags = []
    circle = find_plate(gray)
    if circle is None:
        flags.append("no_plate_found")
        h, w = gray.shape
        circle = (w / 2, h / 2, min(h, w) / 2)
    scale = PLATE_RADIUS / circle[2]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(color, None, fx=scale, fy=scale, interpolation=interp)
    cx, cy = circle[0] * scale, circle[1] * scale
    half = PLATE_RADIUS + 16
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    size = 2 * half
    # Pad with black where the square goes past the photo edge.
    crop = np.zeros((size, size, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, scaled.shape[1]), min(y0 + size, scaled.shape[0])
    crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = scaled[sy0:sy1, sx0:sx1]
    return crop, scale, (x0, y0), circle, flags


def tile_origins(size, tile=TILE, overlap=OVERLAP):
    """Top-left corners of tiles covering 0..size, last tile flush with the edge."""
    if size <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, size - tile, step))
    starts.append(size - tile)
    return starts


def nms(boxes, scores, iou):
    """Non-max suppression. boxes are (x0, y0, x1, y1). Returns kept indices."""
    if len(boxes) == 0:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], 0.0, iou)
    return sorted(int(i) for i in np.array(keep).ravel())


def detect_boxes(color, settings=None):
    """Return (boxes in original pixels as (x0, y0, x1, y1), scores, circle, flags)."""
    s = settings or DetectorSettings()
    model = _load(s.model)
    crop, scale, (x0, y0), circle, flags = plate_crop(color)

    starts = tile_origins(crop.shape[0])
    tiles, offsets = [], []
    for ty in starts:
        for tx in starts:
            tiles.append(crop[ty:ty + TILE, tx:tx + TILE])
            offsets.append((tx, ty))

    # A colony on a tile seam shows up cut off in one tile and whole in its
    # neighbour, and the two boxes overlap too little to merge. Tiles overlap
    # by more than a colony is wide, so drop boxes touching an inner tile edge;
    # the neighbouring tile has the whole colony.
    size = crop.shape[0]
    edge = 2
    boxes, scores = [], []
    for i in range(0, len(tiles), 16):
        results = model.predict(tiles[i:i + 16], imgsz=TILE, conf=s.confidence, iou=s.iou,
                                device=s.device, verbose=False)
        for (tx, ty), r in zip(offsets[i:i + 16], results):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if ((b[0] < edge and tx > 0) or (b[1] < edge and ty > 0)
                        or (b[2] > TILE - edge and tx + TILE < size)
                        or (b[3] > TILE - edge and ty + TILE < size)):
                    continue
                boxes.append(b + [tx, ty, tx, ty])
                scores.append(float(c))

    keep = nms(boxes, scores, s.iou)
    boxes = np.array([boxes[k] for k in keep]).reshape(-1, 4)
    scores = np.array([scores[k] for k in keep])

    # Keep colonies whose center is on the plate.
    centre = PLATE_RADIUS + 16
    cxs, cys = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    on_plate = np.hypot(cxs - centre, cys - centre) <= PLATE_RADIUS * s.rim_fraction
    boxes, scores = boxes[on_plate], scores[on_plate]

    boxes = (boxes + [x0, y0, x0, y0]) / scale
    return boxes, scores, circle, flags


def boxes_to_labels(boxes, shape):
    """Draw each box as a filled ellipse with its own label (biggest first, so small ones stay visible)."""
    labels = np.zeros(shape[:2], np.int32)
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else []
    for n, i in enumerate(order, start=1):
        x0, y0, x1, y1 = boxes[i]
        centre = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
        axes = (max(1, int(round((x1 - x0) / 2))), max(1, int(round((y1 - y0) / 2))))
        cv2.ellipse(labels, centre, axes, 0, 0, 360, n, -1)
    # A box drawn over completely by its neighbours would vanish from the
    # count, so give any missing one a small dot at its center.
    present = set(np.unique(labels))
    for n, i in enumerate(order, start=1):
        if n not in present:
            x0, y0, x1, y1 = boxes[i]
            cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
            labels[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = n
    return labels


def count_colonies_detector(color, settings=None):
    """Return (labels, info) like the classic engine. info also has the raw boxes."""
    s = settings or DetectorSettings()
    boxes, scores, circle, flags = detect_boxes(color, s)
    labels = boxes_to_labels(boxes, color.shape)
    if len(boxes) == 0:
        flags.append("empty")
    return labels, {"circle": circle, "flags": flags, "boxes": boxes, "scores": scores,
                    "rim": s.rim_fraction}
