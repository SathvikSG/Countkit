"""Colony detector: a Faster R-CNN trained on hand-boxed plate photos.

The classic plate engine thresholds the image, so anything that stands out
from the agar (printed text, the dish wall, merged colonies) throws it off.
This engine learned what a colony looks like from about 50,000 colonies boxed
by hand (ADBC + AGAR, see README), so it skips text and splits neighbors far
better. It needs PyTorch + torchvision (`pip install -e ".[detector]"`) and the
trained weights (`countkit get-weights` downloads them once, about 90 MB).

Training (scripts/train_detector.py) and counting both go through
`prepare()`, so the model always sees photos the same way: cropped to the
plate and scaled so the crop is `size` px across. The size and anchor sizes
are saved in the weights file, so counting always matches training.
"""

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .plate import find_plate

SIZE = 1024             # px; default plate crop size (the weights file can override it)
ANCHORS = (8, 16, 32, 64, 128)  # px; one anchor size per FPN level, at SIZE
CROP_MARGIN = 1.04      # crop a little past the plate edge so rim colonies aren't cut
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "colony_detector.pt"
CACHED_WEIGHTS = Path.home() / ".cache" / "countkit" / "colony_detector.pt"
WEIGHTS_URL = "https://github.com/SathvikSG/Countkit/releases/download/detector-v1/colony_detector.pt"
WEIGHTS_SHA256 = "768b266fc80f6f62c5cc9f73fb2954ee763ca0946b25ef9e3ddd4685d65714b1"


@dataclass
class DetectorSettings:
    weights: str = ""            # path to weights; "" = COUNTKIT_DETECTOR or the bundled file
    score_threshold: float = 0.0  # 0 = use the cutoff saved with the weights (picked on validation)
    device: str = "auto"         # auto | cpu | mps | cuda


def available(weights=""):
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return _weights_path(weights) is not None


def _weights_path(path):
    for p in (path, os.environ.get("COUNTKIT_DETECTOR", ""), str(DEFAULT_WEIGHTS),
              str(CACHED_WEIGHTS)):
        if p and Path(p).is_file():
            return Path(p)
    return None


def download_weights(url=WEIGHTS_URL, dest=CACHED_WEIGHTS, sha256=WEIGHTS_SHA256):
    """Download the trained weights once and check the file is the one published."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if sha256 and digest != sha256:
        tmp.unlink()
        raise ValueError(f"Downloaded weights don't match the published checksum ({digest})")
    tmp.replace(dest)
    return dest


def prepare(color, size=SIZE):
    """Crop a BGR photo to the plate and scale it to size x size.

    Returns (rgb uint8 image, (x0, y0, scale)). A point (x, y) in the
    prepared image is at (x0 + x / scale, y0 + y / scale) in the original.
    """
    h, w = color.shape[:2]
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    circle = find_plate(gray)
    if circle is None:
        side = max(h, w)
        x0, y0 = (w - side) / 2, (h - side) / 2
    else:
        cx, cy, r = circle
        side = 2 * r * CROP_MARGIN
        x0, y0 = cx - side / 2, cy - side / 2
    scale = size / side
    # One affine warp does the crop, the padding (if the crop runs off the
    # photo) and the resize together.
    m = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]], dtype=np.float64)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    out = cv2.warpAffine(color, m, (size, size), flags=interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), (x0, y0, scale)


def build_model(num_classes=2, size=SIZE, anchors=ANCHORS):
    """Faster R-CNN (ResNet-50 FPN v2) with anchors sized for colonies."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    # Colonies are about 8 to 120 px across after prepare(), smaller than
    # the COCO default anchors (32 to 512).
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=num_classes,
        min_size=size, max_size=size,
        rpn_pre_nms_top_n_test=6000, rpn_post_nms_top_n_test=3000,
        box_detections_per_img=2000, box_score_thresh=0.05,
    )
    # Same 3 anchors per location as the default, so the RPN head is unchanged.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((int(a),) for a in anchors), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    return model


def pick_device(name="auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_cache = {}


def load(settings=None):
    """Return (model, device, score_threshold, size), loading once per weights file."""
    import torch

    s = settings or DetectorSettings()
    path = _weights_path(s.weights)
    if path is None:
        raise FileNotFoundError(
            "No colony detector weights found. Run `countkit get-weights`, "
            "set COUNTKIT_DETECTOR, or pass --weights. See README.")
    key = (str(path), s.device)
    if key not in _cache:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        size = int(ckpt.get("size", SIZE))
        model = build_model(size=size, anchors=tuple(ckpt.get("anchors", ANCHORS)))
        model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
        device = pick_device(s.device)
        model.to(device).eval()
        _cache[key] = (model, device, float(ckpt.get("score_threshold", 0.5)), size)
    model, device, saved, size = _cache[key]
    return model, device, s.score_threshold or saved, size


def detect(color, settings=None):
    """Return (boxes in original pixels as an Nx4 array x1,y1,x2,y2, scores)."""
    import torch

    model, device, threshold, size = load(settings)
    rgb, (x0, y0, scale) = prepare(color, size)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
    with torch.no_grad():
        out = model([t])[0]
    keep = out["scores"] >= threshold
    boxes = out["boxes"][keep].cpu().numpy().astype(np.float64)
    scores = out["scores"][keep].cpu().numpy()
    boxes = boxes / scale + np.array([x0, y0, x0, y0])
    return boxes, scores


def boxes_to_labels(boxes, shape):
    """Draw each box as a filled ellipse so the rest of CountKit (measure,
    overlay, review) can treat detections like any other labeled object.
    Big ones are drawn first so a small colony sitting on a big one stays visible."""
    labels = np.zeros(shape[:2], dtype=np.int32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i, j in enumerate(np.argsort(-areas), start=1):
        x1, y1, x2, y2 = boxes[j]
        center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        axes = (max(1, int(round((x2 - x1) / 2 * 0.9))), max(1, int(round((y2 - y1) / 2 * 0.9))))
        cv2.ellipse(labels, center, axes, 0, 0, 360, i, -1)
    return labels


def count_colonies_detector(color, settings=None):
    """Return (labels, info) like plate.count_colonies, using the detector."""
    boxes, scores = detect(color, settings)
    labels = boxes_to_labels(boxes, color.shape)
    flags = []
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    circle = find_plate(gray)
    if circle is None:
        flags.append("no_plate_found")
    return labels, {"circle": circle, "flags": flags, "scores": scores}
