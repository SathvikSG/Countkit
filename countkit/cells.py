"""Cell / nucleus counting on microscope images.

Two engines:
  - cellpose: pretrained deep learning model. Best on real data. Needs
    `pip install countkit[cells]` (pulls in PyTorch).
  - classic: threshold + watershed. No extra installs. Fine for clean
    fluorescent nuclei, weaker on brightfield or crowded fields.

`auto` uses cellpose if it's installed and falls back to classic if not.
"""

from dataclasses import dataclass
import warnings

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from .measure import relabel_sequential
from .split import split_touching


@dataclass
class CellSettings:
    engine: str = "auto"         # auto | cellpose | classic
    diameter: float = 0.0        # px; 0 lets cellpose estimate it
    cellpose_model: str = "cyto3"
    min_area: int = 15
    min_distance: int = 7
    dark_cells: bool = False     # classic engine: True for dark cells on a bright field


def cellpose_available():
    try:
        import cellpose  # noqa: F401
        return True
    except ImportError:
        return False


_model_cache = {}


def _cellpose_model(name):
    if name not in _model_cache:
        from cellpose import core, models
        # Use the GPU when there is one (NVIDIA, or Apple silicon). Cellpose 4's
        # model is big: about 80 s per image on a laptop CPU vs a few s on GPU.
        gpu = core.use_gpu()
        # Cellpose 4 dropped model_type/the Cellpose class; 3.x has both.
        try:
            _model_cache[name] = models.CellposeModel(gpu=gpu, model_type=name)
        except TypeError:
            _model_cache[name] = models.CellposeModel(gpu=gpu)
    return _model_cache[name]


def segment_cellpose(gray, s):
    model = _cellpose_model(s.cellpose_model)
    result = model.eval(gray, diameter=s.diameter or None)
    return np.asarray(result[0], dtype=np.int32)


def segment_classic(gray, s):
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    if s.dark_cells:
        img = 255 - img
    # Flatten uneven illumination with a big morphological opening.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    img = cv2.subtract(img, cv2.morphologyEx(img, cv2.MORPH_OPEN, k))
    if img.max() == img.min():
        return np.zeros(gray.shape, dtype=np.int32)
    mask = img > max(threshold_otsu(img), 8)
    mask = remove_small_objects(mask, max_size=s.min_area - 1)
    mask = remove_small_holes(mask, max_size=s.min_area * 4)
    labels = split_touching(mask, min_distance=s.min_distance)
    areas = np.bincount(labels.ravel())
    small = np.nonzero(areas < s.min_area)[0]
    labels[np.isin(labels, small[small > 0])] = 0
    return labels


def resolve_engine(engine):
    if engine == "auto":
        if cellpose_available():
            return "cellpose"
        warnings.warn("Cellpose not installed, using the classic engine. "
                      "Install with: pip install 'countkit[cells]'")
        return "classic"
    if engine == "cellpose" and not cellpose_available():
        raise ImportError("Cellpose isn't installed. Run: pip install 'countkit[cells]'")
    if engine not in ("cellpose", "classic"):
        raise ValueError(f"Unknown cell engine: {engine}")
    return engine


def count_cells(gray, settings=None):
    """Return (labels, info)."""
    s = settings or CellSettings()
    engine = resolve_engine(s.engine)
    labels = segment_cellpose(gray, s) if engine == "cellpose" else segment_classic(gray, s)
    labels = relabel_sequential(labels)
    flags = []
    if labels.max() == 0:
        flags.append("no_cells_found")
    return labels, {"engine": engine, "flags": flags}
