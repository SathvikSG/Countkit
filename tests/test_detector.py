import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from countkit.detector import (OVERLAP, PLATE_RADIUS, TILE, boxes_to_labels, nms,
                               plate_crop, tile_origins)
from countkit.synth import make_plate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_plate_detector as tpd  # noqa: E402


def write_fake_agar(root, groups=2, per_group=4):
    """Synthetic plates saved the way AGAR stores them: <id>.jpg + <id>.json in group folders."""
    for g in range(groups):
        folder = Path(root) / f"group{g}"
        folder.mkdir(parents=True)
        for i in range(per_group):
            n = 20 + 15 * i
            img, count, colonies = make_plate(n, seed=100 * g + i, return_colonies=True)
            stem = f"{g}{i:03d}"
            cv2.imwrite(str(folder / f"{stem}.jpg"), img)
            labels = [{"x": x - r, "y": y - r, "width": 2 * r, "height": 2 * r, "class": "S.aureus"}
                      for x, y, r in colonies]
            (folder / f"{stem}.json").write_text(json.dumps(
                {"colonies_number": count, "labels": labels}))
    # one uncountable plate, which must be skipped
    img, _ = make_plate(5, seed=999)
    cv2.imwrite(str(Path(root) / "group0" / "9999.jpg"), img)
    (Path(root) / "group0" / "9999.json").write_text(json.dumps({"colonies_number": -1, "labels": []}))


def test_tiles_cover_everything():
    for size in (300, 640, 1000, 1632):
        starts = tile_origins(size)
        covered = np.zeros(size, bool)
        for s in starts:
            covered[s:s + TILE] = True
        assert covered.all()
        assert starts[-1] + min(TILE, size) == size or size <= TILE
        if len(starts) > 1:
            assert all(b - a <= TILE - OVERLAP for a, b in zip(starts, starts[1:]))


def test_nms_merges_duplicates_from_overlapping_tiles():
    boxes = [np.array([10, 10, 30, 30]), np.array([11, 11, 31, 31]), np.array([100, 100, 120, 120])]
    keep = nms(boxes, [0.9, 0.8, 0.7], 0.5)
    assert keep == [0, 2]


def test_boxes_to_labels_keeps_every_box():
    boxes = np.array([[10, 10, 60, 60], [20, 20, 30, 30], [100, 100, 110, 110]], float)
    labels = boxes_to_labels(boxes, (200, 200))
    assert len(np.unique(labels[labels > 0])) == 3


def test_plate_crop_maps_back_to_original():
    img, _, colonies = make_plate(10, seed=3, return_colonies=True)
    crop, scale, (x0, y0), circle, flags = plate_crop(img)
    assert crop.shape[:2] == (2 * (PLATE_RADIUS + 16),) * 2
    x, y, _ = colonies[0]
    u, v = x * scale - x0, y * scale - y0
    assert abs((u + x0) / scale - x) < 1e-6 and abs((v + y0) / scale - y) < 1e-6
    # the colony should look like a colony (bright) at that spot in the crop
    assert crop[int(v), int(u)].mean() > 150


def test_split_and_tiles(tmp_path):
    write_fake_agar(tmp_path / "agar")
    plates, uncountable = tpd.find_plates(tmp_path / "agar")
    assert uncountable == 1 and len(plates) == 8
    plates = tpd.split(plates, train_per_group=3, seed=0)
    for g in ("group0", "group1"):
        splits = sorted(p["split"] for p in plates if p["group"] == g)
        assert splits == ["test", "train", "train", "val"]
    # same seed, same split
    again = tpd.split(tpd.find_plates(tmp_path / "agar")[0], 3, 0)
    assert [p["split"] for p in again] == [p["split"] for p in plates]

    images, labels = tmp_path / "img", tmp_path / "lab"
    images.mkdir()
    labels.mkdir()
    n_tiles, n_boxes = tpd.write_tiles(plates[0], images, labels)
    # Every colony is in exactly one tile (by its center), so boxes >= colonies;
    # colonies in the overlap of two tiles are written to both.
    assert n_tiles > 0 and n_boxes >= plates[0]["true_count"]
    for txt in labels.glob("*.txt"):
        for line in txt.read_text().splitlines():
            c, *xywh = line.split()
            assert c == "0" and all(0 <= float(v) <= 1 for v in xywh)


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="ultralytics not installed")
def test_detector_engine_runs_with_a_model(tmp_path):
    from ultralytics import YOLO
    from countkit.detector import DetectorSettings, count_colonies_detector
    model_path = tmp_path / "m.pt"
    YOLO("yolov8n.yaml").save(str(model_path))  # untrained, just checks the plumbing
    img, _ = make_plate(20, seed=5)
    labels, info = count_colonies_detector(img, DetectorSettings(model=str(model_path), device="cpu"))
    assert labels.shape == img.shape[:2]
    assert labels.max() == len(info["boxes"])


class _FakeResult:
    def __init__(self, boxes):
        import types
        arr = np.array(boxes, float).reshape(-1, 4)
        wrap = lambda a: types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: a))  # noqa: E731
        self.boxes = types.SimpleNamespace(xyxy=wrap(arr), conf=wrap(np.full(len(arr), 0.9)))


def test_colony_on_tile_seam_counted_once(monkeypatch):
    from countkit import detector
    img, _ = make_plate(0, seed=1)
    crop, *_ = detector.plate_crop(img)
    starts = detector.tile_origins(crop.shape[0])
    # Colony spanning x = 630..670: cut off at the right edge of the first tile
    # (0..640), whole in the second (512..1152).
    y = crop.shape[0] // 2
    calls = {"n": 0}

    class Model:
        def predict(self, tiles, **kw):
            results = []
            for _ in tiles:
                i = calls["n"]
                calls["n"] += 1
                tx = starts[i % len(starts)]
                ty = starts[i // len(starts)]
                if not (ty <= y < ty + detector.TILE):
                    results.append(_FakeResult([]))
                    continue
                x0, x1 = 630 - tx, 670 - tx
                if x1 <= 0 or x0 >= detector.TILE:
                    results.append(_FakeResult([]))
                    continue
                # what a real model does: the box is cut off at the tile edge
                results.append(_FakeResult([[max(x0, 0), y - 20 - ty, min(x1, detector.TILE), y + 20 - ty]]))
            return results

    monkeypatch.setitem(detector._models, "fake", Model())
    boxes, *_ = detector.detect_boxes(img, detector.DetectorSettings(model="fake"))
    assert len(boxes) == 1
