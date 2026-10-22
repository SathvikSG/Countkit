import hashlib

import cv2
import numpy as np
import pytest

from countkit import detector, pipeline, synth
from countkit.cli import main
from countkit.plate import PlateSettings


def test_prepare_centers_the_plate():
    img, _ = synth.make_plate(10, seed=3)
    rgb, (x0, y0, scale) = detector.prepare(img)
    assert rgb.shape == (detector.SIZE, detector.SIZE, 3)
    # The plate center should land in the middle of the prepared image.
    cx, cy, _ = detector.find_plate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    assert abs((cx - x0) * scale - detector.SIZE / 2) < 2
    assert abs((cy - y0) * scale - detector.SIZE / 2) < 2


def test_prepare_pads_when_plate_runs_off_the_photo():
    img, _ = synth.make_plate(10, seed=3)
    rgb, _ = detector.prepare(img[:, 200:])  # cut off the left side of the dish
    assert rgb.shape == (detector.SIZE, detector.SIZE, 3)


def test_every_box_becomes_one_object():
    boxes = np.array([[10, 10, 90, 90], [40, 40, 50, 50], [100, 100, 120, 130]], float)
    labels = detector.boxes_to_labels(boxes, (200, 200))
    assert len(np.unique(labels)) - 1 == 3  # the small box on the big one is still there


def test_engine_must_fit_mode(tmp_path):
    with pytest.raises(SystemExit):
        main(["count", str(tmp_path), "--mode", "plate", "--engine", "cellpose"])
    with pytest.raises(SystemExit):
        main(["count", str(tmp_path), "--mode", "cells", "--engine", "detector"])


def test_auto_falls_back_to_classic_without_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(detector, "DEFAULT_WEIGHTS", tmp_path / "nope.pt")
    monkeypatch.setattr(detector, "CACHED_WEIGHTS", tmp_path / "nope2.pt")
    monkeypatch.delenv("COUNTKIT_DETECTOR", raising=False)
    assert not detector.available()
    images = tmp_path / "imgs"
    synth.write_dataset(images, "plate", 1, (5, 10), seed=1)
    summaries = pipeline.run(images, tmp_path / "out", "plate", PlateSettings(engine="auto"),
                             progress=None)
    assert summaries[0]["engine"] == "classic"


@pytest.fixture
def random_weights(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    path = tmp_path / "random.pt"
    torch.manual_seed(0)
    model = detector.build_model()
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                "score_threshold": 0.05}, path)
    return path


def test_detector_runs_through_the_pipeline(tmp_path, random_weights):
    # Untrained weights, so the count is meaningless; this checks the plumbing.
    images = tmp_path / "imgs"
    synth.write_dataset(images, "plate", 1, (5, 10), seed=1)
    ps = PlateSettings(engine="detector", detector_weights=str(random_weights))
    summaries = pipeline.run(images, tmp_path / "out", "plate", ps, progress=None)
    assert summaries[0]["engine"] == "detector"
    assert (tmp_path / "out" / "overlays").is_dir()
    assert detector.available(str(random_weights))


def test_download_checks_the_checksum(tmp_path):
    src = tmp_path / "w.pt"
    src.write_bytes(b"pretend weights")
    good = hashlib.sha256(b"pretend weights").hexdigest()
    dest = detector.download_weights(src.as_uri(), tmp_path / "cache" / "w.pt", good)
    assert dest.read_bytes() == b"pretend weights"
    with pytest.raises(ValueError):
        detector.download_weights(src.as_uri(), tmp_path / "cache" / "w2.pt", "0" * 64)
    assert not (tmp_path / "cache" / "w2.pt").exists()


def test_weights_file_sets_the_size(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    path = tmp_path / "small.pt"
    model = detector.build_model(size=512, anchors=(4, 8, 16, 32, 64))
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                "score_threshold": 0.05, "size": 512, "anchors": [4, 8, 16, 32, 64]}, path)
    seen = {}
    real_prepare = detector.prepare

    def spy(color, size=detector.SIZE):
        seen["size"] = size
        return real_prepare(color, size)

    monkeypatch.setattr(detector, "prepare", spy)
    img, _ = synth.make_plate(5, seed=2)
    detector.detect(img, detector.DetectorSettings(weights=str(path), device="cpu"))
    assert seen["size"] == 512
    _, _, _, size = detector.load(detector.DetectorSettings(weights=str(path), device="cpu"))
    assert size == 512
