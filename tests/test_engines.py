import cv2
import numpy as np
import pytest

from countkit.cells import CellSettings, count_cells
from countkit.plate import count_colonies, find_plate
from countkit.split import split_touching
from countkit.synth import make_cells, make_plate


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_find_plate_locates_dish():
    img, _ = make_plate(30, seed=1)
    cx, cy, r = find_plate(gray(img))
    assert abs(cx - 450) < 30 and abs(cy - 450) < 30
    assert 360 < r < 420


@pytest.mark.parametrize("n", [10, 60, 150])
def test_plate_count_close_to_truth(n):
    img, truth = make_plate(n, seed=n)
    labels, info = count_colonies(gray(img))
    assert abs(labels.max() - truth) <= max(2, 0.05 * truth)


def test_empty_plate_counts_zero():
    img, truth = make_plate(0, seed=3)
    labels, _ = count_colonies(gray(img))
    assert truth == 0
    assert labels.max() == 0


def test_rim_label_not_counted():
    # The synthetic label text sits on the rim; nothing there should be counted.
    img, _ = make_plate(0, seed=4)
    labels, info = count_colonies(gray(img))
    cx, cy, r = info["circle"]
    ys, xs = np.nonzero(labels)
    assert len(ys) == 0 or np.all(np.hypot(xs - cx, ys - cy) < r)


@pytest.mark.parametrize("n", [20, 80])
def test_classic_cells_close_to_truth(n):
    img, truth = make_cells(n, seed=n)
    labels, info = count_cells(img, CellSettings(engine="classic"))
    assert info["engine"] == "classic"
    assert abs(labels.max() - truth) <= max(2, 0.05 * truth)


def test_watershed_splits_two_touching_disks():
    mask = np.zeros((60, 100), bool)
    cv2.circle(mask.view(np.uint8), (35, 30), 15, 1, -1)
    cv2.circle(mask.view(np.uint8), (62, 30), 15, 1, -1)
    assert split_touching(mask, min_distance=5).max() == 2


def test_unknown_engine_raises():
    img, _ = make_cells(5, seed=0)
    with pytest.raises(ValueError):
        count_cells(img, CellSettings(engine="nope"))


def test_big_photo_is_scaled_and_mapped_back():
    # A 3x upscaled plate should count the same as the original, at full size.
    img, truth = make_plate(80, seed=11)
    big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    labels, info = count_colonies(gray(big))
    assert labels.shape == big.shape[:2]
    assert abs(labels.max() - truth) <= max(2, 0.05 * truth)
    assert info["circle"][2] > 1000  # circle reported in original pixels


def test_ring_colonies_count_once():
    # Colonies with a pale edge and a center the same color as the agar.
    img, _ = make_plate(0, seed=12)
    rng = np.random.default_rng(0)
    centers = [(300 + 90 * (i % 4), 300 + 90 * (i // 4)) for i in range(12)]
    agar = img[450, 450].tolist()
    for x, y in centers:
        r = int(rng.integers(14, 20))
        cv2.circle(img, (x, y), r, (230, 235, 240), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r // 2, agar, -1, cv2.LINE_AA)
    labels, _ = count_colonies(gray(img))
    assert labels.max() == len(centers)


def test_crowded_plate_is_flagged():
    img, _ = make_plate(0, seed=13)
    for y in range(120, 780, 14):
        for x in range(120, 780, 14):
            if np.hypot(x - 450, y - 450) < 330:
                cv2.circle(img, (x, y), 5, (230, 235, 240), -1)
    _, info = count_colonies(gray(img))
    assert "crowded_plate" in info["flags"]


def test_normal_plate_not_flagged_crowded():
    img, _ = make_plate(250, seed=14)
    _, info = count_colonies(gray(img))
    assert "crowded_plate" not in info["flags"]
