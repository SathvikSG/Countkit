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
