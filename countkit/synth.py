"""Synthetic test images with known counts.

These are for testing the code and for a quick demo when you don't have
data on hand. They are much cleaner than real plates and real cells, so
accuracy on them is NOT a stand-in for accuracy on real data.
"""

import csv
from pathlib import Path

import cv2
import numpy as np


def _place(rng, n, radius_range, center, max_r, touching_share):
    """Pick n circle positions inside a disk. Some are placed touching a neighbor."""
    placed = []
    tries = 0
    while len(placed) < n and tries < n * 400:
        tries += 1
        r = rng.uniform(*radius_range)
        parent = None
        if placed and rng.random() < touching_share:
            parent = int(rng.integers(len(placed)))
            px, py, pr = placed[parent]
            angle = rng.uniform(0, 2 * np.pi)
            d = (pr + r) * 0.92
            x, y = px + d * np.cos(angle), py + d * np.sin(angle)
        else:
            angle = rng.uniform(0, 2 * np.pi)
            dist = max_r * np.sqrt(rng.random())
            x, y = center[0] + dist * np.cos(angle), center[1] + dist * np.sin(angle)
        if np.hypot(x - center[0], y - center[1]) + r > max_r:
            continue
        ok = all(np.hypot(x - qx, y - qy) >= (r + qr) * 1.5
                 for i, (qx, qy, qr) in enumerate(placed) if i != parent)
        if ok:
            placed.append((x, y, r))
    return placed


def make_plate(n_colonies, seed=0, size=900, return_colonies=False):
    """Return (BGR image, true count) for a fake agar plate.

    With return_colonies=True, also return the (x, y, radius) of each colony,
    which is what the detector training test uses as boxes.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (45, 45, 50), np.uint8)          # bench
    c = (size / 2 + rng.uniform(-20, 20), size / 2 + rng.uniform(-20, 20))
    R = size * 0.44
    agar = np.array([95, 150, 185]) + rng.integers(-10, 11, 3)
    cv2.circle(img, (int(c[0]), int(c[1])), int(R + 8), (200, 200, 205), -1)   # dish wall
    cv2.circle(img, (int(c[0]), int(c[1])), int(R), agar.tolist(), -1)
    # handwritten-ish label on the rim
    cv2.putText(img, "E.coli 1e-4", (int(c[0] - R * 0.5), int(c[1] - R * 0.93)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)

    colonies = _place(rng, n_colonies, (4, 11), c, R * 0.85, touching_share=0.15)
    colony_color = np.array([190, 225, 240])
    for x, y, r in colonies:
        shade = colony_color + rng.integers(-15, 11, 3)
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), shade.tolist(), -1, cv2.LINE_AA)

    # uneven lighting + sensor noise
    yy, xx = np.mgrid[0:size, 0:size]
    light = 1 + 0.12 * (xx / size - 0.5) + 0.08 * (yy / size - 0.5)
    img = img.astype(np.float32) * light[..., None] + rng.normal(0, 3, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    if return_colonies:
        return img, len(colonies), colonies
    return img, len(colonies)


def make_cells(n_cells, seed=0, size=512):
    """Return (grayscale image, true count) for fake fluorescent nuclei."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    centre = (size / 2, size / 2)
    cells = _place(rng, n_cells, (6, 11), centre, size * 0.7, touching_share=0.15)
    cells = [(x, y, r) for x, y, r in cells
             if r < x < size - r and r < y < size - r]
    for x, y, r in cells:
        axes = (int(r), int(r * rng.uniform(0.75, 1.0)))
        cv2.ellipse(img, (int(round(x)), int(round(y))), axes, rng.uniform(0, 180),
                    0, 360, float(rng.uniform(140, 230)), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (5, 5), 1.5) + 15 + rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), len(cells)


def write_dataset(out_dir, kind, n_images, count_range, seed=0):
    """Write n images plus truth.csv (image,true_count)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        n = int(rng.integers(count_range[0], count_range[1] + 1))
        img, true_count = (make_plate if kind == "plate" else make_cells)(n, seed=seed + i)
        name = f"{kind}_{i + 1:03d}.png"
        cv2.imwrite(str(out_dir / name), img)
        rows.append({"image": name, "true_count": true_count})
    with open(out_dir / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_count"])
        w.writeheader()
        w.writerows(rows)
    return rows
