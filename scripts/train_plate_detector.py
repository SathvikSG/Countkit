"""Train a colony detector on AGAR and compare it to the classic plate engine.

One command does everything:

  python scripts/train_plate_detector.py path/to/AGAR

  1. Split. Plates are grouped by subfolder (the free AGAR demo has bright,
     dark, vague and lower-resolution folders). In each group, a seeded
     shuffle picks --train-per-group plates for training (one of them is used
     for validation during training) and the rest are held out for testing.
     The split is saved to split.csv. Test plates are never used for training.
  2. Build a YOLO dataset. Each plate is scaled and tiled exactly the way
     the detector engine sees it at count time (countkit/detector.py), and
     AGAR's hand-drawn colony boxes are moved into tile coordinates.
  3. Train YOLO, starting from a model pretrained on everyday photos (COCO).
  4. Count the held-out test plates with the classic engine and with the new
     detector, and validate both against the hand counts.

The trained model is copied to models/plate_detector.pt, which is where
`countkit count --mode plate --engine detector` looks by default.

Needs: pip install -e ".[detect]"
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from countkit import pipeline, validate  # noqa: E402
from countkit.detector import (TILE, DetectorSettings, PLATE_RADIUS,  # noqa: E402
                               plate_crop, tile_origins)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def find_plates(root):
    """Every AGAR plate with a usable count: (image path, group, count, boxes)."""
    root = Path(root)
    plates, uncountable = [], 0
    for jf in sorted(root.rglob("*.json")):
        data = json.loads(jf.read_text())
        image = next((jf.with_suffix(e) for e in IMAGE_EXTENSIONS if jf.with_suffix(e).exists()), None)
        if image is None or "labels" not in data:
            continue
        count = int(data.get("colonies_number", len(data["labels"])))
        if count < 0:  # AGAR marks plates too crowded to count with -1
            uncountable += 1
            continue
        boxes = [(b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]) for b in data["labels"]]
        group = jf.parent.relative_to(root).as_posix() or "all"
        plates.append({"path": image, "group": group, "true_count": count, "boxes": boxes})
    return plates, uncountable


def split(plates, train_per_group, seed):
    """Seeded per-group split. Returns the plates with a 'split' key added."""
    groups = {}
    for p in plates:
        groups.setdefault(p["group"], []).append(p)
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda p: p["path"].name)
        random.Random(seed).shuffle(members)
        for i, p in enumerate(members):
            if i == 0:
                p["split"] = "val"
            elif i < train_per_group:
                p["split"] = "train"
            else:
                p["split"] = "test"
    return plates


def write_tiles(plate, images_dir, labels_dir):
    """Cut one plate into training tiles with YOLO labels. Returns (tiles, boxes) written."""
    color = cv2.imread(str(plate["path"]), cv2.IMREAD_COLOR)
    crop, scale, (x0, y0), _, _ = plate_crop(color)
    boxes = [((a * scale - x0), (b * scale - y0), (c * scale - x0), (d * scale - y0))
             for a, b, c, d in plate["boxes"]]
    starts = tile_origins(crop.shape[0])
    n_tiles = n_boxes = 0
    centre = crop.shape[0] / 2
    for ty in starts:
        for tx in starts:
            # Skip tiles that are entirely off the plate (just padding/bench).
            nearest_x = min(max(centre, tx), tx + TILE)
            nearest_y = min(max(centre, ty), ty + TILE)
            if (nearest_x - centre) ** 2 + (nearest_y - centre) ** 2 > (PLATE_RADIUS * 1.05) ** 2:
                continue
            lines = []
            for bx0, by0, bx1, by1 in boxes:
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                if not (tx <= cx < tx + TILE and ty <= cy < ty + TILE):
                    continue
                cx0, cy0 = max(bx0, tx), max(by0, ty)
                cx1, cy1 = min(bx1, tx + TILE), min(by1, ty + TILE)
                w, h = cx1 - cx0, cy1 - cy0
                if w <= 1 or h <= 1:
                    continue
                lines.append(f"0 {(cx0 + w / 2 - tx) / TILE:.6f} {(cy0 + h / 2 - ty) / TILE:.6f} "
                             f"{w / TILE:.6f} {h / TILE:.6f}")
            name = f"{plate['path'].stem}_{tx}_{ty}"
            cv2.imwrite(str(images_dir / f"{name}.jpg"), crop[ty:ty + TILE, tx:tx + TILE],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (labels_dir / f"{name}.txt").write_text("\n".join(lines))
            n_tiles += 1
            n_boxes += len(lines)
    return n_tiles, n_boxes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agar", help="AGAR folder (subfolders are fine)")
    p.add_argument("--out", default="results/plate_detector")
    p.add_argument("--train-per-group", type=int, default=5,
                   help="plates per group used for training, 1 of them for validation (default 5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-model", default="yolov8n.pt",
                   help="pretrained YOLO to start from (default yolov8n.pt, downloaded automatically)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. mps, cpu, 0. Default: let ultralytics pick")
    p.add_argument("--save-model", default="models/plate_detector.pt")
    a = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Needs ultralytics. Run: pip install -e '.[detect]'")

    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    plates, uncountable = find_plates(a.agar)
    if not plates:
        sys.exit(f"No AGAR plates (image + JSON with labels) found in {a.agar}")
    plates = split(plates, a.train_per_group, a.seed)
    with open(out / "split.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "group", "split", "true_count", "path"])
        w.writeheader()
        for pl in plates:
            w.writerow({"image": pl["path"].name, "group": pl["group"], "split": pl["split"],
                        "true_count": pl["true_count"], "path": str(pl["path"].resolve())})
    counts = {s: sum(pl["split"] == s for pl in plates) for s in ("train", "val", "test")}
    print(f"{len(plates)} plates ({uncountable} skipped as uncountable): "
          f"{counts['train']} train, {counts['val']} val, {counts['test']} test")
    if counts["test"] == 0:
        sys.exit("No plates left for testing. Lower --train-per-group.")

    print("\n== Building tiles ==")
    data = out / "yolo_data"
    for s in ("train", "val"):
        (data / "images" / s).mkdir(parents=True)
        (data / "labels" / s).mkdir(parents=True)
        n_t = n_b = 0
        for pl in plates:
            if pl["split"] == s:
                t, b = write_tiles(pl, data / "images" / s, data / "labels" / s)
                n_t, n_b = n_t + t, n_b + b
        print(f"{s}: {n_t} tiles, {n_b} colony boxes")
    (data / "data.yaml").write_text(
        f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n  0: colony\n")

    print("\n== Training ==")
    model = YOLO(a.base_model)
    model.train(data=str(data / "data.yaml"), imgsz=TILE, epochs=a.epochs, batch=a.batch,
                device=a.device, project=str(out), name="train", exist_ok=True,
                patience=max(20, a.epochs // 4), seed=a.seed, deterministic=True,
                # Plates have no up/down, so flips and rotations are all fair game.
                fliplr=0.5, flipud=0.5, degrees=180, mosaic=1.0, verbose=False)
    best = out / "train" / "weights" / "best.pt"
    Path(a.save_model).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, a.save_model)
    print(f"Saved model to {a.save_model}")

    print("\n== Testing on held-out plates ==")
    test_dir = out / "test_images"
    test_dir.mkdir()
    with open(out / "test_truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "true_count"])
        w.writeheader()
        for pl in plates:
            if pl["split"] == "test":
                (test_dir / pl["path"].name).symlink_to(pl["path"].resolve())
                w.writerow({"image": pl["path"].name, "true_count": pl["true_count"]})

    results = {}
    for engine in ("classic", "detector"):
        res = out / f"test_{engine}"
        settings = DetectorSettings(model=a.save_model, device=a.device) if engine == "detector" else None
        pipeline.run(test_dir, res, "plate", progress=None, detector_settings=settings)
        _, m = validate.run(res / "counts.csv", out / "test_truth.csv", res / "validation",
                            f"AGAR held-out plates, {engine} engine (n={counts['test']})")
        results[engine] = m

    print(f"\nHeld-out plates: {counts['test']}\n")
    print(f"{'':28}{'classic':>12}{'detector':>12}")
    rows = [("Mean absolute error", "mean_abs_error", "{:.2f}"),
            ("Mean absolute % error", "mean_abs_pct_error", "{:.2f}%"),
            ("Median absolute % error", "median_abs_pct_error", "{:.2f}%"),
            ("Within ±10%", "within_10pct", "{:.1%}"),
            ("Bias", "bias", "{:+.2f}"),
            ("Pearson r", "pearson_r", "{:.4f}")]
    lines = []
    for label, key, fmt in rows:
        line = f"{label:28}{fmt.format(results['classic'][key]):>12}{fmt.format(results['detector'][key]):>12}"
        lines.append(line)
        print(line)
    (out / "comparison.txt").write_text(
        f"AGAR held-out plates: {counts['test']}\n{'':28}{'classic':>12}{'detector':>12}\n"
        + "\n".join(lines) + "\n")
    print(f"\nEverything is in {out}")


if __name__ == "__main__":
    main()
