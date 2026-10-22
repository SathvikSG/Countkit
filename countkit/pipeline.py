"""Run a folder of images through one engine and write the shared outputs.

Outputs in the out folder:
  counts.csv     one row per image
  objects.csv    one row per counted object
  overlays/      one PNG per image with every object outlined and numbered
"""

import csv
import time
from pathlib import Path

import cv2

from . import detector
from .cells import CellSettings, count_cells
from .io import list_images, read_image
from .measure import OBJECT_COLUMNS, SUMMARY_COLUMNS, measure_objects, summarize
from .overlay import draw_overlay
from .plate import PlateSettings, count_colonies

LOW_COUNT = 3
HIGH_CLUMP_SHARE = 0.2


def process_image(path, mode, plate_settings=None, cell_settings=None, clump_factor=2.0):
    """Count one image. Returns (summary_row, object_rows, overlay_image)."""
    path = Path(path)
    color, gray = read_image(path)
    start = time.perf_counter()

    if mode == "plate":
        ps = plate_settings or PlateSettings()
        engine = ps.engine
        if engine == "auto":
            engine = "detector" if detector.available(ps.detector_weights) else "classic"
        if engine == "detector":
            ds = detector.DetectorSettings(weights=ps.detector_weights,
                                           score_threshold=ps.score_threshold)
            labels, info = detector.count_colonies_detector(color, ds)
            circle, rim = info["circle"], None  # the detector looks at the whole plate
        elif engine == "classic":
            labels, info = count_colonies(gray, ps)
            circle, rim = info["circle"], ps.rim_fraction
        else:
            raise ValueError(f"plate engine must be auto, detector or classic, got {engine!r}")
    elif mode == "cells":
        labels, info = count_cells(gray, cell_settings or CellSettings())
        engine, circle, rim = info["engine"], None, None
    else:
        raise ValueError(f"mode must be 'plate' or 'cells', got {mode!r}")

    objects = measure_objects(labels, path.name, clump_factor=clump_factor)
    seconds = time.perf_counter() - start

    flags = list(info["flags"])
    if 0 < len(objects) < LOW_COUNT:
        flags.append("low_count")
    if objects and sum(o["is_clump"] for o in objects) / len(objects) > HIGH_CLUMP_SHARE:
        flags.append("many_clumps")

    summary = summarize(objects, path.name, mode, engine, seconds, flags)
    overlay = draw_overlay(color, labels, objects, circle, rim)
    return summary, objects, overlay


def _write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(input_path, out_dir, mode, plate_settings=None, cell_settings=None,
        clump_factor=2.0, progress=print):
    if mode not in ("plate", "cells"):
        raise ValueError(f"mode must be 'plate' or 'cells', got {mode!r}")
    if mode == "plate" and (plate_settings or PlateSettings()).engine not in ("auto", "detector", "classic"):
        raise ValueError(f"plate engine must be auto, detector or classic, "
                         f"got {plate_settings.engine!r}")
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    out_dir = Path(out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    # Rows are written as each image finishes, so a crash or Ctrl-C halfway
    # through a big folder still leaves everything counted so far on disk.
    summaries, errors = [], []
    with open(out_dir / "counts.csv", "w", newline="") as fc, \
            open(out_dir / "objects.csv", "w", newline="") as fo:
        counts_w = csv.DictWriter(fc, fieldnames=SUMMARY_COLUMNS)
        objects_w = csv.DictWriter(fo, fieldnames=OBJECT_COLUMNS)
        counts_w.writeheader()
        objects_w.writeheader()
        for i, path in enumerate(images, start=1):
            try:
                summary, objects, overlay = process_image(
                    path, mode, plate_settings, cell_settings, clump_factor)
            except (ValueError, cv2.error) as e:  # unreadable or broken image: skip it
                errors.append({"image": path.name, "error": str(e).strip().splitlines()[-1]})
                if progress:
                    progress(f"[{i}/{len(images)}] {path.name}: SKIPPED ({errors[-1]['error']})")
                continue
            cv2.imwrite(str(out_dir / "overlays" / f"{path.stem}_overlay.png"), overlay)
            summaries.append(summary)
            counts_w.writerow(summary)
            objects_w.writerows(objects)
            fc.flush()
            fo.flush()
            if progress:
                note = f"  [{summary['flags']}]" if summary["flags"] else ""
                progress(f"[{i}/{len(images)}] {path.name}: {summary['count']}{note}")

    errors_path = out_dir / "errors.csv"
    if errors:
        _write_csv(errors_path, ["image", "error"], errors)
    elif errors_path.exists():
        errors_path.unlink()  # left over from an earlier run
    if not summaries:
        raise ValueError(f"None of the {len(images)} images could be read. See {errors_path}")
    return summaries
