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

from .cells import CellSettings, count_cells
from .detector import DetectorSettings, count_colonies_detector
from .io import list_images, read_image
from .measure import OBJECT_COLUMNS, SUMMARY_COLUMNS, measure_objects, summarize
from .overlay import draw_overlay
from .plate import PlateSettings, count_colonies

LOW_COUNT = 3
HIGH_CLUMP_SHARE = 0.2


def process_image(path, mode, plate_settings=None, cell_settings=None, clump_factor=2.0,
                  detector_settings=None):
    """Count one image. Returns (summary_row, object_rows, overlay_image).

    In plate mode, passing detector_settings uses the trained detector
    instead of the classic engine.
    """
    path = Path(path)
    color, gray = read_image(path)
    start = time.perf_counter()

    if mode == "plate" and detector_settings is not None:
        labels, info = count_colonies_detector(color, detector_settings)
        engine, circle, rim = "detector", info["circle"], info["rim"]
    elif mode == "plate":
        ps = plate_settings or PlateSettings()
        labels, info = count_colonies(gray, ps)
        engine, circle, rim = "classic", info["circle"], ps.rim_fraction
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
        clump_factor=2.0, progress=print, detector_settings=None):
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")
    out_dir = Path(out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    summaries, all_objects = [], []
    for i, path in enumerate(images, start=1):
        summary, objects, overlay = process_image(
            path, mode, plate_settings, cell_settings, clump_factor, detector_settings)
        cv2.imwrite(str(out_dir / "overlays" / f"{path.stem}_overlay.png"), overlay)
        summaries.append(summary)
        all_objects.extend(objects)
        if progress:
            note = f"  [{summary['flags']}]" if summary["flags"] else ""
            progress(f"[{i}/{len(images)}] {path.name}: {summary['count']}{note}")

    _write_csv(out_dir / "counts.csv", SUMMARY_COLUMNS, summaries)
    _write_csv(out_dir / "objects.csv", OBJECT_COLUMNS, all_objects)
    return summaries
