"""Turn a label image into per-object rows and a per-image summary."""

import numpy as np
from skimage.segmentation import relabel_sequential as skimage_relabel
from skimage.measure import regionprops

OBJECT_COLUMNS = [
    "image", "object_id", "x", "y", "area_px",
    "diameter_px", "circularity", "is_clump",
]

SUMMARY_COLUMNS = [
    "image", "mode", "engine", "count", "clumps",
    "mean_area_px", "median_area_px", "sd_area_px",
    "area_p10_px", "area_p90_px", "seconds", "flags",
]


def measure_objects(labels, image_name, clump_factor=2.0, clump_solidity=0.9):
    """One row per labeled object.

    A clump is an object that is both bigger than clump_factor x the median
    area and not convex (solidity below clump_solidity). Size alone isn't
    enough, since a big round colony is still one colony.
    """
    props = regionprops(labels)
    if not props:
        return []
    median_area = float(np.median([p.area for p in props]))
    rows = []
    for p in props:
        perimeter = p.perimeter if p.perimeter > 0 else 1.0
        y, x = p.centroid
        rows.append({
            "image": image_name,
            "object_id": int(p.label),
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "area_px": int(p.area),
            "diameter_px": round(float(p.equivalent_diameter_area), 2),
            "circularity": round(float(min(4 * np.pi * p.area / perimeter**2, 1.0)), 3),
            "is_clump": bool(p.area > clump_factor * median_area
                             and p.solidity < clump_solidity),
        })
    return rows


def summarize(objects, image_name, mode, engine, seconds, flags):
    areas = np.array([o["area_px"] for o in objects], dtype=float)
    has = areas.size > 0
    return {
        "image": image_name,
        "mode": mode,
        "engine": engine,
        "count": int(areas.size),
        "clumps": int(sum(o["is_clump"] for o in objects)),
        "mean_area_px": round(float(areas.mean()), 1) if has else 0.0,
        "median_area_px": round(float(np.median(areas)), 1) if has else 0.0,
        "sd_area_px": round(float(areas.std()), 1) if has else 0.0,
        "area_p10_px": round(float(np.percentile(areas, 10)), 1) if has else 0.0,
        "area_p90_px": round(float(np.percentile(areas, 90)), 1) if has else 0.0,
        "seconds": round(seconds, 3),
        "flags": ";".join(flags),
    }


def relabel_sequential(labels):
    """Renumber labels 1..N so object IDs on the overlay have no gaps.
    Keeps the original order (lowest old label becomes 1)."""
    out, _, _ = skimage_relabel(np.asarray(labels, dtype=np.int32))
    return out.astype(np.int32)
