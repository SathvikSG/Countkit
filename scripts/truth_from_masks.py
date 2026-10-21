"""Build truth.csv from labeled mask images (one integer label per object).

Works for datasets that ship instance masks next to the images, like
DSB2018 / BBBC038 or BBBC039. The true count is the number of distinct
non-zero labels in each mask. Masks are matched to images by file name.

Usage:
  python scripts/truth_from_masks.py path/to/masks --out truth.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from skimage.io import imread
from skimage.measure import label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("masks")
    p.add_argument("--out", default="truth.csv")
    p.add_argument("--binary", action="store_true",
                   help="masks are 0/1, not labeled: count connected blobs instead")
    a = p.parse_args()

    rows = []
    for f in sorted(Path(a.masks).iterdir()):
        if f.suffix.lower() not in (".png", ".tif", ".tiff"):
            continue
        m = imread(f)
        if m.ndim == 3:
            m = m[..., 0]
        n = int(label(m > 0).max()) if a.binary else len(np.unique(m[m > 0]))
        rows.append({"image": f.name, "true_count": n})

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "true_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
