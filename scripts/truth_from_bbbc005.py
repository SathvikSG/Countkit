"""Build truth.csv for BBBC005 (synthetic cells, Broad Bioimage Benchmark Collection).

BBBC005 puts the true cell count in each file name, e.g.
SIMCEPImages_A05_C18_F1_s01_w1.TIF -> 18 cells. F is the focus blur level;
use --max-focus to keep only the sharper images.

Usage:
  python scripts/truth_from_bbbc005.py path/to/BBBC005_images --out truth.csv
"""

import argparse
import csv
import re
from pathlib import Path

PATTERN = re.compile(r"_C(\d+)_F(\d+)_", re.IGNORECASE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="truth.csv")
    p.add_argument("--max-focus", type=int, default=None)
    a = p.parse_args()

    rows, skipped = [], 0
    for f in sorted(Path(a.folder).iterdir()):
        m = PATTERN.search(f.name)
        if not m:
            skipped += 1
            continue
        count, focus = int(m.group(1)), int(m.group(2))
        if a.max_focus is not None and focus > a.max_focus:
            continue
        rows.append({"image": f.name, "true_count": count})

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "true_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.out} ({skipped} files didn't match the name pattern)")


if __name__ == "__main__":
    main()
