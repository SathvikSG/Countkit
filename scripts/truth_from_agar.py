"""Build truth.csv for the AGAR colony dataset.

AGAR ships one JSON annotation file per image. This script reads the
colony count from `colonies_number` if it's there, otherwise counts the
entries in `labels`. It looks in subfolders too (the free AGAR demo is split
into bright/dark/vague/lower-resolution folders), and writes each image's
path so it can be found again.

AGAR marks plates too crowded to count with -1. Those are skipped, since
there's no true count to compare against.

Usage:
  python scripts/truth_from_agar.py path/to/agar_folder --out truth.csv
"""

import argparse
import csv
import json
from pathlib import Path

COUNT_FIELD = "colonies_number"
LABELS_FIELD = "labels"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="truth.csv")
    a = p.parse_args()

    rows, problems, uncountable = [], [], 0
    for jf in sorted(Path(a.folder).rglob("*.json")):
        data = json.loads(jf.read_text())
        if COUNT_FIELD in data:
            count = int(data[COUNT_FIELD])
        elif LABELS_FIELD in data:
            count = len(data[LABELS_FIELD])
        else:
            problems.append(jf.name)
            continue
        image = next((jf.with_suffix(ext) for ext in IMAGE_EXTENSIONS
                      if jf.with_suffix(ext).exists()), None)
        if image is None:
            problems.append(jf.name)
            continue
        if count < 0:
            uncountable += 1
            continue
        rows.append({"image": image.name, "true_count": count, "path": str(image.resolve())})

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "true_count", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.out}")
    if uncountable:
        print(f"Skipped {uncountable} plates marked uncountable (-1)")
    if problems:
        print(f"Skipped {len(problems)} files (no count field or no matching image), "
              f"e.g. {problems[:3]}")


if __name__ == "__main__":
    main()
