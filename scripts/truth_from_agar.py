"""Build truth.csv for the AGAR colony dataset.

AGAR ships one JSON annotation file per image. This script reads the
colony count from `colonies_number` if it's there, otherwise counts the
entries in `labels`. Check a few files by hand first: if the field names in
your copy differ, change COUNT_FIELD / LABELS_FIELD below.

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

    rows, problems = [], []
    for jf in sorted(Path(a.folder).glob("*.json")):
        data = json.loads(jf.read_text())
        if COUNT_FIELD in data:
            count = int(data[COUNT_FIELD])
        elif LABELS_FIELD in data:
            count = len(data[LABELS_FIELD])
        else:
            problems.append(jf.name)
            continue
        image = next((jf.with_suffix(ext).name for ext in IMAGE_EXTENSIONS
                      if jf.with_suffix(ext).exists()), None)
        if image is None:
            problems.append(jf.name)
            continue
        rows.append({"image": image, "true_count": count})

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "true_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {a.out}")
    if problems:
        print(f"Skipped {len(problems)} files (no count field or no matching image), "
              f"e.g. {problems[:3]}")


if __name__ == "__main__":
    main()
