"""Split the free AGAR demo into tuning and held-out plates.

This is the split every AGAR number in the README uses, for both plate
engines. The classic engine's settings and the detector's training only
ever saw the tuning plates.

The demo has 40 plates in 4 folders of 10. Each folder's file names are
sorted, then shuffled with ONE shared random.Random(0) (folders in the order
below), and the first 5 of each folder are tuning plates. Plate 1399 is
marked -1 (too crowded to count) and lands in the held-out half, so the
held-out set is 19 plates.

Usage:
  python scripts/agar_split.py path/to/AGAR_representative --out data/agar_split.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path

GROUPS = {  # name: folder inside AGAR_representative, in shuffle order
    "bright": "higher-resolution/bright",
    "dark": "higher-resolution/dark",
    "vague": "higher-resolution/vague",
    "lowres": "lower-resolution",
}


def make_split(root):
    root = Path(root)
    rng = random.Random(0)
    rows = []
    for group, folder in GROUPS.items():
        names = sorted(p.name for p in (root / folder).glob("*.jpg"))
        rng.shuffle(names)
        for i, name in enumerate(names):
            count = json.loads((root / folder / name).with_suffix(".json").read_text())["colonies_number"]
            rows.append({"image": name, "group": group, "split": "tune" if i < 5 else "test",
                         "true_count": int(count), "path": str((root / folder / name).resolve())})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="the AGAR_representative folder from AGAR_demo.zip")
    p.add_argument("--out", default="data/agar_split.csv")
    a = p.parse_args()
    rows = make_split(a.root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "group", "split", "true_count", "path"])
        w.writeheader()
        w.writerows(rows)
    for split in ("tune", "test"):
        kept = [r for r in rows if r["split"] == split and r["true_count"] >= 0]
        print(f"{split}: {len(kept)} plates, {sum(r['true_count'] for r in kept)} colonies")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
