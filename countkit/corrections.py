"""Manual corrections from the review page.

Saved as corrections.json next to counts.csv:
  {"plate_01.png": {"removed": [4, 17], "added": [[120.5, 88.0]]}, ...}

removed = object IDs that weren't real, added = (x, y) clicks for missed objects.
"""

import json
from pathlib import Path

import pandas as pd


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, corrections):
    Path(path).write_text(json.dumps(corrections, indent=2, sort_keys=True))


def corrected_count(count, entry):
    return int(count) - len(set(entry.get("removed", []))) + len(entry.get("added", []))


def apply(counts_csv, corrections):
    """Return counts with corrected_count and reviewed columns added."""
    df = pd.read_csv(counts_csv)
    df["corrected_count"] = [
        corrected_count(c, corrections.get(img, {})) for img, c in zip(df["image"], df["count"])
    ]
    df["reviewed"] = df["image"].isin(corrections.keys())
    return df
