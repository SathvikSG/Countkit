#!/usr/bin/env bash
# Rerun every real-data test in one go.
#
#   bash scripts/run_real_tests.sh                    # DSB2018 nuclei (downloads ~27 MB)
#   bash scripts/run_real_tests.sh path/to/AGAR       # + AGAR plates (download AGAR yourself)
#
# Cellpose runs too if it's installed (pip install -e ".[cells]").
# Results land in results/, summaries are printed at the end.
set -euo pipefail

AGAR_DIR="${1:-}"

# Macs often have python3 but no plain python.
PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then
  echo "Python not found. Install Python 3.10+ first." >&2; exit 1
fi
if ! command -v countkit >/dev/null; then
  echo "countkit isn't installed in this environment. From the repo folder run:" >&2
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
  exit 1
fi
if [ -n "$AGAR_DIR" ] && [ ! -d "$AGAR_DIR" ]; then
  echo "AGAR folder not found: $AGAR_DIR" >&2; exit 1
fi
AGAR_SAMPLE="${AGAR_SAMPLE:-200}"   # how many AGAR plates to test (all 18k takes hours)
mkdir -p data results

echo "== DSB2018 nuclei =="
if [ ! -d data/dsb2018 ]; then
  curl -L -o data/dsb2018.zip https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
  unzip -q data/dsb2018.zip -d data
fi
"$PY" scripts/truth_from_masks.py data/dsb2018/test/masks --out data/dsb_truth.csv

countkit count data/dsb2018/test/images --mode cells --engine classic --out results/dsb2018_classic
countkit validate --counts results/dsb2018_classic/counts.csv --truth data/dsb_truth.csv \
  --out results/dsb2018_classic/validation

if "$PY" -c "import cellpose" 2>/dev/null; then
  countkit count data/dsb2018/test/images --mode cells --engine cellpose --out results/dsb2018_cellpose
  countkit validate --counts results/dsb2018_cellpose/counts.csv --truth data/dsb_truth.csv \
    --out results/dsb2018_cellpose/validation
else
  echo "Cellpose isn't installed, skipping it. Install with: pip install -e \".[cells]\""
fi

if [ -n "$AGAR_DIR" ]; then
  echo "== AGAR plates (random sample of $AGAR_SAMPLE with at least 1 colony) =="
  "$PY" scripts/truth_from_agar.py "$AGAR_DIR" --out data/agar_truth_all.csv
  rm -rf data/agar_sample && mkdir -p data/agar_sample
  "$PY" - "$AGAR_DIR" "$AGAR_SAMPLE" <<'PY'
import sys, pandas as pd
from pathlib import Path
src, n = Path(sys.argv[1]).resolve(), int(sys.argv[2])
t = pd.read_csv("data/agar_truth_all.csv")
t = t[t["true_count"] > 0].sample(n=min(n, (t["true_count"] > 0).sum()), random_state=0)
t[["image", "true_count"]].to_csv("data/agar_truth.csv", index=False)
for name, path in zip(t["image"], t["path"]):
    Path("data/agar_sample", name).symlink_to(path)
print(f"Sampled {len(t)} plates")
PY
  countkit count data/agar_sample --mode plate --out results/agar
  countkit validate --counts results/agar/counts.csv --truth data/agar_truth.csv \
    --out results/agar/validation
fi

echo
for f in results/*/validation/validation_summary.txt; do
  echo "---- $f"
  cat "$f"
done
