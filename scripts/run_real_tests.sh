#!/usr/bin/env bash
# Rerun every real-data test in one go.
#
#   bash scripts/run_real_tests.sh                       # DSB2018 nuclei (downloads ~27 MB)
#   bash scripts/run_real_tests.sh path/to/AGAR          # + AGAR plates
#   SKIP_DSB=1 bash scripts/run_real_tests.sh path/to/AGAR   # plates only
#
# AGAR folder:
#   - the free demo (AGAR_representative from AGAR_demo.zip): scores the same
#     19 held-out plates as the README (scripts/agar_split.py), never the
#     20 tuning plates
#   - the full set: scores a random sample of AGAR_SAMPLE plates (default 200),
#     including plates with 0 colonies
#
# Cellpose runs if it's installed (pip install -e ".[cells]"), and the plate
# detector if it's installed (pip install -e ".[detector]" && countkit get-weights).
# Results land in results/, and this run's summaries are printed at the end.
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

RAN=()

if [ -z "${SKIP_DSB:-}" ]; then
  echo "== DSB2018 nuclei =="
  if [ ! -d data/dsb2018 ]; then
    curl -L -o data/dsb2018.zip https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip
    unzip -q data/dsb2018.zip -d data
  fi
  "$PY" scripts/truth_from_masks.py data/dsb2018/test/masks --out data/dsb_truth.csv

  countkit count data/dsb2018/test/images --mode cells --engine classic --out results/dsb2018_classic
  countkit validate --counts results/dsb2018_classic/counts.csv --truth data/dsb_truth.csv \
    --out results/dsb2018_classic/validation
  RAN+=(dsb2018_classic)

  if "$PY" -c "import cellpose" 2>/dev/null; then
    countkit count data/dsb2018/test/images --mode cells --engine cellpose --out results/dsb2018_cellpose
    countkit validate --counts results/dsb2018_cellpose/counts.csv --truth data/dsb_truth.csv \
      --out results/dsb2018_cellpose/validation
    RAN+=(dsb2018_cellpose)
  else
    echo "Cellpose isn't installed, skipping it. Install with: pip install -e \".[cells]\""
  fi
fi

if [ -n "$AGAR_DIR" ]; then
  rm -rf data/agar_plates && mkdir -p data/agar_plates
  if [ -d "$AGAR_DIR/higher-resolution" ] && [ -d "$AGAR_DIR/lower-resolution" ]; then
    echo "== AGAR demo: the 19 held-out plates =="
    "$PY" scripts/agar_split.py "$AGAR_DIR" --out data/agar_split.csv
    "$PY" - <<'PY'
import pandas as pd
from pathlib import Path
t = pd.read_csv("data/agar_split.csv")
t = t[(t["split"] == "test") & (t["true_count"] >= 0)]
t[["image", "true_count"]].to_csv("data/agar_truth.csv", index=False)
for name, path in zip(t["image"], t["path"]):
    Path("data/agar_plates", name).symlink_to(path)
print(f"{len(t)} held-out plates")
PY
  else
    echo "== AGAR: random sample of up to $AGAR_SAMPLE plates =="
    "$PY" scripts/truth_from_agar.py "$AGAR_DIR" --out data/agar_truth_all.csv
    "$PY" - "$AGAR_SAMPLE" <<'PY'
import sys, pandas as pd
from pathlib import Path
n = int(sys.argv[1])
t = pd.read_csv("data/agar_truth_all.csv")
t = t.sample(n=min(n, len(t)), random_state=0)  # 0-colony plates stay in: false hits count too
t[["image", "true_count"]].to_csv("data/agar_truth.csv", index=False)
for name, path in zip(t["image"], t["path"]):
    Path("data/agar_plates", name).symlink_to(path)
print(f"Sampled {len(t)} plates")
PY
  fi
  countkit count data/agar_plates --mode plate --engine classic --out results/agar_classic
  countkit validate --counts results/agar_classic/counts.csv --truth data/agar_truth.csv \
    --out results/agar_classic/validation
  RAN+=(agar_classic)
  if "$PY" -c "import countkit.detector as d, sys; sys.exit(0 if d.available() else 1)" 2>/dev/null; then
    countkit count data/agar_plates --mode plate --engine detector --out results/agar_detector
    countkit validate --counts results/agar_detector/counts.csv --truth data/agar_truth.csv \
      --out results/agar_detector/validation
    RAN+=(agar_detector)
  else
    echo "Plate detector isn't installed, skipping it. See README: pip install -e \".[detector]\" && countkit get-weights"
  fi
fi

echo
for name in ${RAN[@]+"${RAN[@]}"}; do
  echo "---- results/$name/validation/validation_summary.txt"
  cat "results/$name/validation/validation_summary.txt"
done
