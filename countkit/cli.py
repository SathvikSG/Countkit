"""Command line entry point: `countkit count|validate|demo|review`."""

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__, pipeline, synth, validate
from .cells import CellSettings
from .plate import PlateSettings


def cmd_count(a):
    plate = PlateSettings(rim_fraction=a.rim, max_area=a.max_area)
    cells = CellSettings(engine=a.engine, diameter=a.diameter, cellpose_model=a.model,
                         dark_cells=a.dark_cells)
    # Only override the per-mode defaults when the user asked to.
    for settings in (plate, cells):
        if a.min_area is not None:
            settings.min_area = a.min_area
        if a.min_distance is not None:
            settings.min_distance = a.min_distance
    summaries = pipeline.run(a.input, a.out, a.mode, plate, cells, a.clump_factor)
    total = sum(s["count"] for s in summaries)
    secs = sum(s["seconds"] for s in summaries) / len(summaries)
    print(f"\n{len(summaries)} images, {total} objects, {secs:.2f} s per image on average")
    print(f"Results: {Path(a.out) / 'counts.csv'}")
    print(f"Overlays: {Path(a.out) / 'overlays'}")


def cmd_validate(a):
    _, m = validate.run(a.counts, a.truth, a.out)
    print(validate.format_metrics(m))
    print(f"\nPer-image table and agreement plot written to {a.out}")


def cmd_demo(a):
    out = Path(a.out)
    for kind, rng in (("plate", (5, 300)), ("cells", (10, 150))):
        images = out / f"{kind}_images"
        synth.write_dataset(images, kind, a.n, rng, seed=a.seed)
        results = out / f"{kind}_results"
        print(f"\n== {kind} ==")
        settings = CellSettings(engine="classic") if kind == "cells" else None
        pipeline.run(images, results, kind, cell_settings=settings, progress=None)
        _, m = validate.run(results / "counts.csv", images / "truth.csv",
                            results / "validation", f"Synthetic {kind} images (n={a.n})")
        print(validate.format_metrics(m))
    print(f"\nDemo written to {out}")


def cmd_review(a):
    app = Path(__file__).with_name("review_app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                              "--", str(Path(a.results).resolve())]))


def build_parser():
    p = argparse.ArgumentParser(prog="countkit", description=__doc__)
    p.add_argument("--version", action="version", version=f"countkit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("count", help="count objects in a folder of images")
    c.add_argument("input", help="image file or folder of images")
    c.add_argument("--mode", required=True, choices=["plate", "cells"])
    c.add_argument("--out", default="countkit_results")
    c.add_argument("--min-area", type=int, default=None,
                   help="smallest object in px (default: 12 plate, 15 cells)")
    c.add_argument("--max-area", type=int, default=20000, help="plate mode: largest colony in px")
    c.add_argument("--min-distance", type=int, default=None,
                   help="min px between centers when splitting touching objects "
                        "(default: 3 plate, 5 cells)")
    c.add_argument("--rim", type=float, default=0.90,
                   help="plate mode: fraction of plate radius to keep (default 0.90)")
    c.add_argument("--clump-factor", type=float, default=2.0,
                   help="flag objects bigger than this x the median area")
    c.add_argument("--engine", default="auto", choices=["auto", "cellpose", "classic"],
                   help="cells mode engine")
    c.add_argument("--diameter", type=float, default=0.0,
                   help="cells mode: typical cell diameter in px (0 = let cellpose guess)")
    c.add_argument("--model", default="cyto3", help="cellpose model name (cellpose 3.x)")
    c.add_argument("--dark-cells", action="store_true",
                   help="classic engine: cells are darker than the background")
    c.set_defaults(func=cmd_count)

    v = sub.add_parser("validate", help="compare counts.csv to hand counts")
    v.add_argument("--counts", required=True, help="counts.csv from `countkit count`")
    v.add_argument("--truth", required=True, help="CSV with columns image,true_count")
    v.add_argument("--out", default="validation")
    v.set_defaults(func=cmd_validate)

    d = sub.add_parser("demo", help="make synthetic images with known counts and run everything")
    d.add_argument("--out", default="demo")
    d.add_argument("-n", type=int, default=50, help="images per mode (default 50)")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("review", help="open the review page for a results folder")
    r.add_argument("results", help="folder with counts.csv, objects.csv and overlays/")
    r.set_defaults(func=cmd_review)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
