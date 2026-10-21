import json

import pandas as pd
import pytest

from countkit import corrections, pipeline, synth, validate
from countkit.cells import CellSettings
from countkit.cli import main
from countkit.measure import OBJECT_COLUMNS, SUMMARY_COLUMNS


@pytest.fixture
def plate_run(tmp_path):
    images = tmp_path / "images"
    synth.write_dataset(images, "plate", 3, (10, 40), seed=7)
    out = tmp_path / "out"
    pipeline.run(images, out, "plate", progress=None)
    return images, out


def test_outputs_have_shared_columns(plate_run):
    _, out = plate_run
    counts = pd.read_csv(out / "counts.csv")
    objects = pd.read_csv(out / "objects.csv")
    assert list(counts.columns) == SUMMARY_COLUMNS
    assert list(objects.columns) == OBJECT_COLUMNS
    assert len(counts) == 3
    # object table and summary agree
    per_image = objects.groupby("image").size()
    for _, row in counts.iterrows():
        assert per_image.get(row["image"], 0) == row["count"]


def test_one_overlay_per_image(plate_run):
    _, out = plate_run
    assert len(list((out / "overlays").glob("*_overlay.png"))) == 3


def test_cells_mode_same_columns(tmp_path):
    images = tmp_path / "cells"
    synth.write_dataset(images, "cells", 2, (10, 30), seed=1)
    pipeline.run(images, tmp_path / "out", "cells",
                 cell_settings=CellSettings(engine="classic"), progress=None)
    assert list(pd.read_csv(tmp_path / "out" / "counts.csv").columns) == SUMMARY_COLUMNS


def test_validation_metrics(tmp_path):
    pred = tmp_path / "pred.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"image": ["a", "b", "c"], "count": [10, 22, 0]}).to_csv(pred, index=False)
    pd.DataFrame({"image": ["a", "b", "c"], "true_count": [10, 20, 0]}).to_csv(truth, index=False)
    df = validate.compare(pred, truth)
    m = validate.metrics(df)
    assert m["images"] == 3
    assert m["mean_abs_error"] == pytest.approx(2 / 3)
    assert m["within_5pct"] == pytest.approx(0.5)   # c has no % error (true 0)
    assert m["bias"] == pytest.approx(2 / 3)


def test_validation_rejects_bad_truth(tmp_path):
    pred = tmp_path / "pred.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"image": ["a"], "count": [1]}).to_csv(pred, index=False)
    pd.DataFrame({"image": ["a"], "n": [1]}).to_csv(truth, index=False)
    with pytest.raises(ValueError):
        validate.compare(pred, truth)


def test_corrections(tmp_path, plate_run):
    _, out = plate_run
    counts = pd.read_csv(out / "counts.csv")
    name, n = counts.loc[0, "image"], counts.loc[0, "count"]
    fixes = {name: {"removed": [1, 2, 2], "added": [[5, 5]]}}
    corrections.save(out / "corrections.json", fixes)
    loaded = corrections.load(out / "corrections.json")
    assert loaded == json.loads(json.dumps(fixes))
    df = corrections.apply(out / "counts.csv", loaded)
    assert df.loc[0, "corrected_count"] == n - 2 + 1
    assert df.loc[0, "reviewed"] and not df.loc[1, "reviewed"]


def test_cli_count_and_validate(tmp_path, capsys):
    images = tmp_path / "imgs"
    synth.write_dataset(images, "plate", 2, (5, 20), seed=2)
    main(["count", str(images), "--mode", "plate", "--out", str(tmp_path / "r")])
    main(["validate", "--counts", str(tmp_path / "r" / "counts.csv"),
          "--truth", str(images / "truth.csv"), "--out", str(tmp_path / "v")])
    assert (tmp_path / "v" / "agreement.png").exists()
    assert "Images compared:          2" in capsys.readouterr().out
