import json

import pandas as pd
import pytest

from countkit import corrections, pipeline, synth, validate
from countkit.cells import CellSettings
from countkit.plate import PlateSettings
from countkit.cli import main
from countkit.measure import OBJECT_COLUMNS, SUMMARY_COLUMNS


@pytest.fixture
def plate_run(tmp_path):
    images = tmp_path / "images"
    synth.write_dataset(images, "plate", 3, (10, 40), seed=7)
    out = tmp_path / "out"
    pipeline.run(images, out, "plate", PlateSettings(engine="classic"), progress=None)
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
    main(["count", str(images), "--mode", "plate", "--engine", "classic", "--out", str(tmp_path / "r")])
    main(["validate", "--counts", str(tmp_path / "r" / "counts.csv"),
          "--truth", str(images / "truth.csv"), "--out", str(tmp_path / "v")])
    assert (tmp_path / "v" / "agreement.png").exists()
    assert "Images compared:          2" in capsys.readouterr().out


def test_bad_image_is_skipped_not_fatal(tmp_path):
    images = tmp_path / "images"
    synth.write_dataset(images, "plate", 2, (10, 20), seed=3)
    (images / "broken.jpg").write_bytes(b"not really a jpeg")
    out = tmp_path / "out"
    summaries = pipeline.run(images, out, "plate", PlateSettings(engine="classic"), progress=None)
    assert len(summaries) == 2
    assert len(pd.read_csv(out / "counts.csv")) == 2
    errors = pd.read_csv(out / "errors.csv")
    assert errors["image"].tolist() == ["broken.jpg"]


def test_empty_or_missing_folder_gives_a_plain_error(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    for folder in (tmp_path / "empty", tmp_path / "missing"):
        with pytest.raises(SystemExit) as e:
            main(["count", str(folder), "--mode", "plate", "--out", str(tmp_path / "r")])
        assert str(e.value).startswith("Error:")


def test_wrong_engine_name_is_not_reported_as_bad_images(tmp_path):
    images = tmp_path / "images"
    synth.write_dataset(images, "plate", 1, (10, 20), seed=3)
    with pytest.raises(ValueError, match="plate engine"):
        pipeline.run(images, tmp_path / "out", "plate", PlateSettings(engine="yolo"), progress=None)


def test_validate_warns_about_unmatched_names(tmp_path):
    counts = tmp_path / "counts.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"image": ["a.png", "b.png"], "count": [5, 7]}).to_csv(counts, index=False)
    pd.DataFrame({"image": ["a.png", "c.png"], "true_count": [5, 9]}).to_csv(truth, index=False)
    notes = []
    df = validate.compare(counts, truth, warn=notes.append)
    assert len(df) == 1
    assert any("b.png" in n for n in notes) and any("c.png" in n for n in notes)


def test_validate_scores_reviewed_counts(tmp_path):
    counts = tmp_path / "counts_corrected.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"image": ["a.png"], "count": [5], "corrected_count": [9]}).to_csv(counts, index=False)
    pd.DataFrame({"image": ["a.png"], "true_count": [9]}).to_csv(truth, index=False)
    df = validate.compare(counts, truth, warn=None)
    assert df.loc[0, "predicted"] == 9
