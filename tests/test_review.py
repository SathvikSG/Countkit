import json
import sys
from pathlib import Path

import pytest

from countkit import pipeline, synth
from countkit.plate import PlateSettings

APP = str(Path(__file__).parents[1] / "countkit" / "review_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("streamlit_image_coordinates")
    images = tmp_path / "imgs"
    synth.write_dataset(images, "plate", 3, (10, 20), seed=4)
    results = tmp_path / "results"
    pipeline.run(images, results, "plate", PlateSettings(engine="classic"), progress=None)
    monkeypatch.setattr(sys, "argv", ["review_app.py", str(results)])
    at = testing.AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    return at, results


def test_removing_several_objects_keeps_all_of_them(app):
    at, results = app
    for oid in (1, 2, 3):
        at.multiselect[0].select(oid).run()
    assert at.multiselect[0].value == [1, 2, 3]
    [b for b in at.button if b.label == "Save corrections"][0].click().run()
    saved = json.loads((results / "corrections.json").read_text())
    first = sorted(saved)[0]
    assert saved[first]["removed"] == [1, 2, 3]


def test_switching_images_sticks(app):
    at, _ = app
    names = at.selectbox[0].options
    at.selectbox[0].select(names[1]).run()
    at.selectbox[0].select(names[2]).run()
    assert at.selectbox[0].value == names[2]
    [b for b in at.button if "Previous" in b.label][0].click().run()
    assert at.selectbox[0].value == names[1]


def test_removals_survive_switching_images(app):
    at, _ = app
    names = at.selectbox[0].options
    at.multiselect[0].select(2).run()
    at.selectbox[0].select(names[1]).run()
    at.selectbox[0].select(names[0]).run()
    assert at.multiselect[0].value == [2]
