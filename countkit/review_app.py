"""Review page: flip through overlays, fix the count, save corrections.

Launch with:  countkit review <results folder>
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from countkit import corrections as corr

results = Path(sys.argv[-1]) if len(sys.argv) > 1 else Path("countkit_results")
counts_path = results / "counts.csv"
corrections_path = results / "corrections.json"

st.set_page_config(page_title="CountKit review", layout="wide")
st.title("CountKit review")

if not counts_path.exists():
    st.error(f"No counts.csv in {results}. Run `countkit count` first.")
    st.stop()

counts = pd.read_csv(counts_path)
objects = pd.read_csv(results / "objects.csv")
if "corrections" not in st.session_state:
    st.session_state.corrections = corr.load(corrections_path)
fixes = st.session_state.corrections

names = counts["image"].tolist()
if "idx" not in st.session_state:
    st.session_state.idx = 0

nav = st.columns([1, 1, 6])
if nav[0].button("◀ Previous") and st.session_state.idx > 0:
    st.session_state.idx -= 1
if nav[1].button("Next ▶") and st.session_state.idx < len(names) - 1:
    st.session_state.idx += 1
name = nav[2].selectbox("Image", names, index=st.session_state.idx)
st.session_state.idx = names.index(name)

row = counts.loc[counts["image"] == name].iloc[0]
entry = fixes.setdefault(name, {"removed": [], "added": []})
overlay_path = results / "overlays" / f"{Path(name).stem}_overlay.png"

left, right = st.columns([3, 1])
with left:
    img = Image.open(overlay_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = max(4, img.width // 150)
    for x, y in entry["added"]:
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255), width=3)
    ids = objects.loc[objects["image"] == name].set_index("object_id")
    for oid in entry["removed"]:
        if oid in ids.index:
            x, y = ids.loc[oid, ["x", "y"]]
            draw.line([x - r, y - r, x + r, y + r], fill=(255, 0, 0), width=3)
            draw.line([x - r, y + r, x + r, y - r], fill=(255, 0, 0), width=3)

    display_width = min(img.width, 900)
    st.caption("Click a spot to mark a missed object (magenta circle).")
    click = streamlit_image_coordinates(img, width=display_width, key=f"img-{name}")
    # The component keeps returning the last click on every rerun, so only act on new ones.
    if click and click != st.session_state.get(f"last-click-{name}"):
        st.session_state[f"last-click-{name}"] = click
        scale = img.width / display_width
        point = [round(click["x"] * scale, 1), round(click["y"] * scale, 1)]
        if point not in entry["added"]:
            entry["added"].append(point)
            st.rerun()

with right:
    # Read the remove list first so the corrected count below is current.
    entry["removed"] = st.multiselect(
        "Remove objects (by number on the overlay)",
        options=ids.index.tolist(), default=entry["removed"])
    st.metric("Auto count", int(row["count"]))
    st.metric("Corrected count", corr.corrected_count(row["count"], entry))
    if isinstance(row["flags"], str) and row["flags"]:
        st.warning(f"Flags: {row['flags']}")
    if st.button("Undo last added point") and entry["added"]:
        entry["added"].pop()
        st.rerun()
    if st.button("Save corrections", type="primary"):
        corr.save(corrections_path, fixes)
        corr.apply(counts_path, fixes).to_csv(results / "counts_corrected.csv", index=False)
        st.success(f"Saved {corrections_path.name} and counts_corrected.csv")
