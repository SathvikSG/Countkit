"""Train the colony detector on hand-boxed plate photos (ADBC + AGAR demo).

Data (download first, see README "Colony detector"):
  data/adbc/annot_COCO.json + data/adbc/images/   ADBC, CC BY 4.0, 369 plates
  data/agar_demo_dl/AGAR_representative/          AGAR demo, CC BY-NC 2.0, 40 plates

Splits are seeded and written to data/detector_splits.csv:
  ADBC: per species, 15% test, 15% validation, rest training
  AGAR: the same 20 tuning / 19 held-out plates used for the classic engine;
        the 20 tuning plates are split 16 training / 4 validation (1 per group)

The score cutoff (how sure the model must be to count a box) is picked on the
validation plates only, then saved with the weights. Test plates are never
used for training or picking anything.

Usage:
  python scripts/train_detector.py --epochs 24 --out countkit/weights/colony_detector.pt
  python scripts/train_detector.py --eval-only --out countkit/weights/colony_detector.pt
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from countkit.detector import ANCHORS, SIZE, build_model, pick_device, prepare  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "detector_cache"


# ---------- data ----------

def adbc_records():
    coco = json.loads((DATA / "adbc" / "annot_COCO.json").read_text())
    boxes = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    rng = random.Random(0)
    by_species = {}
    for im in coco["images"]:
        by_species.setdefault(im["file_name"].split("_")[0], []).append(im)
    out = []
    for sp in sorted(by_species):
        ims = sorted(by_species[sp], key=lambda i: i["file_name"])
        rng.shuffle(ims)
        n = len(ims)
        n_test, n_val = max(1, round(0.15 * n)), max(1, round(0.15 * n))
        for k, im in enumerate(ims):
            split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
            out.append(dict(source="adbc", image=im["file_name"], split=split,
                            path=str(DATA / "adbc" / "images" / im["file_name"]),
                            boxes=boxes.get(im["id"], [])))
    return out


def agar_records():
    split = pd.read_csv(DATA / "agar_split.csv")
    split = split[split.true_count >= 0]
    root = DATA / "agar_demo_dl" / "AGAR_representative"
    paths = {p.name: p for p in root.rglob("*.jpg")}
    rng = random.Random(1)
    tune = split[split.split == "tune"]
    val_names = set()
    for _, g in tune.groupby("group"):
        names = sorted(g.image)
        rng.shuffle(names)
        val_names.add(names[0])  # one validation plate per group
    out = []
    for _, r in split.iterrows():
        p = paths[r.image]
        ann = json.loads(p.with_suffix(".json").read_text())
        boxes = [[l["x"], l["y"], l["x"] + l["width"], l["y"] + l["height"]] for l in ann.get("labels", [])]
        s = "test" if r.split == "test" else ("val" if r.image in val_names else "train")
        # Train on every boxed colony, but score against AGAR's own count, which
        # leaves out colonies boxed as "Contamination" (only plate 734 has any).
        out.append(dict(source="agar", image=r.image, split=s, path=str(p), boxes=boxes,
                        true_count=int(ann["colonies_number"])))
    return out


def build_cache(records, size=SIZE):
    """Run prepare() once per photo and store the size x size crop + moved boxes."""
    cache = CACHE if size == SIZE else CACHE / str(size)
    cache.mkdir(parents=True, exist_ok=True)
    for r in records:
        stem = f"{r['source']}_{Path(r['image']).stem}"
        img_p, box_p = cache / f"{stem}.png", cache / f"{stem}.json"
        r["cache"] = str(img_p)
        r.setdefault("true_count", len(r["boxes"]))
        if img_p.exists() and box_p.exists():
            r["cboxes"] = json.loads(box_p.read_text())
            continue
        color = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        rgb, (x0, y0, scale) = prepare(color, size)
        b = (np.array(r["boxes"], dtype=np.float64).reshape(-1, 4) - [x0, y0, x0, y0]) * scale
        clipped = np.clip(b, 0, size)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        inside = (clipped[:, 2] - clipped[:, 0]) * (clipped[:, 3] - clipped[:, 1])
        keep = (inside >= 0.5 * area) & (clipped[:, 2] - clipped[:, 0] >= 2) & (clipped[:, 3] - clipped[:, 1] >= 2)
        r["cboxes"] = clipped[keep].round(2).tolist()
        cv2.imwrite(str(img_p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        box_p.write_text(json.dumps(r["cboxes"]))
    return records


def augment(img, boxes, rng):
    """Flips, 90 degree turns and color jitter. img is HxWx3 uint8, boxes Nx4."""
    s = img.shape[0]
    if rng.random() < 0.5:
        img = img[:, ::-1]
        boxes = np.stack([s - boxes[:, 2], boxes[:, 1], s - boxes[:, 0], boxes[:, 3]], 1)
    if rng.random() < 0.5:
        img = img[::-1]
        boxes = np.stack([boxes[:, 0], s - boxes[:, 3], boxes[:, 2], s - boxes[:, 1]], 1)
    if rng.random() < 0.5:  # transpose = turn + flip, together with the flips covers all 8
        img = img.transpose(1, 0, 2)
        boxes = boxes[:, [1, 0, 3, 2]]
    img = np.ascontiguousarray(img).astype(np.float32)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)          # contrast, brightness
    img = img * rng.uniform(0.9, 1.1, size=3)                            # color balance
    if rng.random() < 0.3:
        gray = img.mean(2, keepdims=True)
        img = gray + (img - gray) * rng.uniform(0, 1.2)                  # saturation
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def random_crop(img, boxes, rng, size):
    """A random size x size piece of the plate. Training on crops keeps memory
    in check on plates with 700+ colonies; colonies stay the same size, and
    counting still runs on the whole plate. A box is kept if at least half of
    it is inside the crop."""
    s = img.shape[0]
    x, y = (int(rng.integers(0, s - size + 1)) for _ in range(2))
    img = img[y:y + size, x:x + size]
    b = boxes - [x, y, x, y]
    c = np.clip(b, 0, size)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inside = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    keep = (inside >= 0.5 * np.maximum(area, 1e-6)) & (c[:, 2] - c[:, 0] >= 2) & (c[:, 3] - c[:, 1] >= 2)
    return img, c[keep]


def to_tensor(img):
    import torch
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)


# ---------- train / eval ----------

def predict_scores(model, device, recs):
    """Return a list of score arrays (one per record)."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for r in recs:
            img = cv2.cvtColor(cv2.imread(r["cache"]), cv2.COLOR_BGR2RGB)
            pred = model([to_tensor(img).to(device)])[0]
            out.append(pred["scores"].cpu().numpy())
    return out


def count_metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ape = np.abs(pred - true) / np.maximum(true, 1) * 100
    return dict(n=len(true), true=int(true.sum()), counted=int(pred.sum()),
                mae=round(float(np.abs(pred - true).mean()), 2),
                mape=round(float(ape.mean()), 2), medape=round(float(np.median(ape)), 2),
                within5=round(float((ape <= 5).mean() * 100), 1),
                within10=round(float((ape <= 10).mean() * 100), 1),
                bias=round(float((pred - true).mean()), 2),
                r=round(float(np.corrcoef(pred, true)[0, 1]), 4) if len(true) > 2 else None)


def pick_threshold(scores, true):
    best = None
    for t in np.arange(0.20, 0.96, 0.01):
        pred = [int((s >= t).sum()) for s in scores]
        m = count_metrics(pred, true)
        if best is None or m["mape"] < best[1]["mape"]:
            best = (round(float(t), 2), m)
    return best


def evaluate(model, device, recs, threshold, label):
    rows = []
    groups = {}
    scores = predict_scores(model, device, recs)
    for r, s in zip(recs, scores):
        n = int((s >= threshold).sum())
        rows.append(dict(source=r["source"], image=r["image"], split=r["split"],
                         true_count=r["true_count"], predicted=n))
        groups.setdefault(r["source"], []).append((n, r["true_count"]))
    print(f"\n== {label} (score cutoff {threshold}) ==")
    for src, pairs in groups.items():
        p, t = zip(*pairs)
        print(src, json.dumps(count_metrics(p, t)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=512, help="training crop size in px")
    ap.add_argument("--crops-per-plate", type=int, default=2, help="crops per plate per epoch")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default=str(ROOT / "countkit" / "weights" / "colony_detector.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-agar", action="store_true", help="train on ADBC only")
    ap.add_argument("--max-iters", type=int, default=0, help="stop early (for timing tests)")
    ap.add_argument("--size", type=int, default=SIZE, help="plate crop size in px (1024 or 2048)")
    ap.add_argument("--anchors", default=",".join(map(str, ANCHORS)),
                    help="anchor size per FPN level, in px at --size")
    ap.add_argument("--init", default="", help="start from these weights instead of COCO")
    a = ap.parse_args()

    import torch
    torch.manual_seed(0)
    recs = adbc_records() + agar_records()
    build_cache(recs, a.size)
    anchors = tuple(int(x) for x in a.anchors.split(","))
    pd.DataFrame([{k: r[k] for k in ("source", "image", "split", "true_count")} for r in recs]) \
        .to_csv(DATA / "detector_splits.csv", index=False)
    train = [r for r in recs if r["split"] == "train" and not (a.no_agar and r["source"] == "agar")]
    val = [r for r in recs if r["split"] == "val"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"train {len(train)}, val {len(val)}, test {len(test)} plates; "
          f"train colonies {sum(len(r['cboxes']) for r in train)}")

    device = pick_device(a.device)
    model = build_model(size=a.size, anchors=anchors)
    out = Path(a.out)

    if not a.eval_only:
        if a.init:  # e.g. the 1024 px model, to train the 2048 px one faster
            init = torch.load(a.init, map_location="cpu", weights_only=True)["model"]
            model.load_state_dict({k: v.float() for k, v in init.items()})
        else:  # COCO weights for everything except the 2-class box head
            from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                                      fasterrcnn_resnet50_fpn_v2)
            coco = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).state_dict()
            own = model.state_dict()
            model.load_state_dict({k: v for k, v in coco.items() if k in own and own[k].shape == v.shape},
                                  strict=False)
        model.rpn._pre_nms_top_n["training"] = 4000
        model.rpn._post_nms_top_n["training"] = 2000
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 512
        model.transform.min_size, model.transform.max_size = (a.crop,), a.crop  # don't upscale crops
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
        steps_per_epoch = math.ceil(len(train) * a.crops_per_plate / a.batch)
        total = steps_per_epoch * a.epochs
        warm = min(300, total // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))
        rng = np.random.default_rng(0)
        # Keep 1024 px plates in memory; read bigger ones from disk as needed.
        keep = a.size <= SIZE
        imgs = {r["cache"]: cv2.cvtColor(cv2.imread(r["cache"]), cv2.COLOR_BGR2RGB) for r in train} if keep else {}
        load_img = (lambda c: imgs[c]) if keep else (lambda c: cv2.cvtColor(cv2.imread(c), cv2.COLOR_BGR2RGB))
        step, t0 = 0, time.time()
        for ep in range(a.epochs):
            model.train()
            order = rng.permutation(np.repeat(np.arange(len(train)), a.crops_per_plate))
            losses = []
            for i in range(0, len(order), a.batch):
                batch_imgs, targets = [], []
                for j in order[i:i + a.batch]:
                    r = train[j]
                    img, b = augment(load_img(r["cache"]), np.array(r["cboxes"], np.float32).reshape(-1, 4), rng)
                    img, b = random_crop(img, b, rng, a.crop)
                    batch_imgs.append(to_tensor(img).to(device))
                    targets.append({"boxes": torch.as_tensor(b, dtype=torch.float32, device=device),
                                    "labels": torch.ones(len(b), dtype=torch.int64, device=device)})
                loss = sum(model(batch_imgs, targets).values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
                step += 1
                if a.max_iters and step >= a.max_iters:
                    print(f"{step} iters in {time.time() - t0:.1f}s ({(time.time() - t0) / step:.2f}s/iter)")
                    return
            print(f"epoch {ep + 1}/{a.epochs} loss {np.mean(losses):.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)  # save every epoch in case the run dies
            torch.save({"model": {k: v.half() for k, v in model.state_dict().items()},
                        "score_threshold": 0.5, "size": a.size, "anchors": list(anchors)}, out)
        model.transform.min_size, model.transform.max_size = (a.size,), a.size

    ckpt = torch.load(out, map_location="cpu", weights_only=True)
    if int(ckpt.get("size", SIZE)) != a.size:
        sys.exit(f"{out} was trained at --size {ckpt.get('size', SIZE)}; pass that --size to evaluate it")
    model.load_state_dict({k: v.float() for k, v in ckpt["model"].items()})
    model.to(device)

    val_scores = predict_scores(model, device, val)
    t, m = pick_threshold(val_scores, [r["true_count"] for r in val])
    print(f"\nscore cutoff picked on validation plates: {t}  ({json.dumps(m)})")
    ckpt["score_threshold"] = t
    torch.save(ckpt, out)

    rows = evaluate(model, device, test, t, "TEST plates (never trained on)")
    pd.DataFrame(rows).to_csv(DATA / "detector_test_counts.csv", index=False)
    print(f"\nWeights: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
