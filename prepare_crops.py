#!/usr/bin/env python
"""Cut MegaDetector animal boxes out of the original frames, as the iWildCam 2021 winners do.

    uv run python prepare_crops.py --source ~/iWildCam2020 --size 448

The 1st-place iWildCam 2021 solution (github.com/alcunha/iwildcam2021ufam, following the 2020
winners) trains a classifier on MegaDetector boxes and fuses it with a full-image one. Its crop
(`classification/preprocessing.py`, `square_crop`) is replicated here: a square of side
max(box width, box height) in pixels, capped at the frame's shorter side, centred on the box and
shifted back inside the frame, no padding, then resized to `--size` x `--size`. Every animal box
with confidence >= `--conf` is cut (0.6, their `conf_threshold`).

Crops come from the original frames, not from `images_h*/`: in a 224px squashed frame the
median animal is about 77 px.

Writes `data/crops_s<size>/<image_id>_<rank>.jpg` (rank 0 = the most confident box of the
image) and `data/crops_s<size>/index.csv`. Resumable, like prepare.py.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from prepare import DATA, JPEG_QUALITY, METADATA, RESAMPLE, find_images

ANIMAL = "1"                       # MegaDetector category id of "animal"


def load_boxes(md_json: Path, conf: float) -> dict[str, list[tuple[float, list[float]]]]:
    """`{image_id: [(conf, [x, y, w, h]), ...]}`, animal boxes >= `conf`, most confident first."""
    md = json.loads(Path(md_json).read_text())
    boxes = {}
    for im in md["images"]:
        dets = sorted(((d["conf"], d["bbox"]) for d in im.get("detections") or []
                       if d["category"] == ANIMAL and d["conf"] >= conf), key=lambda t: -t[0])
        if dets:
            boxes[im["id"]] = dets
    return boxes


def square_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel `(left, top, right, bottom)` of the winners' square crop around a normalised box."""
    x, y, w, h = bbox
    side = min(max(round(w * width), round(h * height), 1), width, height)
    cx, cy = round((x + w / 2) * width), round((y + h / 2) * height)
    left = min(max(0, cx - side // 2), width - side)
    top = min(max(0, cy - side // 2), height - side)
    return left, top, left + side, top + side


def crop_one(src: Path, jobs: list[tuple[Path, list[float]]], size: int) -> str:
    """Write every crop of one frame. Returns "" on success, else the error."""
    if not src.exists():
        return "missing from the archive"
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            for dst, bbox in jobs:
                out = im.crop(square_box(bbox, im.width, im.height)).resize((size, size), RESAMPLE)
                buf = io.BytesIO()
                out.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=False)
                tmp = dst.with_name(dst.name + ".part")      # write then rename, as prepare.py
                tmp.write_bytes(buf.getvalue())
                os.replace(tmp, dst)
        return ""
    except Exception as e:                                    # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, type=Path,
                    help="folder holding the unzipped archive's `train/` directory")
    ap.add_argument("--md", type=Path, help="MegaDetector results json; default: the one in "
                    "--source (iwildcam2020_megadetector_results.json, MegaDetector v3)")
    ap.add_argument("--size", type=int, default=448, help="side of the saved square crops")
    ap.add_argument("--conf", type=float, default=0.6, help="minimum box confidence")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="only the first N frames (a test)")
    a = ap.parse_args()

    src_dir = find_images(a.source.expanduser())
    md_json = a.md or a.source.expanduser() / "iwildcam2020_megadetector_results.json"
    boxes = load_boxes(md_json, a.conf)
    meta = pd.read_csv(METADATA, usecols=["image_id", "file_name", "category"])
    meta = meta[(meta["category"] != "empty") & meta["image_id"].isin(boxes)]
    if a.limit:
        meta = meta.head(a.limit)
    out = DATA / f"crops_s{a.size}"
    out.mkdir(parents=True, exist_ok=True)

    rows, todo = [], []
    for image_id, file_name in zip(meta["image_id"], meta["file_name"]):
        jobs = []
        for rank, (conf, bbox) in enumerate(boxes[image_id]):
            name = f"{image_id}_{rank}.jpg"
            rows.append({"image_id": image_id, "rank": rank, "conf": conf, "x": bbox[0],
                         "y": bbox[1], "w": bbox[2], "h": bbox[3], "file": name})
            if not (out / name).exists():
                jobs.append((out / name, bbox))
        if jobs:
            todo.append((src_dir / file_name, jobs))
    print(f"{len(meta):,} frames with an animal box >= {a.conf} ({md_json.name}), "
          f"{len(rows):,} crops; {sum(len(j) for _, j in todo):,} to write to {out}")

    errors: dict[str, int] = {}
    todo.sort(key=lambda t: t[0].name)               # roughly sequential reads, as prepare.py
    with ThreadPoolExecutor(a.workers) as ex, tqdm(total=len(todo), unit="frame") as bar:
        queue = iter(todo)
        pending = deque(ex.submit(crop_one, s, j, a.size) for s, j in islice(queue, 4 * a.workers))
        while pending:
            err = pending.popleft().result()
            for s, j in islice(queue, 1):
                pending.append(ex.submit(crop_one, s, j, a.size))
            if err:
                errors[err.split(":")[0]] = errors.get(err.split(":")[0], 0) + 1
            bar.update(1)
    index = pd.DataFrame(rows)
    index = index[[(out / f).exists() for f in index["file"]]]
    index.to_csv(out / "index.csv", index=False)
    print(f"index: {len(index):,} crops of {index['image_id'].nunique():,} frames -> {out / 'index.csv'}")
    if errors:
        print(f"{sum(errors.values()):,} frames failed -- "
              + ", ".join(f"{k} x{v}" for k, v in sorted(errors.items())))


if __name__ == "__main__":
    main()
