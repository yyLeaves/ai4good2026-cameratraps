"""Test-time tricks that need no training, as the iWildCam 2021 winners apply them.

    uv run python -m iwildcam.posthoc --config configs/lpft_dinov2g_vitg14.yaml --alpha 0.5

Replicates the prediction side of github.com/alcunha/iwildcam2021ufam (1st place iWildCam 2021,
following the 2020 winners), full-image branch only:

* flip TTA (`classification/predict_bbox_n_full.py`, `use_flip_image`): the softmax outputs of
  the image and its mirror, weighted equally;
* sequence averaging (`classification/iwildcamlib.py`, `_get_average_prediction`): the mean
  softmax over all images of a camera sequence (`seq_id`), argmax, given to every image of the
  sequence; `_get_majority_vote_prediction` (voting) is scored too, their ablation baseline.

Nothing is tuned, so val is only reported. The model's class log-probabilities on val and test
(plain and mirrored) are saved to `results/posthoc/<run_name>-a<alpha>.npz`, and the test change
against the plain model gets a 95% CI from resampling test cameras.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import (ImageDataset, build_transform, class_texts, crop_task, image_dir_for,
                   load_fold, load_task)
from .models import build_model
from .train import AMP_DTYPE
from .train_utils import RESULTS, load_config, run_name

OUT = RESULTS / "posthoc"


def load_trained(cfg: dict, task, ckpt_dir: str, alpha: float = 1.0):
    """The checkpointed model of `cfg`, WiSE-FT interpolated with weight `alpha` if alpha < 1.

    theta_0 is the pretrained model with the head training started from: the linear-probe head
    for LP-FT runs (`init_head` in the checkpoint), the zero-shot head for CLIP-style runs.
    """
    kw = {k: v for k, v in cfg.get("model_kwargs", {}).items() if k != "drop_path_rate"}
    if kw.get("zeroshot_head"):
        kw["classnames"] = class_texts(task.classes, kw.pop("text_type", "common"))
    model = build_model(cfg["model"], task.n_classes, pretrained=True, **kw)
    ck = torch.load(Path(ckpt_dir) / f"{run_name(cfg)}.pt", map_location="cpu", weights_only=False)
    if alpha < 1:
        if "init_head" in ck:
            model.head.load_state_dict(ck["init_head"])
        theta0 = model.state_dict()
        model.load_state_dict({k: ((1 - alpha) * theta0[k].float() + alpha * v.float()).to(v.dtype)
                               if v.is_floating_point() else v for k, v in ck["model"].items()})
    else:
        model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def log_probs(model, task, idx, size: int, flip: bool, batch_size: int = 128,
              num_workers: int = 8) -> np.ndarray:
    """Class log-probabilities `(len(idx), n_classes)`, of the mirrored images if `flip`."""
    loader = DataLoader(ImageDataset(task, idx, build_transform(size, False, model.mean, model.std)),
                        batch_size=batch_size, num_workers=num_workers)
    out = []
    for xb, _, _ in loader:
        xb = xb.cuda(non_blocking=True)
        if flip:
            xb = xb.flip(-1)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            out.append(model(xb).float().log_softmax(1).cpu())
    return torch.cat(out).numpy()


def sequence_average(p: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """Argmax of the mean probabilities over each sequence, given to all its images."""
    _, inv = np.unique(seq, return_inverse=True)
    sums = np.zeros((inv.max() + 1, p.shape[1]))
    np.add.at(sums, inv, p)
    return sums.argmax(1)[inv]


def sequence_vote(pred: np.ndarray, seq: np.ndarray) -> np.ndarray:
    """Most frequent prediction of each sequence, given to all its images."""
    _, inv = np.unique(seq, return_inverse=True)
    votes = np.zeros((inv.max() + 1, pred.max() + 1), dtype=int)
    np.add.at(votes, (inv, pred), 1)
    return votes.argmax(1)[inv]


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", help="override the config's split")
    ap.add_argument("--alpha", type=float, default=1.0, help="WiSE-FT weight of the fine-tuned model")
    ap.add_argument("--ckpt-dir", default="../checkpoints")
    a = ap.parse_args()
    cfg = load_config(a.config, {"split": a.split})
    task = load_task(cfg.get("min_per_camera", 10), cfg.get("min_cameras", 5),
                     image_dir=cfg.get("image_dir") or image_dir_for(cfg["height"]),
                     require_files=True)
    if cfg.get("crops"):
        task = crop_task(task, cfg["split"], cfg["crops"])
    fold = load_fold(task, cfg["split"])
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"{run_name(cfg)}-a{a.alpha}.npz"
    if cache.exists():
        lp = dict(np.load(cache))
    else:
        model = load_trained(cfg, task, a.ckpt_dir, a.alpha).cuda().eval()
        lp = {f"{s}{'_flip' if f else ''}": log_probs(model, task, fold[s], cfg.get("size", 448), f,
                                                      num_workers=cfg.get("num_workers", 8))
              for s in ("val", "test") for f in (False, True)}
        np.savez_compressed(cache, **lp)
    y = {s: task.y[fold[s]] for s in ("val", "test")}
    seq = {s: task.df["seq_id"].to_numpy()[fold[s]] for s in ("val", "test")}

    def variants(s):
        p, pf = np.exp(lp[s]), np.exp(lp[f"{s}_flip"])
        tta = (p + pf) / 2
        return {"plain": p.argmax(1),
                "flip TTA": tta.argmax(1),
                "flip TTA + seq voting": sequence_vote(tta.argmax(1), seq[s]),
                "flip TTA + seq averaging": sequence_average(tta, seq[s])}

    cams = task.camera[fold["test"]]
    rng = np.random.default_rng(0)
    ucams = np.unique(cams)
    boot = [np.concatenate([np.flatnonzero(cams == c) for c in rng.choice(ucams, len(ucams))])
            for _ in range(1000)]
    val, test = variants("val"), variants("test")
    print(f"{run_name(cfg)}  alpha={a.alpha}  split={cfg['split']}")
    summary = []
    for name, pred in test.items():
        d = np.array([macro_f1(y["test"][b], pred[b]) - macro_f1(y["test"][b], test["plain"][b])
                      for b in boot])
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        row = {"variant": name, "val": macro_f1(y["val"], val[name]),
               "test": macro_f1(y["test"], pred), "test_ci_vs_plain": ci}
        print(f"  {name:26s} val {row['val']:.3f}  test {row['test']:.3f}  "
              f"vs plain 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        summary.append(row)
    (OUT / f"{run_name(cfg)}-a{a.alpha}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
