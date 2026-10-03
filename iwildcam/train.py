"""Training and evaluation.

    uv run python -m iwildcam.train --config configs/reference.yaml
    uv run python -m iwildcam.train --config configs/fast.yaml --split random_burst
    uv run python -m iwildcam.train --config configs/fast.yaml --seeds 0,1,2

`run_fold` is the training loop and the place to add a method. It is written out by hand
so you can read it top to bottom and insert a term -- an extra loss on `features()`, a
domain head reading the third element of each batch -- without hunting for a hook.

The headline metric is macro F1 over the species present in the test split, which is what
WILDS reports, so your numbers stay comparable to the published ones.

Every completed run is appended to `results/runs.jsonl`, identified by `run_key` (this
run) and `config_key` (this setting, across its seeds). Everything the run prints is also
written to `logs/<run_name>.log`, which carries the same `run_key`; set `log_dir: null` in
the config to turn that off.

This file is the experiment: score a loader (`evaluate`), train one fold (`run_fold`), run
it once per seed (`run_one`, `main`). The machinery around it -- the run keys, the log
file, the resume, the config reader -- is in `train_utils.py` and you can leave it alone.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .data import (build_transform, image_dir_for, load_fold, load_task, load_taxonomy,
                   make_loaders)
from .models import build_model
from .train_utils import (JSONL, PRED_DIR, RESULTS, config_key, find_result, fmt_secs,
                          group_name, load_config, per_camera_scores, progress, run_key,
                          run_name, set_seed, tee_console, wandb_run)

# Mixed precision in bfloat16, which has the same exponent range as float32. That is what
# lets the loop call `loss.backward()` directly: float16 gradients underflow to zero and
# need a GradScaler around every step, bfloat16 does not. Ampere and newer only.
AMP_DTYPE = torch.bfloat16


# --- scoring ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, n_classes: int,
             amp: bool = True, desc: str | None = None) -> dict:
    """Run the model over `loader` and score its predictions.

    Args:
        model: the model to evaluate. Put in eval mode and left there.
        loader: yields `(x, y, domain)`; the domain is not used here.
        device: device to run on, e.g. `"cuda"`.
        n_classes: number of species, so absent classes still appear in the report.
        amp: use fp16 autocast.
        desc: label for a progress bar over the batches. None means no bar.

    Returns:
        Dict with `accuracy`, `macro_f1`, `macro_f1_present`, and the `pred` and `true`
        arrays.
    """
    model.eval()
    preds, trues = [], []
    for xb, yb, _domain in progress(loader, desc):
        xb = xb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
            logits = model(xb)
        preds.append(logits.float().argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    labels = list(range(n_classes))
    # Classes absent from the held-out cameras score F1 = 0 and still enter the mean, so
    # macro F1 over ALL labels is capped at (n_present / n_classes) and is not comparable
    # between folds -- a fold with 2 of 5 classes present cannot exceed 0.4 however good
    # the model is. `macro_f1_present` averages only over the species the test cameras
    # actually contain; that is the number to compare across folds.
    present = [c for c in labels if (t == c).any()]
    return {
        "macro_f1": float(f1_score(t, p, average="macro", labels=labels, zero_division=0)),
        "macro_f1_present": float(f1_score(t, p, average="macro", labels=present,
                                           zero_division=0)),
        "accuracy": float(accuracy_score(t, p)),
        "n_classes_present": len(present),
        "per_class_f1": [float(v) for v in
                         f1_score(t, p, average=None, labels=labels, zero_division=0)],
        "support": [int((t == c).sum()) for c in labels],
        "n": int(len(t)),
        "_pred": p, "_true": t,
    }


# --- optimiser set-up for fine-tuning large pretrained backbones -----------------------

def param_groups(model: nn.Module, cfg: dict):
    """AdamW parameter groups. With neither key set, all parameters in one group, as before.

    `head_lr_mult` gives the head `lr * head_lr_mult` (a fresh head next to a backbone
    tuned at a small lr). `wd_skip_1d` exempts biases, norms, layer scales and the
    position / class / register tokens from weight decay, the usual ViT fine-tuning set-up.
    """
    mult, skip = cfg.get("head_lr_mult", 1.0), cfg.get("wd_skip_1d", False)
    if mult == 1.0 and not skip:
        return model.parameters()
    wd = cfg.get("weight_decay", 1e-4)
    head = {id(p) for p in model.head.parameters()}
    groups = {}
    for name, p in model.named_parameters():
        is_head = id(p) in head
        no_wd = skip and (p.ndim <= 1 or name.endswith(("pos_embed", "cls_token", "reg_token")))
        groups.setdefault((is_head, no_wd), {
            "params": [], "lr": cfg["lr"] * (mult if is_head else 1.0),
            "weight_decay": 0.0 if no_wd else wd})["params"].append(p)
    return list(groups.values())


def lr_schedule(opt, cfg: dict, total_steps: int):
    """Per-step LR schedule: `warmup_steps` of linear warmup, then constant or, with
    `lr_schedule: cosine`, cosine decay to zero. None (constant LR, as before) if neither
    is set."""
    warmup, cosine = cfg.get("warmup_steps", 0), cfg.get("lr_schedule") == "cosine"
    if not warmup and not cosine:
        return None

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if not cosine:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


# --- the training loop -----------------------------------------------------------------

def run_fold(cfg: dict) -> dict:
    """Train one model on one fold and score it on the test split.

    Trains for `cfg["epochs"]`, keeping the weights that scored best on validation, then
    evaluates those on the test split. This is the loop to edit when adding a method: the
    training batches are `(x, y, domain)`, so an adaptation term needs no changes
    elsewhere.

    Args:
        cfg: the run config. See `configs/fast.yaml` for every key it reads.

    Returns:
        Scores, per-camera scores, the per-epoch history, and the identifying keys. Test
        predictions are written to `cfg["pred_dir"]` separately.
    """
    started = time.time()
    set_seed(cfg["seed"])
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp", True)) and device == "cuda"

    task = load_task(cfg.get("min_per_camera", 10), cfg.get("min_cameras", 5),
                     image_dir=cfg.get("image_dir") or image_dir_for(cfg["height"]),
                     require_files=True)
    fold = load_fold(task, cfg["split"])

    model_kwargs = dict(cfg.get("model_kwargs", {}))
    if model_kwargs.get("zeroshot_head"):
        model_kwargs["classnames"] = list(load_taxonomy()["common_name"][task.classes])
    model = build_model(cfg["model"], task.n_classes,
                        pretrained=cfg.get("pretrained", True), **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Normalise with the backbone's own constants (ImageNet's for ResNets, as before).
    size = cfg.get("size", 448)
    loaders = make_loaders(task, fold, cfg["batch_size"], size, cfg.get("num_workers", 8),
                           build_transform(size, True, model.mean, model.std),
                           build_transform(size, False, model.mean, model.std))
    print(f"{task.summary()}\nfold {cfg['split']}: "
          + "  ".join(f"{k}={len(v):,}" for k, v in fold.items()), flush=True)
    log = wandb_run(cfg)

    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 1e-4))
    sched = lr_schedule(opt, cfg, cfg["epochs"] * len(loaders["train"]))
    crit = nn.CrossEntropyLoss()

    best, best_state = {"macro_f1_present": -1.0}, None
    hist = []
    verbose = cfg.get("verbose", True)
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_started = time.time()
        model.train()
        total, seen = 0.0, 0
        bar = progress(loaders["train"],
                       f"epoch {epoch}/{cfg['epochs']}" if verbose else None)
        for xb, yb, domain in bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if sched is not None:
                sched.step()
            total += loss.item() * len(yb)
            seen += len(yb)
            # refresh=False: the bar redraws on its own schedule, not once per batch.
            bar.set_postfix_str(f"loss {total / seen:.3f}", refresh=False)

        val = evaluate(model, loaders["val"], device, task.n_classes, amp,
                       desc="val" if verbose else None)
        hist.append({"epoch": epoch, "loss": total / max(seen, 1),
                     "val_macro_f1_present": val["macro_f1_present"],
                     "seconds": round(time.time() - epoch_started, 1)})
        log.log({**hist[-1], "lr": cfg["lr"],
                 "val_accuracy": val["accuracy"]}, step=epoch)
        if val["macro_f1_present"] > best["macro_f1_present"]:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {epoch:2d}  loss {hist[-1]['loss']:.3f}  "
                  f"val macroF1(present) {val['macro_f1_present']:.3f}  "
                  f"{fmt_secs(hist[-1]['seconds'])}", flush=True)

    # Restore the best-validation checkpoint before the held-out evaluation. 
    if best_state is not None:
        model.load_state_dict(best_state)
    # Keep those weights on disk when `ckpt_dir` is set; named by `run_name`, like the log.
    if cfg.get("ckpt_dir"):
        ckpt_path = Path(cfg["ckpt_dir"]) / f"{run_name(cfg)}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "run_key": run_key(cfg),
                    "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                    "val_macro_f1_present": best["macro_f1_present"]}, ckpt_path)
        print(f"checkpoint: {ckpt_path}", flush=True)
    test = evaluate(model, loaders["test"], device, task.n_classes, amp,
                    desc="test" if verbose else None)

    test_cam = task.camera[fold["test"]]
    per_cam = per_camera_scores(test["_pred"], test["_true"], test_cam)

    pred_dir = Path(cfg.get("pred_dir", PRED_DIR))
    pred_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{cfg['split']}_{cfg['model']}_s{cfg['seed']}_sz{cfg.get('size', 448)}"
           f"_{'pre' if cfg.get('pretrained', True) else 'scratch'}_e{cfg['epochs']}")
    np.savez_compressed(pred_dir / f"{tag}.npz", pred=test["_pred"], true=test["_true"],
                        camera=test_cam, index=fold["test"])

    log.log({"test_macro_f1_present": test["macro_f1_present"],
             "test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
             "val_macro_f1_present_best": best["macro_f1_present"],
             "seconds": round(time.time() - started, 1)})
    log.finish()

    return {
        # Two identifiers: `run_key` is this particular run, 
        # `config_key` is this run's config across every seed. Group on the
        # second to average seeds, here and in W&B alike.
        "config_key": config_key(cfg), "group": group_name(cfg),
        "split": cfg["split"], "model": cfg["model"],
        "pretrained": bool(cfg.get("pretrained", True)), "seed": cfg["seed"],
        "size": cfg.get("size", 448), "height": cfg["height"],
        "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "n_params": n_params,
        "seconds": round(time.time() - started, 1),
        "min_per_camera": cfg.get("min_per_camera", 10),
        "min_cameras": cfg.get("min_cameras", 5),
        "n_classes": task.n_classes,
        "n_train": len(fold["train"]), "n_val": len(fold["val"]),
        "n_test": len(fold["test"]),
        "val_macro_f1_present": best["macro_f1_present"],
        "test_macro_f1": test["macro_f1"],
        "test_macro_f1_present": test["macro_f1_present"],
        "test_accuracy": test["accuracy"],
        "test_n_classes_present": test["n_classes_present"],
        "test_per_class_f1": test["per_class_f1"], "test_support": test["support"],
        "test_per_camera": per_cam,
        "pred_file": str(pred_dir / f"{tag}.npz"),
        "history": hist,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }


# --- one run, or the one already recorded ---------------------------------------------

def run_one(cfg: dict) -> dict:
    """Train and evaluate one config, unless it has already been run.

    Each completed run is appended to `results/runs.jsonl` immediately, so a crash part
    way through a sweep does not lose the runs that finished.

    Args:
        cfg: the run config.

    Returns:
        The result dict, either freshly computed or read back from `results/runs.jsonl`.
    """
    key = run_key(cfg)
    cached = find_result(key)
    if cached is not None:
        print(f"skip (already in {JSONL.name}): {cfg.get('split')} "
              f"{cfg.get('model')} seed={cfg.get('seed')} [{key}]")
        return cached
    res = run_fold(cfg)
    res["run_key"] = key
    RESULTS.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a") as fh:
        fh.write(json.dumps(res) + "\n")
    return res


# --- CLI --------------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point. Runs one config, once per seed, and prints a summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--seeds", help="comma-separated seeds, e.g. 0,1,2; runs each and "
                                    "prints the mean and standard deviation")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--model")
    ap.add_argument("--size", type=int)
    # An unset flag must leave a `true` already in the YAML alone, so both directions of
    # this flag default to None rather than to True/False.
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=None)
    ap.add_argument("--out", help="also write the full result of the last run here")
    ap.add_argument("--log-dir", dest="log_dir",
                    help="where the per-run console log goes; \"\" turns it off")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir",
                    help="save the best-validation weights here as <run_name>.pt; "
                         "unset saves nothing")
    ap.add_argument("--wandb", action="store_true", default=None,
                    help="log to Weights & Biases; seeds of one config share a group")
    a = ap.parse_args()

    cfg = load_config(a.config, {"split": a.split, "seed": a.seed, "epochs": a.epochs,
                                 "model": a.model, "size": a.size,
                                 "pretrained": a.pretrained, "wandb": a.wandb,
                                 "log_dir": a.log_dir, "ckpt_dir": a.ckpt_dir})
    seed_list = [int(s) for s in a.seeds.split(",")] if a.seeds else [cfg["seed"]]

    skip = {"test_per_class_f1", "test_support", "history", "test_per_camera"}
    scores = []
    for sd in seed_list:
        run_cfg = dict(cfg, seed=sd)
        # One log file per run, named by `run_name`, holding exactly what the console
        # showed -- including a traceback if the run dies before it reaches runs.jsonl.
        with tee_console(run_cfg) as log_file:
            res = run_one(run_cfg)
            scores.append(res["test_macro_f1_present"])
            print(json.dumps({k: v for k, v in res.items() if k not in skip}, indent=2))
        if log_file is not None:
            print(f"log: {log_file}")
        if a.out:
            out_path = Path(a.out)
            if len(seed_list) > 1:
                out_path = out_path.with_name(f"{out_path.stem}_s{sd}{out_path.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(res, indent=2))

    if len(seed_list) > 1:
        arr = np.array(scores)
        print(f"\n{len(scores)} seeds {seed_list}: test macroF1(present) "
              f"mean {arr.mean():.4f}  std {arr.std():.4f}")


if __name__ == "__main__":
    main()
