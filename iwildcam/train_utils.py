"""Everything a run needs that is not the training loop itself.

Four groups, in order:

* **run identity** -- `run_key` (one run) and `config_key` (one setting across its seeds).
  These decide what counts as the same run, so `run_one` can skip a repeat and W&B can
  average the seeds of one config.
* **local logging** -- `tee_console`, always on: everything a run prints also lands in
  `logs/<run_name>.log`. `progress` draws the in-epoch bars, on the terminal only.
* **W&B logging** -- `wandb_run`, off unless you ask for it, with a do-nothing stand-in so
  the training loop needs no `if`.
* **resume and config** -- `find_result` reads back a completed run, `load_config` merges
  a YAML file with the command line.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
JSONL = RESULTS / "runs.jsonl"
PRED_DIR = RESULTS / "preds"
LOG_DIR = ROOT / "logs"


# --- reproducibility and per-camera reporting -----------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch so a run is reproducible.

    Args:
        seed: the seed to set everywhere.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_camera_scores(pred: np.ndarray, true: np.ndarray,
                      camera: np.ndarray) -> list[dict]:
    """Score each test camera separately.

    Test images are spread very unevenly across cameras, so one large site can carry the
    aggregate score. Use this to check whether a gain is real or came from one camera.

    Args:
        pred: predicted class index per image.
        true: true class index per image.
        camera: camera id per image.

    Returns:
        One dict per camera -- `camera`, `n`, `accuracy`, `macro_f1_present` -- sorted by
        image count, largest first.
    """
    rows = []
    for c in np.unique(camera):
        m = camera == c
        tt, pp = true[m], pred[m]
        present = [int(k) for k in np.unique(tt)]
        rows.append({
            "camera": int(c),
            "n": int(m.sum()),
            "n_species": len(present),
            "macro_f1_present": float(f1_score(tt, pp, average="macro", labels=present,
                                               zero_division=0)),
            "accuracy": float(accuracy_score(tt, pp)),
            "top_species_frac": float(np.bincount(tt).max() / len(tt)),
        })
    return sorted(rows, key=lambda r: -r["n"])


# --- run identity: one hash per run, one per config ------------------------------------

# Everything that changes what a run computes. Anything not in here is either recorded
# output, a property of the machine, or logging -- never of the experiment.
IGNORED_IN_KEY = {"device", "num_workers", "verbose", "amp", "out", "image_dir",
                  "wandb", "wandb_project", "log_dir", "ckpt_dir", "grad_checkpointing"}


def _fingerprint(d: dict, ignore: set[str]) -> str:
    """Hash a config, skipping `ignore` and any key starting with an underscore.

    Args:
        d: the config.
        ignore: keys to leave out of the hash.

    Returns:
        The first 16 hex characters of the SHA1.
    """
    fields = {k: v for k, v in d.items() if k not in ignore and not k.startswith("_")}
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def run_key(d: dict) -> str:
    """Identify one run: a hash of every config field that can change its result.

    Two runs with the same `run_key` compute the same thing, so `run_one` reuses the
    recorded result instead of repeating it. The whole config is hashed, not a chosen
    subset, so changing any hyperparameter really does give you a new run. Fields in
    `IGNORED_IN_KEY` are left out because they cannot change the numbers.

    Args:
        d: the run config.

    Returns:
        A 16-character hex string.
    """
    return _fingerprint(d, IGNORED_IN_KEY)


def config_key(d: dict) -> str:
    """Identify one setting: `run_key` with the seed left out.

    Every seed of one config shares a `config_key`, so this is what you group on to
    average seeds, in W&B or in `results/runs.jsonl`.

    Args:
        d: the run config.

    Returns:
        A 16-character hex string.
    """
    return _fingerprint(d, IGNORED_IN_KEY | {"seed"})


def _stem(cfg: dict) -> str:
    """The readable part of a run or group name: the settings you scan for.

    Args:
        cfg: the run config.

    Returns:
        A stem like `official_ood-resnet18-224-pre-e10`.
    """
    return (f"{cfg['split']}-{cfg['model']}-{cfg.get('size', 448)}"
            f"-{'pre' if cfg.get('pretrained', True) else 'scratch'}-e{cfg['epochs']}")


def group_name(cfg: dict) -> str:
    """A readable name for the seeds sharing one config.

    The settings you scan for go in front, and `config_key` on the end keeps two
    otherwise similar configs from merging.

    Args:
        cfg: the run config.

    Returns:
        A name like `official_ood-resnet18-224-pre-e10-75d665dc`.
    """
    return f"{_stem(cfg)}-{config_key(cfg)[:8]}"


def run_name(cfg: dict) -> str:
    """A readable, unique name for one run: the settings, the seed, and `run_key`.

    Used for the W&B run name and for the local log file. The hash is `run_key`, not
    `config_key`, so two runs that differ only in a hyperparameter not shown in the stem
    still get different names.

    Args:
        cfg: the run config.

    Returns:
        A name like `official_ood-resnet18-224-pre-e10-seed0-9f1c2ab3`.
    """
    return f"{_stem(cfg)}-seed{cfg['seed']}-{run_key(cfg)[:8]}"


# --- what a run shows on screen -------------------------------------------------------

# The real terminal, grabbed at import time -- before `tee_console` swaps `sys.stderr` for
# a stream that also writes to the log file. Progress bars go here so their thousands of
# carriage-return redraws stay on screen and out of `logs/<run_name>.log`.
CONSOLE = sys.stderr


def progress(iterable, desc: str | None, total: int | None = None) -> tqdm:
    """Wrap `iterable` in a progress bar, drawn on the terminal and nowhere else.

    The bar draws nothing when `desc` is None or the terminal is not interactive (a
    redirected run, `nohup`, a batch queue), where a redrawing bar is only noise. It is
    still a `tqdm`, so `set_postfix_str` works either way and the caller needs no `if`.
    The per-epoch summary line prints regardless, so a quiet bar loses nothing.

    Args:
        iterable: what to iterate over, e.g. a DataLoader.
        desc: label shown at the left of the bar. None turns the bar off.
        total: number of steps, if `len(iterable)` is not right.

    Returns:
        A `tqdm` around `iterable`, drawing or silent.
    """
    return tqdm(iterable, desc=desc, total=total, file=CONSOLE, leave=False,
                disable=desc is None or not CONSOLE.isatty(),
                dynamic_ncols=True, mininterval=0.5, unit="batch")


def fmt_secs(seconds: float) -> str:
    """A short, readable duration: `48.3s` under a minute, `12m04s` above it.

    Args:
        seconds: the duration to format.

    Returns:
        A string like `48.3s` or `12m04s`.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


# --- always-on local logging ----------------------------------------------------------

class _Tee:
    """Writes every line to several streams at once.

    Stands in for `sys.stdout` so a run still prints to the console while the same text
    lands in its log file.
    """

    def __init__(self, *streams) -> None:
        """Args:
            streams: the streams to write to, in order.
        """
        self.streams = streams

    def write(self, text: str) -> int:
        """Write `text` to every stream and return how much was written."""
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush every stream."""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """Report a plain stream, so libraries do not emit terminal control codes."""
        return False


def log_path(cfg: dict) -> Path | None:
    """Where this run's console log goes, or None when local logging is off.

    Args:
        cfg: the run config. Reads `log_dir`; an empty value turns logging off.

    Returns:
        The path to the run's `.log` file, named by `run_name`.
    """
    log_dir = cfg.get("log_dir", LOG_DIR)
    if not log_dir:
        return None
    return Path(log_dir) / f"{run_name(cfg)}.log"


@contextlib.contextmanager
def tee_console(cfg: dict):
    """Mirror everything printed inside the block into this run's log file.

    Always on, unlike W&B: a run that scrolled off the terminal, or that died halfway,
    still leaves the epoch lines and the traceback on disk under a name you can match
    back to `results/runs.jsonl` by `run_key`. The file is appended to, so re-running the
    same config adds a new block instead of erasing the last one.

    Args:
        cfg: the run config. `log_dir: null` turns this off and prints as before.

    Yields:
        The log file's path, or None when logging is off.
    """
    path = log_path(cfg)
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                 f"{' '.join(sys.argv)}\n")
        fh.write(f"run_key {run_key(cfg)}  config_key {config_key(cfg)}\n")
        fh.write(json.dumps({k: v for k, v in cfg.items()
                             if not k.startswith("_")}, sort_keys=True, default=str)
                 + "\n\n")
        fh.flush()
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, fh), _Tee(err, fh)
        try:
            yield path
        except BaseException:
            # The interpreter prints the traceback only after this block has restored the
            # streams, so a crashing run would otherwise leave a log that stops mid-epoch
            # with no reason given. Write it here, while the file is still open.
            traceback.print_exc(file=fh)
            raise
        finally:
            sys.stdout, sys.stderr = out, err


# --- optional Weights & Biases logging --------------------------------------------------

class _NoLogger:
    """Stands in for a W&B run when logging is off, so the loop needs no `if`."""

    def log(self, *_a, **_k) -> None:
        """Discard whatever was logged."""

    def finish(self) -> None:
        """Do nothing."""


def wandb_run(cfg: dict):
    """Start a Weights & Biases run, or a do-nothing stand-in when logging is off.

    Runs are grouped by `group_name`, so W&B folds the seeds of one config into a single
    line with a mean and a spread band.

    Args:
        cfg: the run config. Reads `wandb` to decide whether to log at all, and
            `wandb_project` for the project name. `WANDB_PROJECT` overrides it.

    Returns:
        A W&B run, or an object with the same `log`/`finish` methods that does nothing.

    Raises:
        SystemExit: if `wandb` is on but the package is not installed.
    """
    if not cfg.get("wandb"):
        return _NoLogger()
    try:
        import wandb
    except ImportError as e:                                   # pragma: no cover
        raise SystemExit("--wandb needs the extra: uv sync --extra wandb") from e
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT") or cfg.get("wandb_project",
                                                           "ai4good-iwildcam"),
        group=group_name(cfg),
        name=run_name(cfg),
        job_type=cfg["split"],
        config={**cfg, "config_key": config_key(cfg), "run_key": run_key(cfg)},
    )


# --- resume: never lose a completed run to a crash ------------------------------------
# The keys themselves are defined above, next to the W&B grouping that reuses them.


def find_result(key: str) -> dict | None:
    """Look up an already-recorded run.

    Args:
        key: the `run_key` to look for.

    Returns:
        The recorded result, or None if `results/runs.jsonl` has no run with that key.
    """
    if not JSONL.exists():
        return None
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("run_key") == key:
            return r
    return None


# --- configs --------------------------------------------------------------------------

def load_config(path: str, overrides: dict) -> dict:
    """Read a YAML config and apply command-line overrides.

    Args:
        path: path to the YAML file.
        overrides: values from the command line. Entries that are None are ignored, so
            an unset flag never overwrites what the file says.

    Returns:
        The merged config.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
