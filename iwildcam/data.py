"""Loading the task, the image files, and the datasets built on top of them.

`load_task` is the entry point. It returns a `Task`: the labelled images, the class set,
and the `empty` frames kept separately as unlabelled data.

Two rules hold throughout:

1. **The class set comes from the full metadata, not from the files on disk.** Which
   images you happen to have prepared must not change what the classes are.
2. **`empty` frames are not part of the labelled task.** A third of the images contain no
   animal. They are handed back unlabelled by `unlabelled_frames`, for adaptation methods.

The two evaluation folds are shipped as columns of the metadata rather than computed
here, so every student gets the same partition however much of the archive they have
prepared. `load_fold` reads them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Image files do NOT live in the repo. `prepare.py` writes resized copies into
# data/images_h<H>/; set IWILDCAM_DATA_ROOT to keep them on another disk.
IMG_ROOT = Path(os.environ.get("IWILDCAM_DATA_ROOT", DATA))
# 224 to match `prepare.py`'s default, so a bare `prepare.py --source X` and a bare
# `load_task()` agree about where the images are.
DEFAULT_HEIGHT = int(os.environ.get("IWILDCAM_HEIGHT", 224))


def image_dir_for(height: int = DEFAULT_HEIGHT) -> Path:
    """Path to the folder of resized JPEGs that `prepare.py` writes.

    Any folder of JPEGs named by `file_name` works, so `image_dir:` in a config may point
    somewhere else -- at the unzipped archive, to read the originals.

    Args:
        height: the height the images were resized to, e.g. 224.

    Returns:
        Path to `images_h<height>/`.
    """
    return IMG_ROOT / f"images_h{height}"


IMG_DIR = image_dir_for()

# ImageNet statistics. These are fixed constants of the pretrained backbone, not
# statistics of our training set, so using them leaks nothing about held-out cameras.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EMPTY = "empty"

# Annotator fallbacks and camera-operation markers, not animals: not a visual category a
# model can learn, so each would score near zero and depress macro F1 for a reason
# unrelated to transfer. WILDS drops exactly these categories too.
JUNK_CATEGORIES = {"motorcycle", "unknown", "unidentifiable", "misfire", "start", "end"}


def _is_junk(category: pd.Series) -> pd.Series:
    """True where `category` is in `JUNK_CATEGORIES` or starts with `"unknown "`."""
    return category.isin(JUNK_CATEGORIES) | category.str.startswith("unknown ")


@dataclass
class Task:
    """The labelled task and the unlabelled pool, kept apart.

    Attributes:
        df: one row per labelled image, with `y` as the class index. `empty` frames and
            junk categories are already removed.
        empty: the `empty` frames, carrying no species label.
        classes: species names; `classes[y]` is the name of class `y`.
        image_dir: folder the JPEGs are read from.
    """

    df: pd.DataFrame           # labelled rows: one per image, `empty` and junk removed
    empty: pd.DataFrame        # the unlabelled pool: `empty` frames, no species label
    classes: list[str]         # index -> species name
    image_dir: Path

    @property
    def n_classes(self) -> int:
        """Number of species."""
        return len(self.classes)

    @property
    def y(self) -> np.ndarray:
        """Class index of every labelled image."""
        return self.df["y"].to_numpy()

    @property
    def camera(self) -> np.ndarray:
        """Camera id of every labelled image. Stored in the `location` column."""
        return self.df["location"].to_numpy()

    def cameras(self) -> np.ndarray:
        """The distinct camera ids in the task, sorted."""
        return np.unique(self.camera)

    @property
    def n_domains(self) -> int:
        """Number of distinct cameras. Size a domain classifier head to this."""
        return len(self.cameras())

    @property
    def domain(self) -> np.ndarray:
        """The camera of each image as an index in `0..n_domains-1`.

        Camera ids themselves are sparse, so they cannot be used as classifier targets.
        The mapping is fixed for the whole task, so a camera has the same index in train,
        val and test.
        """
        return np.searchsorted(self.cameras(), self.camera)

    @property
    def burst_col(self) -> str:
        """Name of the column holding the burst id splits must not break.

        Raises:
            KeyError: if the metadata has no `burst_id`. `seq_id` is not a substitute --
                it is the raw camera sequence, capped at 10 frames, so one burst spans
                several of them and splitting on it would leak near-duplicate frames.
        """
        if "burst_id" not in self.df.columns:
            raise KeyError("metadata has no `burst_id` column")
        return "burst_id"

    def summary(self) -> str:
        """One line: image, species, camera and burst counts."""
        return (f"{len(self.df):,} labelled images, {self.n_classes} species, "
                f"{self.df['location'].nunique()} cameras, "
                f"{self.df[self.burst_col].nunique():,} bursts; "
                f"{len(self.empty):,} unlabelled empty frames "
                f"from {self.empty['location'].nunique()} cameras")


METADATA = DATA / "metadata.csv.gz"


def load_metadata(path: Path | None = None) -> pd.DataFrame:
    """Read the metadata table: one row per image, including the `empty` frames.

    Args:
        path: metadata file to read. Defaults to the shipped `data/metadata.csv.gz`.

    Returns:
        DataFrame with `file_name`, `location`, `category`, `datetime`, `burst_id` and
        the EXIF columns. 201,399 rows.

        `burst_id` is the unit a split must not break. `seq_id` is the raw sequence the
        camera recorded, capped at 10 frames, so one burst usually spans several of them;
        it is kept for reference and nothing here splits on it. `official_ood` and
        `random_burst` hold this image's split -- train, val or test -- for each fold.
    """
    return pd.read_csv(path or METADATA, parse_dates=["datetime", "date"])


def load_task(min_per_camera: int = 10, min_cameras: int = 5,
              metadata: Path | None = None, image_dir: Path | str | None = None,
              require_files: bool = True) -> Task:
    """Build the labelled task from the metadata and a folder of JPEGs.

    A species joins the class set if at least `min_cameras` distinct cameras saw it at
    least `min_per_camera` times. The defaults give 60 species; `(1, 1)` keeps them all.
    The class set is decided before any file check, so it does not depend on how many
    images you have prepared.

    Args:
        min_per_camera: images a camera needs before it counts for a species.
        min_cameras: cameras a species needs to join the class set.
        metadata: metadata file to read. Defaults to the shipped one.
        image_dir: folder holding the JPEGs. Defaults to `images_h<DEFAULT_HEIGHT>/`.
        require_files: drop images missing from `image_dir`. Set False to inspect the
            metadata with no images on disk.

    Returns:
        A `Task`.

    Raises:
        FileNotFoundError: if `require_files` and `image_dir` does not exist.
    """
    img_dir = Path(image_dir) if image_dir else IMG_DIR
    df = load_metadata(metadata)

    full_lab = df[(df["category"] != EMPTY) & ~_is_junk(df["category"])]
    mat = pd.crosstab(full_lab["category"], full_lab["location"])
    keep = mat.index[(mat >= min_per_camera).sum(axis=1) >= min_cameras]

    if require_files:
        if not img_dir.exists():
            m = re.fullmatch(r"images_h(\d+)", img_dir.name)
            build = (f"`uv run python prepare.py --source <archive> --height {m[1]}`"
                     if m else "`uv run python prepare.py --source <archive>`")
            raise FileNotFoundError(
                f"no image folder at {img_dir} -- build it with {build}, or point "
                f"`image_dir:` in your config at a folder of JPEGs you already have")
        have = {p.name for p in img_dir.iterdir()}
        missing = (~df["file_name"].isin(have)).sum()
        if missing:
            print(f"note: {missing:,} of {len(df):,} images are not in {img_dir}, "
                  f"dropping them")
        df = df[df["file_name"].isin(have)].copy()

    empty = df[df["category"] == EMPTY].copy()
    lab = df[df["category"].isin(keep)].copy()

    classes = sorted(lab["category"].unique())
    lab["y"] = lab["category"].map({c: i for i, c in enumerate(classes)}).astype("int64")
    return Task(df=lab.reset_index(drop=True), empty=empty.reset_index(drop=True),
                classes=classes, image_dir=img_dir)


FOLDS = ("official_ood", "random_burst")


def load_fold(task: Task, name: str = "official_ood") -> dict[str, np.ndarray]:
    """Read one of the two shipped evaluation folds.

    The folds are fixed, stored as a column per fold in the metadata.

    `official_ood` holds out 48 whole cameras as test and 32 more as validation: no image
    from a test or validation camera is anywhere in training. 
    `random_burst` is the in-distribution reference: the same cameras can appear in train,
      val, and test. Images from the same burst are never split across train, val, and test.
    
    The drop in performance from 'random_burst' to `official_ood` is the gap the project is about.
    Both split samples per whole bursts, never single images.

    Args:
        task: the task to index into.
        name: `"official_ood"` or `"random_burst"`.

    Returns:
        `{"train": idx, "val": idx, "test": idx}`, each an index array into `task.df`.

    Raises:
        ValueError: if `name` is not one of `FOLDS`.
        KeyError: if the metadata has no column for it.
    """
    if name not in FOLDS:
        raise ValueError(f"unknown fold {name!r}, have {' and '.join(FOLDS)}")
    if name not in task.df.columns:
        raise KeyError(f"metadata has no `{name}` column")
    col = task.df[name].to_numpy()
    return {s: np.flatnonzero(col == s) for s in ("train", "val", "test")}


def load_taxonomy(path: Path | None = None) -> pd.DataFrame:
    """Read the taxonomy: the genus, family, order and class of each species.

    Nothing uses this by default.
    This is extra information you could use to improve your model. 

    Args:
        path: taxonomy file to read. Defaults to the shipped `data/taxonomy.csv`.

    Returns:
        DataFrame indexed by species name, with `genus`, `family`, `order`, `class` and
        the English `common_name`.
    """
    df = pd.read_csv(path or (DATA / "taxonomy.csv"))
    return df.set_index("query")[["genus", "family", "order", "class", "common_name"]]


def official_test_cameras(path: Path | None = None) -> list[int]:
    """The 48 camera ids held out in the official ood test split.

    Args:
        path: JSON file to read. Defaults to `data/official_test_cameras.json`.

    Returns:
        The 48 camera ids.
    """
    data = json.loads((path or (DATA / "official_test_cameras.json")).read_text())
    return list(data["cameras"])


# --- datasets -------------------------------------------------------------------------




class ImageDataset(Dataset):
    """Labelled images, read one JPEG at a time from `task.image_dir`.

    Each item is `(image, label, domain)`. The third element is the camera the photograph
    came from, so an adaptation or adversarial method needs no change here. Plain
    supervised training ignores it.
    """

    def __init__(self, task: Task, idx: np.ndarray, transform):
        """Wrap one split of a fold.

        Args:
            task: the task to read images from.
            idx: which rows of `task.df` this dataset covers, e.g. one split of a fold.
            transform: image pipeline applied to every item, from `build_transform`.
                Whether it augments is the caller's choice -- see `make_loaders`.
        """
        self.names = task.df["file_name"].to_numpy()[idx]
        self.y = task.df["y"].to_numpy()[idx].astype(np.int64)
        self.d = task.domain[idx].astype(np.int64)
        self.image_dir = task.image_dir
        self.transform = transform

    def __len__(self) -> int:
        """Number of images in this split."""
        return len(self.y)

    def __getitem__(self, i):
        """Return `(image, label, domain)` for item `i`."""
        im = Image.open(self.image_dir / self.names[i]).convert("RGB")
        return self.transform(im), int(self.y[i]), int(self.d[i])


class UnlabelledDataset(Dataset):
    """Images with no label, for adaptation methods.

    `__getitem__` returns a bare tensor rather than a tuple, so no species label can be
    read off these frames even by accident.
    """

    def __init__(self, image_dir: Path, names: list[str], transform=None, size: int = 448):
        """Wrap a list of image files.

        Args:
            image_dir: folder holding the JPEGs.
            names: file names to read, relative to `image_dir`.
            transform: image pipeline applied to every item. Defaults to
                `build_transform(size)`, which does not augment.
            size: side length in pixels, used only to build the default transform.
        """
        self.image_dir, self.names = Path(image_dir), list(names)
        self.transform = transform if transform is not None else build_transform(size)

    def __len__(self) -> int:
        """Number of images."""
        return len(self.names)

    def __getitem__(self, i) -> torch.Tensor:
        """Return the image tensor for item `i`. No label, by design."""
        im = Image.open(self.image_dir / self.names[i]).convert("RGB")
        return self.transform(im)


def unlabelled_frames(task: Task, cameras=None, size: int = 448,
                      max_per_camera: int | None = None,
                      seed: int = 0, transform=None) -> UnlabelledDataset:
    """The `empty` frames from the given cameras, as an unlabelled dataset.

    Passing the *test* cameras is allowed and is the intended use: these frames carry no
    species label, so training on them is unsupervised adaptation, not cheating. They
    show a held-out camera's background and lighting at no annotation cost.

        target = unlabelled_frames(task, cameras=official_test_cameras())
        for xb in DataLoader(target, batch_size=32, shuffle=True):
            ...                      # xb is a tensor; there is no yb

    Args:
        task: the task whose `empty` pool to draw from.
        cameras: camera ids to draw from. None takes every camera.
        size: side length in pixels the images are resized to.
        max_per_camera: cap on frames per camera. None takes all of them.
        seed: seed for the per-camera sampling, used only with `max_per_camera`.
        transform: image pipeline. Defaults to `build_transform(size)`, which does not
            augment.

    Returns:
        An `UnlabelledDataset`.
    """
    e = task.empty
    if cameras is not None:
        e = e[e["location"].isin(list(cameras))]
    if max_per_camera:
        e = e.groupby("location", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_per_camera), random_state=seed))
    return UnlabelledDataset(task.image_dir, list(e["file_name"]), transform, size)


def build_transform(size: int, train: bool = False, mean=IMAGENET_MEAN,
                    std=IMAGENET_STD) -> v2.Compose:
    """The image pipeline: a PIL image in, a normalised `(3, size, size)` tensor out.

    Augmentation goes here. Only a random horizontal flip is applied, and only when
    `train` is true. `make_loaders` calls this twice, once per setting, so that val and
    test see no augmentation; add yours to the `aug` list, or build a pipeline of your
    own and hand it to `make_loaders`.

    Args:
        size: side length in pixels of the square output.
        train: include the training augmentation. Leave false for val and test.
        mean, std: normalisation constants. Use the backbone's own (`model.mean`,
            `model.std`); CLIP, for one, was not trained with ImageNet's.

    Returns:
        A `v2.Compose` transform.
    """
    aug = [v2.RandomHorizontalFlip(p=0.5)] if train else []
    return v2.Compose([
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR),
        *aug,
        v2.ToImage(),                              # PIL -> uint8 tensor, (3, H, W)
        v2.ToDtype(torch.float32, scale=True),     # uint8 0..255 -> float 0..1
        v2.Normalize(mean=mean, std=std),
    ])


PREFETCH_FACTOR = 2


def make_loaders(task: Task, fold: dict[str, np.ndarray], batch_size: int = 32,
                 size: int = 448, num_workers: int = 8,
                 train_transform=None, eval_transform=None) -> dict[str, DataLoader]:
    """Build one DataLoader per split of a fold, with the right transform on each.

    The train loader augments and shuffles. Val and test get a fixed pipeline and no
    shuffle, so their scores depend on the model alone. This split is the reason the two
    transforms are chosen here rather than inside the dataset.

    Args:
        task: the task the fold indexes into.
        fold: `{"train": idx, "val": idx, "test": idx}` from `load_fold`.
        batch_size: images per batch.
        size: side length in pixels the images are resized to.
        num_workers: DataLoader worker processes per loader.
        train_transform: pipeline for the train split. Defaults to
            `build_transform(size, train=True)`; pass your own to add augmentation.
        eval_transform: pipeline for val and test. Defaults to `build_transform(size)`.
            Augmenting here changes what the score means, so leave it alone unless that
            is the point (test-time augmentation).

    Returns:
        `{"train": loader, "val": loader, "test": loader}`.
    """
    if train_transform is None:
        train_transform = build_transform(size, train=True)
    if eval_transform is None:
        eval_transform = build_transform(size, train=False)
    loaders = {}
    for split, idx in fold.items():
        is_train = split == "train"
        ds = ImageDataset(task, idx, train_transform if is_train else eval_transform)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=is_train and num_workers > 0,
            prefetch_factor=PREFETCH_FACTOR if num_workers else None)
    return loaders
