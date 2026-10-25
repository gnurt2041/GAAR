"""UPAR annotation files and the training dataset.

Layout of the challenge data (`data_root`):
    annotations/task2/{train,val}/gt.csv       "# image,<40 attribute names>" then "<path>,<40 x 0/1>"
    annotations/task2/val/{ids.csv,queries.csv} semantic id of each image, one query vector per line
    Market1501/..., PA100k/..., PETA/...       images; paths in gt.csv start with the dataset name
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

DATASETS = ["Market1501", "PA100k", "PETA"]


def dataset_of(path: str) -> str:
    return path.split("/", 1)[0]


def read_gt(csv_path: str) -> tuple[list[str], list[str], np.ndarray]:
    """Returns (attribute names, image paths, labels (N, 40) int8)."""
    names, paths, labels = None, [], []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("#"):
                if names is None and "image" in row[0]:
                    names = [c.strip() for c in row[1:]]
                continue
            paths.append(row[0])
            labels.append([int(float(v)) for v in row[1:41]])
    return names, paths, np.asarray(labels, dtype=np.int8)


def read_retrieval_split(split_dir: str, dataset: str):
    """Gallery of one dataset in a retrieval split, with the queries that have at least one match.
    Returns (names, gallery paths, gallery labels, gallery ids, queries (Q, 40), query ids)."""
    names, paths, labels = read_gt(os.path.join(split_dir, "gt.csv"))
    lab = dict(zip(paths, labels))
    with open(os.path.join(split_dir, "ids.csv"), newline="") as f:
        ids = {r[0]: int(r[1]) for r in csv.reader(f) if r and not r[0].startswith("#")}
    with open(os.path.join(split_dir, "queries.csv"), newline="") as f:
        queries = np.asarray([[int(float(v)) for v in r] for r in csv.reader(f) if r and not r[0].startswith("#")],
                             dtype=np.int8)
    gallery = sorted(p for p in ids if dataset_of(p) == dataset)
    g_ids = np.asarray([ids[p] for p in gallery])
    q_ids = np.asarray(sorted(set(g_ids.tolist())))
    return names, gallery, np.stack([lab[p] for p in gallery]), g_ids, queries[q_ids], q_ids


@dataclass
class Fold:
    name: str
    train_datasets: list[str]
    val_dataset: str | None


def get_fold(name: str) -> Fold:
    """`loo_<Dataset>`: train on the two other datasets, validate on <Dataset>. `final`: train on all three."""
    if name == "final":
        return Fold("final", list(DATASETS), None)
    held_out = name.removeprefix("loo_")
    if held_out not in DATASETS:
        raise ValueError(f"unknown fold {name!r}; use final or loo_{{{','.join(DATASETS)}}}")
    return Fold(name, [d for d in DATASETS if d != held_out], held_out)


class UPARDataset(Dataset):
    def __init__(self, csv_paths: list[str], data_root: str, datasets: list[str] | None = None, transform=None):
        paths, labels = [], []
        for p in csv_paths:
            _, ps, ls = read_gt(p)
            paths += ps
            labels.append(ls)
        labels = np.concatenate(labels)
        keep = [i for i, p in enumerate(paths) if datasets is None or dataset_of(p) in datasets]
        self.paths = [paths[i] for i in keep]
        self.labels = labels[keep].astype(np.float32)
        self.dataset_names = [dataset_of(p) for p in self.paths]
        self.data_root, self.transform = data_root, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.data_root, self.paths[i])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "target": self.labels[i], "dataset": self.dataset_names[i]}


def balanced_sampler(ds: UPARDataset) -> WeightedRandomSampler:
    """Each source dataset contributes the same expected number of images per batch."""
    counts = {n: ds.dataset_names.count(n) for n in set(ds.dataset_names)}
    weights = [1.0 / counts[n] for n in ds.dataset_names]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
