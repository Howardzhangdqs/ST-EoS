"""CIFAR-10 / CIFAR-100 loader (no torchvision needed). Pulls from the HF hub
via `datasets` on first call, caches a normalized NCHW float32 .npz locally.
Requires the 7890 proxy for the one-time download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_CIFAR_MEAN = {
    10: np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1),
    100: np.array([0.5071, 0.4865, 0.4409], dtype=np.float32).reshape(1, 3, 1, 1),
}
_CIFAR_STD = {
    10: np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1),
    100: np.array([0.2673, 0.2564, 0.2762], dtype=np.float32).reshape(1, 3, 1, 1),
}
_HF_NAME = {10: "uoft-cs/cifar10", 100: "uoft-cs/cifar100"}
_LABEL_KEY = {10: "label", 100: "fine_label"}


def build_cache(n_classes: int = 10, cache_path: Path | None = None):
    from datasets import load_dataset
    cache_path = cache_path or Path(f"data/cifar{n_classes}.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HF_NAME[n_classes])

    def to_nchw(split):
        X = np.stack([np.array(im) for im in ds[split]["img"]], axis=0).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        y = np.asarray(ds[split][_LABEL_KEY[n_classes]], dtype=np.int64)
        return X, y

    Xtr, ytr = to_nchw("train")
    Xte, yte = to_nchw("test")
    Xtr = (Xtr - _CIFAR_MEAN[n_classes]) / _CIFAR_STD[n_classes]
    Xte = (Xte - _CIFAR_MEAN[n_classes]) / _CIFAR_STD[n_classes]
    np.savez_compressed(cache_path, Xtr=Xtr.astype(np.float32), ytr=ytr, Xte=Xte.astype(np.float32), yte=yte)
    print(f"[cifar{n_classes}] cached {len(Xtr)} train + {len(Xte)} test -> {cache_path}")
    return cache_path


def load_cifar(n_classes: int = 10, n_train: int | None = None, device=None, cache: Path | None = None):
    cache = cache or Path(f"data/cifar{n_classes}.npz")
    if not cache.exists():
        build_cache(n_classes, cache)
    z = np.load(cache)
    Xtr, ytr = z["Xtr"], z["ytr"]
    if n_train is not None:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
    Xv, yv = torch.from_numpy(z["Xte"]), torch.from_numpy(z["yte"]).long()
    if device:
        Xt, yt, Xv, yv = Xt.to(device), yt.to(device), Xv.to(device), yv.to(device)
    task = {"type": "classification", "loss": "ce", "in_dim": (3, 32, 32),
            "out_dim": n_classes, "n_classes": n_classes}
    return Xt, yt, task, Xv, yv


# backwards-compatible names
def load_cifar10(*a, **kw):
    return load_cifar(10, *a, **kw)
