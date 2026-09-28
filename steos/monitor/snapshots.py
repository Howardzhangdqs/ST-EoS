"""Periodic full-parameter snapshots for trajectory analysis (Gap A).

Every `every` steps we flatten ALL trainable parameters into one 1-D vector
and store it. After training these vectors are stacked into a [S, P] matrix
that PCA/UMAP-project to reveal the low-dimensional geometry of the weight
trajectory — which we then color by sharpness.

Also stored: the ordered list of (name, shape, numel) so any snapshot can be
re-expanded back into the model (useful for sanity checks and later for
estimating the top Hessian eigenvector at a chosen snapshot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def flatten_params(model: torch.nn.Module) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int]]]:
    flat = []
    spec: list[tuple[str, tuple[int, ...], int]] = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            t = p.detach().reshape(-1)
            flat.append(t.cpu().numpy().astype(np.float32, copy=False))
            spec.append((name, tuple(p.shape), p.numel()))
    return np.concatenate(flat), spec


class SnapshotManager:
    def __init__(self, every: int = 10, ref_center: bool = True):
        self.every = int(every)
        self.ref_center = ref_center
        self.steps: list[int] = []
        self.vecs: list[np.ndarray] = []
        self.spec: list[tuple[str, tuple[int, ...], int]] | None = None
        self._ref: np.ndarray | None = None

    @torch.no_grad()
    def maybe_add(self, step: int, model: torch.nn.Module):
        if step % self.every != 0:
            return
        vec, spec = flatten_params(model)
        if self.spec is None:
            self.spec = spec
            if self.ref_center:
                self._ref = vec.copy()
        if self._ref is not None:
            vec = vec - self._ref
        self.steps.append(int(step))
        self.vecs.append(vec)

    def stack(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(self.vecs, axis=0), np.asarray(self.steps, dtype=np.int64)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        W, steps = self.stack()
        names, shapes, numels = [], [], []
        if self.spec:
            names, shapes, numels = zip(*[(n, s, k) for (n, s, k) in self.spec])
        np.savez_compressed(
            path,
            W=W.astype(np.float32),
            steps=steps,
            param_names=np.array(names, dtype=object),
            param_shapes=np.array(shapes, dtype=object),
            param_numels=np.array(numels, dtype=np.int64),
            ref=self._ref if self._ref is not None else np.zeros(0, dtype=np.float32),
        )

    def __len__(self):
        return len(self.vecs)
