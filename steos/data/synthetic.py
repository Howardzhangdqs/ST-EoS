"""Synthetic dataset factories for clean EoS dynamics.

Two families, both returning (X, y) float32 tensors on CPU + a task descriptor:

  * 'teacher_student' — regression. X ~ N(0, I), y = teacher_mlp(X) + noise,
    MSE loss. A classic EoS testbed: smooth, full-batch-friendly, tunable
    sharpness via teacher width. Default teacher is a wide tanh MLP so the
    student exhibits progressive sharpening.

  * 'gaussian_cls' — 2-class linearly/barely-separable Gaussian classification,
    cross-entropy loss. Good for checking EoS under classification.

Data difficulty is exposed via a `difficulty` knob (controls overlap / noise)
so we can later reproduce the data-difficulty sweep of Yoo et al. (ICML 2025).

A `make_loader` helper turns a dataset into full-batch or mini-batch iterables.
"""

from __future__ import annotations

import numpy as np
import torch

DATASETS = ("teacher_student", "gaussian_cls", "multisharp_teacher_student")


def _teacher_forward(X: np.ndarray, Ws, bs, activation="tanh") -> np.ndarray:
    h = X
    act = np.tanh if activation == "tanh" else (lambda z: np.maximum(z, 0.0))
    for W, b in zip(Ws[:-1], bs[:-1]):
        h = act(h @ W + b)
    out = h @ Ws[-1] + bs[-1]
    if out.shape[-1] == 1:
        out = out.squeeze(-1)   # single output → (n,); multi-output (k>1) kept as (n, k)
    return out


def make_dataset(
    name: str = "teacher_student",
    n: int = 1024,
    dim: int = 20,
    seed: int = 0,
    difficulty: float = 0.5,
    teacher_hidden: tuple[int, ...] = (128, 128),
    out_dim: int = 1,
    out_norm: float = 1.0,
    device: str | None = None,
):
    rng = np.random.default_rng(seed)
    if name == "teacher_student":
        X = rng.standard_normal((n, dim)).astype(np.float32)
        # normalize input columns to unit norm so input Hessian is well-conditioned
        X /= max(np.linalg.norm(X, axis=0).mean(), 1e-6)
        # build a random teacher tanh MLP
        tWs, tbs = [], []
        d = dim
        for h in teacher_hidden:
            W = (rng.standard_normal((d, h)) * np.sqrt(1.0 / d)).astype(np.float32)
            b = np.zeros((h,), dtype=np.float32)
            tWs.append(W); tbs.append(b); d = h
        W = (rng.standard_normal((d, 1)) * np.sqrt(1.0 / d)).astype(np.float32)
        tWs.append(W); tbs.append(np.zeros((1,), dtype=np.float32))
        y_clean = _teacher_forward(X, tWs, tbs, "tanh")
        y_clean = y_clean - y_clean.mean()
        y_clean /= max(y_clean.std(), 1e-6)  # unit-variance targets
        noise_std = float(difficulty)  # difficulty ~ noise level
        y = (y_clean + noise_std * rng.standard_normal(n)).astype(np.float32)
        task = {"type": "regression", "loss": "mse", "in_dim": dim, "out_dim": 1}
    elif name == "multisharp_teacher_student":
        # Multi-output ISOTROPIC teacher-student — the multi-sharp EoS testbed.
        #
        # Teacher is a tanh MLP with ``out_dim=k`` outputs whose output-weight
        # columns all have EQUAL norm (``out_norm``). For MSE, the output-layer
        # parameters are distinct per output, so the Gauss-Newton Hessian block
        # restricted to the output layer is block-diagonal with k blocks, each
        # ~ (2/n)·Σ_i r_ik² · h_i h_iᵀ (h_i = last-hidden activation). With
        # equal-norm output columns the residuals r_ik are comparable across k,
        # hence the k blocks have comparable top eigenvalues ⇒ k comparably-sharp
        # directions at EoS (Li et al. NeurIPS 2022: output layer dominates the
        # top Hessian eigenvalues). This is the multi-sharp landscape on which
        # the "spectral migration" mechanism (damping v_1 ⇒ λ_2 rises toward
        # 2/η) can manifest — unlike the 1-D-sharp standard teacher-student.
        k = max(1, int(out_dim))
        X = rng.standard_normal((n, dim)).astype(np.float32)
        X /= max(np.linalg.norm(X, axis=0).mean(), 1e-6)   # isotropic inputs
        tWs, tbs = [], []
        d = dim
        for h in teacher_hidden:
            W = (rng.standard_normal((d, h)) * np.sqrt(1.0 / d)).astype(np.float32)
            tWs.append(W); tbs.append(np.zeros((h,), dtype=np.float32)); d = h
        # output layer: (hidden, k) with EQUAL-NORM columns (isotropic outputs)
        Wout = rng.standard_normal((d, k)).astype(np.float32)
        Wout = Wout / np.linalg.norm(Wout, axis=0, keepdims=True) * float(out_norm)
        tWs.append(Wout); tbs.append(np.zeros((k,), dtype=np.float32))
        y_clean = _teacher_forward(X, tWs, tbs, "tanh")     # (n, k) — no squeeze (k>1)
        y_clean = y_clean - y_clean.mean(axis=0, keepdims=True)
        y_clean = y_clean / np.maximum(y_clean.std(axis=0, keepdims=True), 1e-6)
        noise_std = float(difficulty)
        y = (y_clean + noise_std * rng.standard_normal((n, k))).astype(np.float32)
        task = {"type": "regression", "loss": "mse", "in_dim": dim, "out_dim": k}
    elif name == "gaussian_cls":
        # two classes; difficulty controls class separation (higher -> closer means)
        sep = 3.0 * (1.0 - float(difficulty)) + 0.5
        mean0 = np.zeros(dim, dtype=np.float32)
        mean1 = (np.zeros(dim, dtype=np.float32))
        mean1[0] = sep
        cov = np.eye(dim, dtype=np.float32)
        n0, n1 = n // 2, n - n // 2
        X0 = rng.multivariate_normal(mean0, cov, n0).astype(np.float32)
        X1 = rng.multivariate_normal(mean1, cov, n1).astype(np.float32)
        X = np.vstack([X0, X1])
        X /= max(np.linalg.norm(X, axis=0).mean(), 1e-6)
        y = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)
        task = {"type": "classification", "loss": "bce", "in_dim": dim, "out_dim": 1}
    else:
        raise ValueError(f"Unknown dataset '{name}'. Known: {DATASETS}")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    if y_t.dim() == 1 and task["out_dim"] == 1:
        y_t = y_t.unsqueeze(-1)
    if device:
        X_t = X_t.to(device); y_t = y_t.to(device)
    return X_t, y_t, task


def loss_fn(task: dict):
    if task["loss"] == "mse":
        return lambda pred, y: ((pred - y) ** 2).mean()
    if task["loss"] == "bce":
        import torch.nn.functional as F
        return lambda pred, y: F.binary_cross_entropy_with_logits(pred, y)
    raise ValueError(task["loss"])
