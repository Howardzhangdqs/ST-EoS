"""Architecture-agnostic training loop with full monitoring instrumentation.

This is the harness all five research gaps share. Per step it records cheap
metrics; periodically it snapshots full weights. A `sharpness_hook` slot is
reserved for Phase 2 (lambda_max / lambda_c measurement) so the loop will not
need to change when we add curvature estimation.

Design notes
------------
* Full-batch GD is the default — the cleanest regime for the Edge-of-Stability
  phenomenon (Cohen 2021, Li 2022, Damian 2023 all use full-batch). Mini-batch
  SGD is supported via `train.batch_size`.
* Δθ (the per-step weight change) is obtained by diffing params before/after
  the optimizer step — optimizer-agnostic and exact, works for any optimizer.
* Outputs land in `runs/<run_name>/`: metrics.csv, snapshots.npz, manifest.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import loss_fn, make_dataset
from .models import build_model
from .monitor import PerStepRecorder, SnapshotManager
from .optim import SGD, register_param_names
from .sharpness import make_sharpness_hook


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: dict, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(cfg["seed"])

        mcfg = dict(cfg["model"])
        self.model = build_model(mcfg.pop("name"), **mcfg).to(self.device)
        self.params = dict(self.model.named_parameters())  # views; stay valid after in-place step

        dcfg = dict(cfg["data"])
        self.X, self.y, self.task = make_dataset(dcfg.pop("name"), device=self.device, **dcfg)
        self.loss = loss_fn(self.task)
        self.n = self.X.shape[0]

        ocfg = cfg["optimizer"]
        self.opt = SGD(
            self.model.parameters(),
            lr=ocfg["lr"],
            momentum=ocfg.get("momentum", 0.0),
            weight_decay=ocfg.get("weight_decay", 0.0),
            nesterov=ocfg.get("nesterov", False),
        )
        register_param_names(self.opt, self.model)

        mon = cfg.get("monitor", {})
        # resolve the output layer per architecture (MLP "fc_out", ResNet "fc",
        # ViT "head", timm wrapper auto-detects) so `wnorm_output` is never
        # silently NaN on non-MLP models.
        self.recorder = PerStepRecorder(
            output_layer=getattr(self.model, "output_layer_name", None)
        )
        self.snapshots = SnapshotManager(every=mon.get("snapshot_every", 10))
        self.log_every = int(mon.get("log_every", 100))

        tr = cfg["train"]
        self.steps = int(tr["steps"])
        bs = tr.get("batch_size", "full")
        self.batch_size = self.n if (bs in (None, "full") or bs >= self.n) else int(bs)

        self.sharpness_hook = None  # default; overridden below if enabled
        self._sharp_records: list[dict] = []
        self._vtop_records: list[dict] = []
        self._dtheta_records: list[dict] = []  # single-step Δθ at sharpness steps

        shcfg = cfg.get("sharpness", {})
        if shcfg.get("enabled", False):
            self.sharpness_hook = make_sharpness_hook(
                self.opt,
                self.loss,
                kinds=tuple(shcfg.get("kinds", ["lambda_max", "lambda_c"])),
                n_iter=int(shcfg.get("n_iter", 200)),
                tol=float(shcfg.get("tol", 1e-4)),
                lr_guess=float(shcfg.get("lr_guess", 1e-3)),
                tol_power=int(shcfg.get("tol_power", 4)),
                warm_start_vec=bool(shcfg.get("warm_start_vec", True)),
            )

    def _batch(self, rng: np.random.Generator):
        if self.batch_size >= self.n:
            return self.X, self.y
        idx = rng.integers(0, self.n, self.batch_size)
        return self.X[idx], self.y[idx]

    @torch.no_grad()
    def _param_clone(self) -> dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.params.items()}

    def train(self) -> dict:
        rng = np.random.default_rng(self.cfg["seed"] + 1)
        sharp_every = int(self.cfg.get("sharpness", {}).get("every", 0))
        t0 = time.time()
        self.model.train()

        for step in range(self.steps):
            Xb, yb = self._batch(rng)
            self.opt.zero_grad(set_to_none=True)
            loss = self.loss(self.model(Xb), yb)
            loss.backward()

            # measure curvature at θ_t (BEFORE the step): Δθ is the update about
            # to be applied — this matches the λ_c / EoS definitions.
            is_sharp_step = (
                self.sharpness_hook is not None and sharp_every > 0 and step % sharp_every == 0
            )
            if is_sharp_step:
                rec = self.sharpness_hook(step, self.model, self.X, self.y, float(loss.item()))
                rec["step"] = step
                vflat = rec.pop("vtop_flat", None)  # pulled out → vtops.npz, not CSV
                self._sharp_records.append(rec)
                if vflat is not None:
                    self._vtop_records.append({"step": step, "v": vflat})

            before = self._param_clone()
            self.opt.step()
            with torch.no_grad():
                delta = {n: self.params[n] - before[n] for n in before}

            # persist the SINGLE-step Δθ at sharpness steps for the Damian
            # alignment test (Δθ_t vs v_top_t). Multi-step snapshot diffs are
            # NOT the right quantity — at EoS they cancel the period-2 oscillation.
            if is_sharp_step:
                dflat = np.concatenate(
                    [delta[n].detach().cpu().numpy().ravel() for n in self.params]
                ).astype(np.float32)
                self._dtheta_records.append({"step": step, "d": dflat})

            row = self.recorder.record(step, self.model, float(loss.item()), delta)
            self.snapshots.maybe_add(step, self.model)

            if step == 0 or (step + 1) % self.log_every == 0 or step == self.steps - 1:
                print(
                    f"step {step:5d} | loss {row['loss']:.5f} | "
                    f"grad {row['grad_norm']:.4f} | dθ {row['update_norm']:.4f} | "
                    f"‖W_out‖ {row['wnorm_output']:.4f}",
                    flush=True,
                )

        elapsed = time.time() - t0
        summary = self.save(elapsed)
        print(f"\n[done] {self.steps} steps in {elapsed:.1f}s -> {self.run_dir}")
        return summary

    def save(self, elapsed: float) -> dict:
        self.recorder.save_csv(self.run_dir / "metrics.csv")
        self.snapshots.save(self.run_dir / "snapshots.npz")
        if self._sharp_records:
            import pandas as pd
            pd.DataFrame(self._sharp_records).to_csv(self.run_dir / "sharpness.csv", index=False)
        if self._vtop_records:
            steps = np.array([r["step"] for r in self._vtop_records], dtype=np.int64)
            V = np.stack([r["v"] for r in self._vtop_records]).astype(np.float32)
            np.savez_compressed(self.run_dir / "vtops.npz", steps=steps, V=V)
        if self._dtheta_records:
            steps = np.array([r["step"] for r in self._dtheta_records], dtype=np.int64)
            D = np.stack([r["d"] for r in self._dtheta_records]).astype(np.float32)
            np.savez_compressed(self.run_dir / "dthetas.npz", steps=steps, D=D)
        n_params = int(sum(p.numel() for p in self.model.parameters()))
        manifest = {
            "config": self.cfg,
            "n_params": n_params,
            "n_train": int(self.n),
            "device": str(self.device),
            "elapsed_s": elapsed,
            "final_loss": float(self.recorder.rows[-1]["loss"]),
            "n_snapshots": len(self.snapshots),
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return manifest


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_from_config(config_path: str | Path, run_root: str | Path = "runs"):
    cfg = load_config(config_path)
    run_dir = Path(run_root) / cfg["run_name"]
    return Trainer(cfg, run_dir).train()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()
    run_from_config(args.config, args.run_root)
