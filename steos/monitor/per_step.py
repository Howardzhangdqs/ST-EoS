"""Cheap per-step instrumentation.

Everything here is computed from existing `.grad` tensors and a per-param
delta (Δθ) dict the trainer produces by diffing params before/after the step —
no extra forward/backward passes, so it can run every single step.

Recorded per step:
  * loss
  * global grad norm   ‖∇L‖₂
  * global update norm ‖Δθ‖₂          (the "weight change" magnitude)
  * per-Linear-layer:  weight Frobenius norm ‖W‖_F   (the sharpness proxy of
                        Li et al. NeurIPS'22 — esp. the OUTPUT layer)
                        weight-grad Frobenius norm
  * output-layer weight norm tagged explicitly

Persists to CSV for downstream analysis/plotting.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class PerStepRecorder:
    def __init__(self, output_layer: str | None = None):
        # Name of the output layer as it appears in `model.named_modules()`.
        # None -> resolved lazily in `record()` via the model's
        # `output_layer_name` class attribute, falling back to the legacy
        # MLP name "fc_out" (with a diagnostic marker).
        self.output_layer = output_layer
        self.rows: list[dict] = []

    def _grad_norm_sq(self, model: torch.nn.Module) -> float:
        s = 0.0
        for p in model.parameters():
            if p.grad is not None:
                s += float(p.grad.detach().pow(2).sum().item())
        return s

    @staticmethod
    def _delta_norm_sq(delta: dict[str, torch.Tensor]) -> float:
        s = 0.0
        for d in delta.values():
            s += float(d.detach().pow(2).sum().item())
        return s

    @torch.no_grad()
    def record(
        self,
        step: int,
        model: torch.nn.Module,
        loss: float,
        delta: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        grad_norm = math.sqrt(self._grad_norm_sq(model))
        update_norm = math.sqrt(self._delta_norm_sq(delta)) if delta else float("nan")

        row: dict = {
            "step": step,
            "loss": float(loss),
            "grad_norm": grad_norm,
            "update_norm": update_norm,
        }
        # per-Linear-layer weight & weight-grad Frobenius norms
        for name, m in model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            wn = float(m.weight.detach().pow(2).sum().sqrt().item())
            row[f"wnorm_{name}"] = wn
            if m.weight.grad is not None:
                row[f"gnorm_{name}"] = float(m.weight.grad.detach().pow(2).sum().sqrt().item())
            if m.bias is not None and m.bias.grad is not None:
                row[f"gbiasnorm_{name}"] = float(m.bias.grad.detach().pow(2).sum().sqrt().item())
        # explicit output-layer weight norm (sharpness proxy). Resolution:
        #   1. explicit `output_layer` given to the constructor (Trainer),
        #   2. else the model's `output_layer_name` protocol attribute
        #      (MLP "fc_out", ResNet "fc", ViT "head", timm wrapper resolves
        #      its classifier at build time),
        #   3. else legacy behavior: the hardcoded MLP name "fc_out", plus a
        #      `_wnorm_output_missing` diagnostic flag so downstream analysis
        #      can see the column is not trustworthy rather than silently NaN.
        out_layer = self.output_layer
        if out_layer is None:
            out_layer = getattr(model, "output_layer_name", None)
        if out_layer is not None:
            row["wnorm_output"] = row.get(f"wnorm_{out_layer}", float("nan"))
        else:
            row["wnorm_output"] = row.get("wnorm_fc_out", float("nan"))
            row["_wnorm_output_missing"] = True
        self.rows.append(row)
        return row

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_csv(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_df().to_csv(path, index=False)
