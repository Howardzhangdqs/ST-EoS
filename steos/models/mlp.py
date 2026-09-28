"""Configurable MLP.

Layer modules are attached as TOP-LEVEL named attributes (`fc1`, `fc2`, ...,
`fc_out`) rather than nested in a container, so `model.named_modules()` yields
clean names. This matters because the monitor attributes per-layer weight/grad
norms by module name, and the *output* layer (`fc_out`) — the sharpness proxy
of Li et al. (NeurIPS 2022) — must be addressable as a stable, predictable key.

Built as: fc1 -> [act] -> fc2 -> [act] -> ... -> fc_out
Optional LayerNorm per hidden block (relevant later for the
weight-norm -> sharpness link under BatchNorm/LayerNorm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACT = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


class MLP(nn.Module):
    # canonical name of the output layer, as seen by `named_modules()` — the
    # monitor's `wnorm_output` column resolves through this.
    output_layer_name = "fc_out"

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int = 1,
        activation: str = "tanh",
        layer_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        act_cls = _ACT[activation]
        dims = [in_dim, *hidden_dims]
        self._linears: list[str] = []
        self._order: list[str] = []
        for i in range(len(dims) - 1):
            name = f"fc{i + 1}"
            lin = nn.Linear(dims[i], dims[i + 1], bias=bias)
            setattr(self, name, lin)
            self._linears.append(name)
            self._order.append(name)
            if layer_norm:
                ln_name = f"ln{i + 1}"
                setattr(self, ln_name, nn.LayerNorm(dims[i + 1]))
                self._order.append(ln_name)
            if i < len(dims) - 2:  # no activation after the last hidden block
                a_name = f"act{i + 1}"
                setattr(self, a_name, act_cls())
                self._order.append(a_name)
        # output layer (no activation, no norm) — the sharpness-proxy layer
        setattr(self, "fc_out", nn.Linear(dims[-1], out_dim, bias=bias))
        self._linears.append("fc_out")
        self._order.append("fc_out")

    def forward(self, x):
        for name in self._order:
            x = getattr(self, name)(x)
        return x

    @property
    def output_layer(self) -> nn.Linear:
        return getattr(self, "fc_out")  # type: ignore[return-value]

    def linear_layer_names(self) -> list[str]:
        """nn.Linear module names in forward order (e.g. ['fc1','fc2','fc_out'])."""
        return list(self._linears)

    def last_hidden_weight(self) -> torch.Tensor:
        """Weight tensor of the last HIDDEN Linear (the layer before fc_out) —
        the best free sharpness proxy per Finding 9."""
        names = self._linears[:-1]  # drop the output layer
        if not names:
            raise ValueError("MLP has no hidden layer")
        return getattr(self, names[-1]).weight
