"""Minimal ViT for CIFAR (32×32) with LayerNorm.

Purpose: test whether the EoS findings — especially Finding 14 ("the
normalization affine γ is the sharpness proxy") — generalize from a BatchNorm
convnet to a LayerNorm Transformer. LayerNorm also has an affine scale γ that,
like BN's γ, multiplies the normalized activations, so mechanistically it should
*also* be the sharpness proxy if the principle is general.

~0.8 M params (dim=128, depth=4, heads=4, patch=4 → 64+1 tokens).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.h = heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))

    def forward(self, x):
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # each [B, H, N, head_dim]
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(a)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    output_layer_name = "head"

    def __init__(self, in_channels: int = 3, num_classes: int = 10, dim: int = 128,
                 depth: int = 4, heads: int = 4, mlp_dim: int = 512, patch: int = 4, img: int = 32):
        super().__init__()
        assert img % patch == 0
        n_tokens = (img // patch) ** 2 + 1
        self.patch_embed = nn.Conv2d(in_channels, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        f = self.patch_embed(x).flatten(2).transpose(1, 2)          # [B, N, dim]
        c = self.cls.expand(f.shape[0], -1, -1)
        x = torch.cat([c, f], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def linear_layer_names(self) -> list[str]:
        return ["head"]
