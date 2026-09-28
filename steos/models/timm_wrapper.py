"""Thin wrapper around `timm.create_model` so configs/scripts can pick credible,
well-tested architectures (ViT, ConvNeXt, …) by name. Each factory fixes the
timm model name and whether to reconfigure geometry (img_size/patch_size) for
32×32 CIFAR inputs.
"""

from __future__ import annotations

import torch.nn as nn


class TimmModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 10, img_size: int = 32,
                 patch_size: int = 4, in_chans: int = 3, pass_geom: bool = True,
                 drop_rate: float = 0.0):
        super().__init__()
        import timm
        kw = dict(pretrained=False, num_classes=num_classes, in_chans=in_chans, drop_rate=drop_rate)
        if pass_geom:
            kw["img_size"] = img_size
            kw["patch_size"] = patch_size
        self.net = timm.create_model(model_name, **kw)
        self.model_name = model_name
        self.output_layer_name = self._resolve_output_layer_name()

    def _resolve_output_layer_name(self) -> str | None:
        """Name of the classifier submodule, as seen by `named_modules()` on
        THIS wrapper (so typically prefixed with `net.`), or None if unknown.

        Resolution order:
          1. timm protocol: `self.net.get_classifier()` — then locate that
             module in `named_modules()` by identity. Identity matching also
             handles nested heads (e.g. ConvNeXt's `head.fc`).
          2. Fallback for models without `get_classifier()` (torchvision-style
             nets): probe the conventional attributes `head` / `fc` /
             `classifier`.
          3. Give up -> None; the per-step recorder then keeps the legacy
             `fc_out` lookup and tags `_wnorm_output_missing`.

        Note: the recorder only scans `nn.Linear` submodules, so if the
        resolved classifier is not a Linear the `wnorm_output` column will
        still be NaN — acceptable, since timm classification heads are Linear.
        """
        clf = None
        get_clf = getattr(self.net, "get_classifier", None)
        if callable(get_clf):
            clf = get_clf()
        if clf is None:
            for attr in ("head", "fc", "classifier"):
                if hasattr(self.net, attr):
                    clf = getattr(self.net, attr)
                    break
        if clf is None:
            return None
        for name, m in self.named_modules():
            if m is clf:
                return name
        return None


    def forward(self, x):
        return self.net(x)


def timm_vit_tiny(**kw):
    return TimmModel("vit_tiny_patch16_224", pass_geom=True, **kw)


def timm_convnext_atto(**kw):
    return TimmModel("convnext_atto", pass_geom=False, **kw)


def timm_convnext_femto(**kw):
    return TimmModel("convnext_femto", pass_geom=False, **kw)
