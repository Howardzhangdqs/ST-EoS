"""Tiny architecture registry so configs can pick models by name.

Model protocol (optional but recommended): registered classes may expose an
`output_layer_name` class attribute — the name of the classifier submodule as
it appears in `named_modules()` (MLP "fc_out", ResNet "fc", ViT "head"; the
timm wrapper auto-detects its classifier). The per-step monitor uses it to
fill the `wnorm_output` column, so without it that column silently falls back
to the legacy MLP name and reads NaN on other architectures.
"""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def get_model_info():
    return dict(_REGISTRY)


# register built-ins
from . import mlp as _mlp  # noqa: F401,E402
from .mlp import MLP  # noqa: E402
from .resnet import ResNet  # noqa: E402
from .vit import ViT  # noqa: E402
from .timm_wrapper import timm_convnext_atto, timm_convnext_femto, timm_vit_tiny  # noqa: E402

register_model("mlp")(MLP)
register_model("resnet")(ResNet)
register_model("vit")(ViT)
register_model("timm_vit_tiny")(timm_vit_tiny)
register_model("timm_convnext_atto")(timm_convnext_atto)
register_model("timm_convnext_femto")(timm_convnext_femto)
