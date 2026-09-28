"""Small PreAct-ResNet for CIFAR with BatchNorm.

BatchNorm matters here: it is the regime where weight-norm has real actuator
authority (activations scale with ‖W‖/√ fan-in under BN), so this model lets us
test BOTH (a) do the 11 findings generalize to a normalized convnet, and (b)
does Mechanism C gain control authority under BN.

Configurable: widths + blocks-per-stage. Default (16/32/64 × 2) ≈ 0.17 M params
→ fast on a single 16 GB GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + sc


class ResNet(nn.Module):
    output_layer_name = "fc"

    def __init__(self, widths=(16, 32, 64), blocks=(2, 2, 2), num_classes: int = 10,
                 in_channels: int = 3):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
        layers, cin = [], widths[0]
        # stage 1 keeps 32x32; stages 2,3 downsample (stride 2)
        for s, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if s == 0 else 2
            for b in range(n):
                layers.append(PreActBlock(cin, w, stride if b == 0 else 1))
                cin = w
        self.body = nn.Sequential(*layers)
        self.bn_end = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        self._cin = cin

    def forward(self, x):
        x = self.conv0(x)
        x = self.body(x)
        x = F.relu(self.bn_end(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

    def linear_layer_names(self) -> list[str]:
        return ["fc"]
