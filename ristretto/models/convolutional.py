import torch as _torch
import torch.nn as _nn
import ristretto.activations as _ra
import numpy as _np
from itertools import chain as _chain
from functools import partial as _partial
import random as _random

class ResidualBlock(_nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation=_partial(_ra.ReLU, 0)):
        super().__init__()

        self.C1 = _nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
        )

        self.N1 = _nn.BatchNorm2d(num_features=out_channels)
        self.R1 = activation()
        self.C2 = _nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.N2 = _nn.BatchNorm2d(num_features=out_channels)
        self.R2 = activation()

        self.has_skip_conv = stride != 0 or in_channels != out_channels
        if self.has_skip_conv:
            self.C3 = _nn.Conv2d(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
            )
            self.N3 = _nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        skip_x = x

        x = self.C1(x)
        x = self.N1(x)
        x = self.R1(x)
        x = self.C2(x)
        x = self.N2(x)

        if self.has_skip_conv:
            skip_x = self.C3(skip_x)
            skip_x = self.N3(skip_x)

        x = self.R2(x + skip_x)
        return x

def set_random_seed(seed):
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True

class ResNet(_nn.Module):
    def __init__(self, depth = 20, base_width = 16, activation = _partial(_ra.ReLU, 0), seed = None):
        super().__init__()

        if seed is not None:
            set_random_seed(seed)

        modules = [
            _nn.Conv2d(1, base_width, 3, padding=1),
            _nn.BatchNorm2d(base_width),
            activation(),
        ]

        # Blocks and stages (based off the configuration used in the ResNet paper)
        blocks_per_stage = (depth - 2) // 6
        assert depth == blocks_per_stage * 6 + 2
        in_channels = base_width
        out_channels = base_width
        for stage_idx in range(3):
            for block_idx in range(blocks_per_stage):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                modules.append(
                    ResidualBlock(
                        in_channels,
                        out_channels,
                        stride,
                    )
                )
                in_channels = out_channels
            out_channels = out_channels * 2

        # Output layers
        modules.extend(
            [
                _nn.AdaptiveAvgPool2d(1),
                _nn.Flatten(),
                _nn.Linear(in_channels, 10),
            ]
        )

        self.sequence = _nn.Sequential(*modules)

    def forward(self,x):
        x = self.sequence(x)
        return x