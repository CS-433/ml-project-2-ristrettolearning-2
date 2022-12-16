import numpy as _np
import torch as _torch
import torch.nn as _nn
import ristretto.activations as _ra
import ristretto.utils as _ru
from functools import partial as _partial


class LeNetModel(_nn.Module):
    def __init__(self, activation=_partial(_ra.ReLU, 0), seed=None):
        """From: LeCun et al., 1998. Gradient-Based Learning Applied to Document Recognition"""
        super().__init__()

        if seed is not None:
            _ru.set_random_seed(seed)

        self.conv1 = _nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = _nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = _nn.Dropout2d()
        self.fc1 = _nn.Linear(320, 50)
        self.fc2 = _nn.Linear(50, 10)
        self.relu1 = activation()
        self.relu2 = activation()
        self.relu3 = activation()

    def forward(self, x):
        max_pool2d = _nn.functional.max_pool2d

        x = self.relu1(max_pool2d(self.conv1(x), 2))
        x = self.relu2(max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.relu3(self.fc1(x))
        x = _nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
