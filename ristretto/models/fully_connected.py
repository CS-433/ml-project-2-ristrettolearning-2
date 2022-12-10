import torch as _torch
import torch.nn as _nn
import ristretto.activations as _ra
import numpy as _np
from itertools import chain as _chain


# Define model to test the functions

class FullyConnected(_nn.Module):
    def __init__(self, activation=lambda: _ra.ReLU(0), hidden_dims=[128], seed=None):
        super(FullyConnected, self).__init__()

        if seed is not None:
            _np.random.seed(seed)
            _torch.manual_seed(seed)

        dims = [784] + hidden_dims + [10]

        self.linear = [_nn.Linear(dims[i], dims[i + 1])
                       for i in range(len(dims) - 1)]
        activations = [activation() for _ in range(len(dims)-2)]

        self.sequence = _nn.Sequential(
            *_chain.from_iterable(zip(self.linear,
                                  activations)), self.linear[-1]
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.sequence(x)
        return x
