import torch as _torch
import torch.nn as _nn
import ristretto.activations as _ra
import ristretto.utils as _ru
import numpy as _np
from itertools import chain as _chain
from functools import partial as _partial

# Define model to test the functions


class FullyConnected(_nn.Module):
    def __init__(self, activation=_partial(_ra.ReLU, 0), reg=_nn.Identity, hidden_dims=[128], seed=None):
        super(FullyConnected, self).__init__()

        if seed is not None:
            _ru.set_random_seed(seed)

        dims = [784] + hidden_dims + [10]

        linear = [_nn.Linear(dims[i], dims[i + 1])
                  for i in range(len(dims) - 1)]
        activations = [activation() for _ in range(len(dims)-2)]
        regularizers = [reg() for _ in range(len(dims) - 2)]

        if reg is _nn.Identity:
            self.sequence = _nn.Sequential(
                *_chain.from_iterable(zip(linear,
                                          activations)), linear[-1]
            )
        else:
            self.sequence = _nn.Sequential(
                *_chain.from_iterable(zip(linear,
                                          activations, regularizers)), linear[-1]
            )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.sequence(x)
        return x
