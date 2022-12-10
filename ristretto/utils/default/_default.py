import torch as _torch
from torchvision import (
    datasets as _datasets,
    transforms as _transforms
)

DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
DEFAULT_OPTIMIZER_KWARGS = {
    'lr': 1e-3,
}
DATA_LOADERS = {
    'MNIST': lambda: (
        _torch.utils.data.DataLoader(
            _datasets.MNIST('./data', train=True, download=True,
                            transform=_transforms.ToTensor()),
            batch_size=BATCH_SIZE,
        ),
        _torch.utils.data.DataLoader(
            _datasets.MNIST('./data', train=False, download=True,
                            transform=_transforms.ToTensor()),
            batch_size=BATCH_SIZE,
        )
    ),
    'FashionMNIST': lambda: (
        _torch.utils.data.DataLoader(
            _datasets.FashionMNIST(
                './data', train=True, download=True, transform=_transforms.ToTensor()),
            batch_size=BATCH_SIZE,
        ),
        _torch.utils.data.DataLoader(
            _datasets.FashionMNIST(
                './data', train=False, download=True, transform=_transforms.ToTensor()),
            batch_size=BATCH_SIZE,
        )
    ),
}
