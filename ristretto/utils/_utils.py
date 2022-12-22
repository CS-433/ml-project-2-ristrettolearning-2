import ristretto.utils.default as _default
import ristretto.models as _rm
import ristretto.activations as _ra
import torch as _torch
import torch.nn.functional as _F
import torchvision.transforms as _transforms
import torch.optim as _optim
from torch.utils.data import (
    random_split as _random_split,
    DataLoader as _DataLoader
)
import pandas as _pd
import numpy as _np
import math as _math
from itertools import chain as _chain
from functools import partial as _partial
import random as _random
from ray import tune as _tune
import os as _os


def set_random_seed(seed):
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.deterministic = True


@_torch.no_grad()
def get_weight_sum(model):
    filtered = filter(lambda x: x[0].endswith(
        "weight"), model.named_parameters())
    mapped = map(lambda x: x[1].sum(), filtered)
    return sum(mapped).item()


def train_loop(
    model,
    optimizer,
    criterion,
    train_loader,
    device=_default.DEVICE,
    metrics_fn=None,
    dtype=None,
    verbose=False
):

    # getting the size of the batch just to measure the progress
    size = len(train_loader.dataset)

    model.train()
    metrics = []

    for batch, (X, y) in enumerate(train_loader):
        # train step
        X, y = X.to(device), y.to(device)
        if dtype is not None:
            X = X.to(dtype=dtype)

        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute metrics
        metrics.append({
            'loss': loss.item(),
            'accuracy': (pred.argmax(1) == y).sum().item() / len(y),
        })
        if metrics_fn is not None:
            metrics[-1] = dict(**metrics[-1], **metrics_fn(model, pred, y))

        # Log training progress
        if batch % 100 == 0 and verbose:
            loss, current = loss.item(), batch * len(X)
            print(f"""Training [{current:>5d}/{size:>5d}]
    batch loss:     {metrics[-1]['loss']:.3e}
    batch accuracy: {metrics[-1]['accuracy'] * 100:.2f}""")

    return metrics


@_torch.no_grad()
def val_loop(
    model,
    criterion,
    val_loader,
    device=_default.DEVICE,
    metrics_fn=None,
    dtype=None
):

    model.eval()
    metrics = []

    for X, y in val_loader:
        # val step
        X, y = X.to(device), y.to(device)
        if dtype is not None:
            X = X.to(dtype=dtype)
        pred = model(X)
        loss = criterion(pred, y)

        # compute metrics
        metrics.append({
            'loss': loss.item(),
            'correct': (pred.argmax(1) == y).sum().item(),
            ### maybe also return accuracy for plots 
            'accuracy': (pred.argmax(1) == y).sum().item()/ len(val_loader.dataset)
        })
        if metrics_fn is not None:
            metrics[-1] = dict(**metrics[-1], **metrics_fn(model, pred, y))

    # compute the average loss and accuracy
    metrics_df = _pd.DataFrame(metrics)

    print(f"""Validation
    loss:     {metrics_df['loss'].mean():.3e}
    accuracy: {metrics_df['correct'].sum() / len(val_loader.dataset) * 100:.2f}""")
    return metrics


def train_model(
    model,
    data_loader,
    data_loader_transform=None,
    epochs=10,
    optimizer_fn=_optim.Adam,
    optimizer_kwargs={},
    criterion=_F.cross_entropy,
    device=_default.DEVICE,
    metrics_fn=None,
    verbose=False
):
    if data_loader_transform is not None:
        train_loader, val_loader = data_loader(data_loader_transform)
    else:
        train_loader, val_loader = data_loader()

    optimizer = optimizer_fn(model.parameters(), **optimizer_kwargs)

    train_metrics = []
    val_metrics = []
    epoch_accuracy = []
    for epoch in range(epochs):
        print(
            f"---------- Epoch {epoch+1:{_math.ceil(_math.log10(epochs+1))}d} ----------")
        train_metrics.append(train_loop(
            model, optimizer, criterion, train_loader, device, metrics_fn, verbose=verbose))
        val_metrics.append(val_loop(
            model, criterion, val_loader, device, metrics_fn))
        epoch_accuracy

    return {
        "train": _pd.DataFrame(_chain.from_iterable(train_metrics)),
        "validation": _pd.DataFrame(_chain.from_iterable(val_metrics))
    }  


def train_multiple_models(
    models,
    data_loader,
    data_loader_transform=None,
    epochs=10,
    optimizer_fn=_optim.Adam,
    optimizer_kwargs={},
    criterion=_F.cross_entropy,
    device=_default.DEVICE,
    metrics_fn=None,
    seed=None,
    verbose=False
):
    """
    returns a list of metrics for each model
    each metric is a dictionary with two keys: "train" and "validation" 
    """
    
    metrics = []

    # call data loader to download data before starting the actual training
    data_loader()

    for i, model in enumerate(models):
        model = model.to(device)

        if seed is not None:
            set_random_seed(seed)

        print(
            f"===== Model {i+1:{_math.ceil(_math.log10(len(models)+1))}d} ({model.__class__.__name__}) =====")
        metrics.append(
            train_model(
                model,
                data_loader,
                data_loader_transform,
                epochs,
                optimizer_fn,
                optimizer_kwargs,
                criterion,
                device,
                metrics_fn,
                verbose
            )
        )
        print("")
        del model

    return metrics


def tune_train_mobilenet(
    config,
    checkpoint_dir=None,
    seed=None,
    epochs=10,
    verbose=False
):
    _torch.set_default_tensor_type(_torch.FloatTensor)
    _torch.set_default_dtype(_torch.bfloat16)

    model = _rm.mobilenet_v3_small(activation=_partial(
        _ra.ReLU6, config["alpha"], config["beta"]), seed=seed).to(_default.DEVICE)
    # model = model.bfloat16()

    optimizer = _optim.Adam(model.parameters(), lr=config["lr"])
    criterion = _F.cross_entropy

    if checkpoint_dir:
        model_state, optimizer_state = _torch.load(
            _os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_set, test_set = _default.DATASETS["MNIST"](_transforms.Compose([
        _transforms.Grayscale(num_output_channels=3),
        _transforms.ToTensor(),
    ]))

    train_size = int(len(train_set) * 0.8)
    train, val = _random_split(
        train_set, [train_size, len(train_set) - train_size])

    train_loader = _DataLoader(
        train, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    val_loader = _DataLoader(
        val, batch_size=config["batch_size"], shuffle=False, num_workers=8)

    train_metrics = []
    val_metrics = []
    for epoch in range(epochs):
        print(
            f"---------- Epoch {epoch+1:{_math.ceil(_math.log10(epochs+1))}d} ----------")
        train_metrics.append(train_loop(
            model, optimizer, criterion, train_loader, verbose=verbose, dtype=_torch.bfloat16))
        val_metrics.append(val_loop(
            model, criterion, val_loader, dtype=_torch.bfloat16))

        with _tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = _os.path.join(checkpoint_dir, "checkpoint")
            _torch.save((model.state_dict(), optimizer.state_dict()), path)

        metrics_df = _pd.DataFrame(val_metrics[-1])

        _tune.report(loss=metrics_df["loss"].mean(), accuracy=(
            metrics_df["correct"].sum() / len(val_loader.dataset) * 100))

    return {
        "train": _pd.DataFrame(_chain.from_iterable(train_metrics)),
        "validation": _pd.DataFrame(_chain.from_iterable(val_metrics))
    }


def tune_train_fullyconnected(
    config,
    checkpoint_dir=None,
    seed=None,
    epochs=10,
    verbose=False
):
    _torch.set_default_tensor_type(_torch.FloatTensor)
    _torch.set_default_dtype(_torch.bfloat16)

    model = _rm.FullyConnected(activation=_partial(
        _ra.ReLU6, config["alpha"], config["beta"]), hidden_dims=[2000], seed=seed).to(_default.DEVICE)
    optimizer = _optim.Adam(model.parameters(), lr=config["lr"])
    criterion = _F.cross_entropy

    if checkpoint_dir:
        model_state, optimizer_state = _torch.load(
            _os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_set, test_set = _default.DATASETS["MNIST"](_transforms.Compose([
        _transforms.ToTensor()
    ]))

    train_size = int(len(train_set) * 0.8)
    train, val = _random_split(
        train_set, [train_size, len(train_set) - train_size])

    train_loader = _DataLoader(
        train, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    val_loader = _DataLoader(
        val, batch_size=config["batch_size"], shuffle=False, num_workers=8)

    train_metrics = []
    val_metrics = []
    for epoch in range(epochs):
        print(
            f"---------- Epoch {epoch+1:{_math.ceil(_math.log10(epochs+1))}d} ----------")
        train_metrics.append(train_loop(
            model, optimizer, criterion, train_loader, verbose=verbose, dtype=_torch.bfloat16))
        val_metrics.append(val_loop(
            model, criterion, val_loader, dtype=_torch.bfloat16))

        with _tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = _os.path.join(checkpoint_dir, "checkpoint")
            _torch.save((model.state_dict(), optimizer.state_dict()), path)

        metrics_df = _pd.DataFrame(val_metrics[-1])

        _tune.report(loss=metrics_df["loss"].mean(), accuracy=(
            metrics_df["correct"].sum() / len(val_loader.dataset) * 100))

    return {
        "train": _pd.DataFrame(_chain.from_iterable(train_metrics)),
        "validation": _pd.DataFrame(_chain.from_iterable(val_metrics))
    }
