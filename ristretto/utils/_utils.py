import ristretto.utils.default as _default
import torch as _torch
import torch.nn.functional as _F
import torch.optim as _optim
import pandas as _pd
import numpy as _np
import math as _math
from itertools import chain as _chain
import random as _random

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
    verbose=False
):

    # getting the size of the batch just to measure the progress
    size = len(train_loader.dataset)

    model.train()
    metrics = []

    for batch, (X, y) in enumerate(train_loader):
        # train step
        X, y = X.to(device), y.to(device)
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
    metrics_fn=None
):

    model.eval()
    metrics = []

    for X, y in val_loader:
        # val step
        X, y = X.to(device), y.to(device)
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
            model, optimizer, criterion, train_loader, device, metrics_fn, verbose))
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
