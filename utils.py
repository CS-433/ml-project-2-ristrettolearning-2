# defining the train loop 
import torch
from torch import nn, optim
from typing import Optional
from CustomReLUs import *
import numpy as np

def train_loop(dataloader, model, loss_fn, DEVICE, optimizer):
    # getting the size of the batch just to messure the progress 
    size = len(dataloader.dataset)
    model.train(True)
    # For every bach of data....
    for batch, (X, y) in enumerate(dataloader):
        X , y = X.to(DEVICE), y.to(DEVICE)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation and optimization 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Progress of the training
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    model.train(False)

    return loss




def test_loop(dataloader, model, loss_fn, DEVICE, status = 'Test'):
   
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    test_loss = 0
    correct  =  0
    
    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X , y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"{status} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss



def train_model(
    epochs, model, optimizer , 
    train_dataloader, test_dataloader, 
    loss_fn, DEVICE): #, valid_dataloader = Optional[None]):
    
    train_loss = []
    test_loss = []
    valid_loss = []
    model = model.to(DEVICE)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_l = train_loop(train_dataloader, model, loss_fn, DEVICE, optimizer)
        te_l = test_loop(test_dataloader, model, loss_fn, DEVICE, status = 'Test')
        #v_l = test_loop(valid_dataloader, model, loss_fn, DEVICE, status = 'Validation')
        train_loss.append(tr_l)
        #valid_loss.append(v_l)
        test_loss.append(te_l) 
            
    print("Done!")
    
    return train_loss, test_loss #, valid_loss


def loss_to_numpy(losses):
  losses_ = []
  for i in range(len(losses)):
      losses_.append(losses[i].item())
  return np.array(losses_)