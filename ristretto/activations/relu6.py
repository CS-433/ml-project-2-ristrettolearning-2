## adapted from## adapted from https://github.com/deel-ai/relu-prime/blob/6c359e0eab8fa12f710cadf50b333de2a8d1d24d/relu.py

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
# Custom ReLU6 activation function as a module that can be used in a sequential model


class ReLU6Function(torch.autograd.Function):
    print_when_zero = False

    @staticmethod
    def forward(ctx, input, alpha, beta, inplace):
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.save_for_backward(input.clone())

        if inplace:
            return input.clamp_(min=0, max=6)

        return input.clamp(min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 6] = 0
        grad_input[input == 0] = ctx.alpha
        grad_input[input == 6] = ctx.beta

        if ReLU6Function.print_when_zero:
            _sum = (input == 0).sum()
            if _sum > 0:
                print(
                    f"Found {_sum.item()} item{'s' if _sum.item() > 1 else ''} with input == 0")

            _sum = (input == 6).sum()
            if _sum > 0:
                print(
                    f"Found {_sum.item()} item{'s' if _sum.item() > 1 else ''} with input == 6")

        return grad_input, None, None, None


class ReLU6(nn.Module):
    def __init__(self, alpha, beta, inplace=False):
        super(ReLU6, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.inplace = inplace

    def forward(self, input):
        return ReLU6Function.apply(input, self.alpha, self.beta, self.inplace)
