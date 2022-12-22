## adapted from ## adapted from https://github.com/deel-ai/relu-prime/blob/6c359e0eab8fa12f710cadf50b333de2a8d1d24d/relu.py

import torch
import torch.nn as nn

# ------------------------------------------------------------
# Custom ReLU activation function
# The functions can be customized using the torch.autograd.Function class
# https://pytorch.org/docs/s    table/autograd.html#function for more details


# First we need to subclass torch.autograd.Function
class ReLUFunction(torch.autograd.Function):
    print_when_zero = False

    @staticmethod
    def forward(ctx, input, alpha, inplace):
        """_This is a modified version of the ReLU function
        where the ReLU'(0) is not zero, but any alpha value_
        Args:
            ctx (-): It is a context object that can be used to stash information for backward computation.
            input (tensor): Input tensor.
            alpha (int): Alpha value, value of the derivative at 0: RELU'(0) = alpha.
        Returns:
            grad_output (tensor): Output tensor for the forward pass.
        """
        ctx.save_for_backward(input.clone())
        ctx.alpha = alpha
        if inplace:
            # ctx.mark_dirty(input)
            return input.clamp_(min=0)

        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the gradient of the modified ReLU function
        Args:
            ctx (_type_): It is a context object that can be used to stash information for backward computation.
            grad_output (_type_): Output tensor from the forward pass.
        Returns:
            _type_: Output tensor with the gradient of the modified ReLU function.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input == 0] = ctx.alpha

        if ReLUFunction.print_when_zero:
            _sum = (input == 0).sum()
            if _sum > 0:
                print(
                    f"Found {_sum.item()} item{'s' if _sum.item() > 1 else ''} with input == 0")

        return grad_input, None, None

# Custom ReLU activation function as a module that can be used in a sequential model


class ReLU(nn.Module):
    "alpha = value of the derivative at 0: ReLU'(0) = alpha."

    def __init__(self, alpha, inplace=False):
        super(ReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return ReLUFunction.apply(input, self.alpha, self.inplace)
