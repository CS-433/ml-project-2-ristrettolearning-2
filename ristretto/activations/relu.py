import torch
import torch.nn as nn

# ------------------------------------------------------------
# Custom ReLU activation function
# The functions can be customized using the torch.autograd.Function class
# https://pytorch.org/docs/s    table/autograd.html#function for more details


# First we need to subclass torch.autograd.Function
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        """_This is a modified version of the ReLU function
        where the ReLU'(0) is not zero, but any alpha value_
        Args:
            ctx (-): It is a context object that can be used to stash information for backward computation.
            input (tensor): Input tensor.
            alpha (int): Alpha value, value of the derivative at 0: RELU'(0) = alpha.
        Returns:
            grad_output (tensor): Output tensor for the forward pass.
        """
        ctx.save_for_backward(input)
        ctx.alpha = alpha
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
        return grad_input, None

# Custom ReLU activation function as a module that can be used in a sequential model


class ReLU(nn.Module):
    "alpha = value of the derivative at 0: RELU'(0) = alpha."

    def __init__(self, alpha):
        super(ReLU, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return ReLUFunction.apply(input, self.alpha)
