import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
# Custom LeakyReLU activation function as a module that can be used in a sequential model
# alpha is the function at f'(0) = alpha
# Negative slope is the slope of the function at f'(x<0) = negative_slope*x


class LeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope, alpha, inplace):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        ctx.alpha = alpha

        if inplace:
            result = input.clamp(min=0) + input.clamp(max=0) * negative_slope
            ctx.save_for_backward(result)
            return result

        result = input.clamp(min=0) + input.clamp(max=0) * negative_slope
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input <= 0] = ctx.negative_slope
        grad_input = (
            grad_input * (input > 0).float()
            + grad_input * (input < 0).float() * ctx.negative_slope
            + grad_input * (input == 0).float() * ctx.alpha
        )  # Check this expression
        return grad_input, None, None


class LeakyReLU(nn.Module):
    """
    LeakyReLU activation function as a module that can be used in a sequential model
    negative_slope is the slope of the function at LeakyReLU(x<0) = negative_slope*x
    alpha is the subgradient at LeakyReLU'(0) = alpha
    """

    def __init__(self, negative_slope, alpha=None, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        if alpha == None:
            self.alpha = negative_slope
        else:
            self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return LeakyReLUFunction.apply(input, self.negative_slope, self.alpha, self.inplace)
