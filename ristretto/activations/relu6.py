import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
# Custom ReLU6 activation function as a module that can be used in a sequential model


class ReLU6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, beta, inplace):
        ctx.alpha = alpha
        ctx.beta = beta

        if inplace:
            result = input.clamp_(min=0, max=6)
            ctx.save_for_backward(result)
            return result

        result = input.clamp(min=0, max=6)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 6] = 0
        grad_input[input == 0] = ctx.alpha
        grad_input[input == 6] = ctx.beta
        return grad_input, None, None, None


class ReLU6(nn.Module):
    def __init__(self, alpha, beta, inplace=False):
        super(ReLU6, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.inplace = inplace

    def forward(self, input):
        return ReLU6Function.apply(input, self.alpha, self.beta, self.inplace)
