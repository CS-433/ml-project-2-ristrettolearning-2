import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
#import copy


#------------------------------------------------------------
# Custom ReLU activation function 
# The functions can be customized using the torch.autograd.Function class
# https://pytorch.org/docs/s    table/autograd.html#function for more details

class ReLUAlphaFunction(torch.autograd.Function): # First we need to subclass torch.autograd.Function
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

## Custom ReLU activation function as a module that can be used in a sequential model
class ReLUAlpha(nn.Module):
    "alpha = value of the derivative at 0: RELU'(0) = alpha."
    def __init__(self, alpha):
        super(ReLUAlpha, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return ReLUAlphaFunction.apply(input, self.alpha)
#--------------------------------------------------------------------------------


###Perhaps we don't need this becasue we can use the ReLUAlphaFunction directly
# class ReLURandomFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (input,) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         rdm = torch.rand_like(grad_output)
#         grad_input[input < 0] = 0
#         grad_input[input == 0] = rdm[input == 0]
#         return grad_input
#
# class ReLURandom(nn.Module):
#     def __init__(self):
#         super(ReLURandom, self).__init__()
#
#     def forward(self, input):
#         return ReLURandomFunction.apply(input)

#--------------------------------------------------------------------------------
# Custom ReLU6 activation function as a module that can be used in a sequential model
class ReLU6AlphaBetaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, beta):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.beta = beta
        return input.clamp(min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 6] = 0
        grad_input[input == 0] = ctx.alpha
        grad_input[input == 6] = ctx.beta
        return grad_input, None, None


class ReLU6AlphaBeta(nn.Module):
    def __init__(self, alpha, beta):
        super(ReLU6AlphaBeta, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input):
        return ReLU6AlphaBetaFunction.apply(input, self.alpha, self.beta)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
# Custom LeakyReLU activation function as a module that can be used in a sequential model
# alpha is the function at f'(0) = alpha
# Negative slope is the slope of the function at f'(x<0) = negative_slope*x
class LeakyReLUAlphaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, negative_slope, alpha):
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        ctx.alpha = alpha
        return input.clamp(min=0) + input.clamp(max=0) * negative_slope

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input <= 0] = ctx.negative_slope
        grad_input = (
            grad_input * (input > 0).float()
            + grad_input * (input < 0).float() * ctx.negative_slope
            + grad_input * (input == 0).float() * ctx.alpha
        ) #Check this expression
        return grad_input, None, None


class LeakyReLUAlpha(nn.Module):
    def __init__(self, negative_slope, alpha=None):
        super(LeakyReLUAlpha, self).__init__()
        self.negative_slope = negative_slope
        if alpha == None:
            self.alpha = negative_slope
        else:
            self.alpha = alpha

    def forward(self, input):
        return LeakyReLUAlphaFunction.apply(input, self.negative_slope, self.alpha)
#--------------------------------------------------------------------------------

# inputs1 = torch.tensor((0.0,1.0,2.0,3.0,4.0,5.0,-6.0,-7.0, 8), requires_grad=True)#torch.randn(100, requires_grad=True)
# inputs2 = copy.deepcopy(inputs1)
# inputs3 = copy.deepcopy(inputs1)
# outputs1 = LeakyReLUAlpha(0.2, 0.2)(inputs1)
# outputs2 = LeakyReLUAlpha(0.2, 3)(inputs2)
# outputs3 = nn.LeakyReLU(0.2)(inputs3)
# torch.sum(outputs1).backward()
# torch.sum(outputs2).backward()
# torch.sum(outputs3).backward()

# assert torch.equal(outputs1, outputs3)
# #assert torch.equal(outputs1, outputs2)

# assert not torch.equal(inputs1.grad, inputs2.grad)
# assert torch.equal(inputs1.grad, inputs3.grad)

# net1 = nn.Sequential(nn.Linear(9, 10),  LeakyReLUAlpha(0.2,0.2),  nn.Linear(10, 10))
# net2 = nn.Sequential(nn.Linear(9, 10),  nn.LeakyReLU(0.2),  nn.Linear(10, 10))
# net2.load_state_dict(net1.state_dict())

# inputs = torch.tensor((0.0,1.0,2.0,3.0,4.0,5.0,-6.0,-7.0, 8), requires_grad=True)
# outputs1 = net1(inputs)
# outputs2 = net2(inputs)
# torch.sum(outputs2).backward()
# torch.sum(outputs1).backward()

# assert torch.equal(net1[0].weight.grad, net2[0].weight.grad)