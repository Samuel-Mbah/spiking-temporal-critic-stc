"""Surrogate gradient functions for backpropagation through discrete spike events.

Implements FastSigmoid and CoshFunction surrogate gradients compatible with snntorch.
"""
import torch
import torch.nn as nn
from typing import Any

class FastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient function for spiking neurons.
    Implements the 'Fast Sigmoid' derivative: f'(u) = 1 / (1 + k|u|)^2
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.slope = slope
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        (input,) = ctx.saved_tensors
        slope = ctx.slope
        grad_input = grad_output.clone()
        
        # Derivative of fast sigmoid
        grad = slope / ((1 + torch.abs(slope * input)) ** 2)
        return grad_input * grad, None

def fast_sigmoid_surrogate(slope: float = 25.0):
    """Factory function for FastSigmoid spike generation."""
    def spike_fn(input_tensor):
        return FastSigmoid.apply(input_tensor, slope)
    return spike_fn

def cosh_surrogate(alpha: float = 10.0, beta: float = 1.0):
    """
    Hyperbolic cosine surrogate gradient.
    Commonly used in recent temporal coding papers.
    """
    class CoshFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            sgax = (input * alpha)
            grad_x = (1.0 / (beta * torch.cosh(sgax)) ** 2) * alpha * 0.5  # Approx derivative
            return grad_output * grad_x

    return CoshFunction.apply
