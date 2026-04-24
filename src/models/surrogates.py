"""Surrogate gradient functions for backpropagation through discrete spike events.

Implements FastSigmoid and CoshFunction surrogate gradients compatible with snntorch.
"""
import torch
import torch.nn as nn
from typing import Any
from snntorch import surrogate as snn_surrogate

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
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> Any:
        (input,) = ctx.saved_tensors
        slope = ctx.slope
        grad_output = grad_outputs[0]
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
        def backward(ctx, *grad_outputs):
            (input,) = ctx.saved_tensors
            grad_output = grad_outputs[0]
            sgax = (input * alpha)
            grad_x = (1.0 / (beta * torch.cosh(sgax)) ** 2) * alpha * 0.5  # Approx derivative
            return grad_output * grad_x

    return CoshFunction.apply


def _normalize_surrogate_name(name: str) -> str:
    norm = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fastsigmoid": "fast_sigmoid",
        "snntorch_fast_sigmoid": "fast_sigmoid",
        "cosh_surrogate": "cosh",
    }
    return aliases.get(norm, norm)


def make_surrogate(
    surrogate_type: str = "fast_sigmoid",
    *,
    slope: float = 25.0,
    alpha: float = 10.0,
    beta: float = 1.0,
):
    """Return a spike_grad callable usable by snntorch neuron layers."""
    name = _normalize_surrogate_name(surrogate_type)

    if name == "fast_sigmoid":
        # Preserve historical behavior used by the actor implementation.
        return snn_surrogate.fast_sigmoid(slope=int(slope))

    if name == "cosh":
        return cosh_surrogate(alpha=alpha, beta=beta)

    raise ValueError(
        f"Unknown surrogate_type '{surrogate_type}'. Supported: fast_sigmoid, cosh"
    )
