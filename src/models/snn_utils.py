import torch
from typing import Tuple, List

def poisson_encode(x: torch.Tensor, T: int, rate_scale: float = 1.0) -> torch.Tensor:
    """
    Generates Poisson-distributed spike trains from continuous input.
    
    Args:
        x: Input tensor [Batch, Dim].
        T: Number of time steps.
        rate_scale: Scaling factor for firing probability.
    
    Returns:
        Spike tensor [T, Batch, Dim] containing 0s and 1s.
    """
    # Probability of spiking at each step
    prob = torch.clamp(x.abs() * rate_scale, 0.0, 1.0)
    
    # Expand time dimension: [T, Batch, Dim]
    prob_expanded = prob.unsqueeze(0).expand(T, *prob.shape)
    
    # Sample Bernoulli
    return torch.rand_like(prob_expanded).le(prob_expanded).float()

def init_lif_states(
    x: torch.Tensor, 
    *hidden_dims: int
) -> List[torch.Tensor]:
    """
    Initializes membrane potential tensors for multiple layers.
    
    Args:
        x: Input tensor (used for getting batch size and device).
        hidden_dims: Integers representing the size of each hidden layer.
        
    Returns:
        List of initialized membrane tensors (zeros).
    """
    batch_size = x.size(0)
    device = x.device
    states = []
    
    for dim in hidden_dims:
        states.append(torch.zeros(batch_size, dim, device=device))
        
    return states