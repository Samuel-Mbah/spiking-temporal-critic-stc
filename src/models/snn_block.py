import torch
import torch.nn as nn
import snntorch as snn
from typing import Tuple

class SNNBlock(nn.Module):
    """
    A single layer of an SNN: Linear Transformation -> Leaky Integrate-and-Fire.
    Tracks spike statistics for energy estimation.
    """
    def __init__(self, linear: nn.Linear, lif: snn.Leaky):
        super().__init__()
        self.linear = linear
        self.lif = lif
        
        # Instrumentation state
        self.register_buffer("total_spikes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("total_timesteps", torch.tensor(0, dtype=torch.long))

    def forward_step(self, x: torch.Tensor, mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs one time-step of the layer.
        
        Args:
            x: Input spikes [Batch, InFeatures]
            mem: Current membrane potential [Batch, OutFeatures]
            
        Returns:
            spk: Output spikes [Batch, OutFeatures]
            mem: Updated membrane potential
            cur: Current (Synaptic input) - often used for "soft" readout
        """
        cur = self.linear(x)
        spk, mem = self.lif(cur, mem)
        
        # Track stats (detached to avoid graph retention)
        if self.training or not torch.is_grad_enabled():
            self.total_spikes += spk.detach().sum().long()
            self.total_timesteps += (x.size(0) * self.linear.out_features)
            
        return spk, mem, cur

    def reset_stats(self):
        """Resets the internal spike counters."""
        self.total_spikes.zero_()
        self.total_timesteps.zero_()