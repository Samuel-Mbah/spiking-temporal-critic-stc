"""
Spike-count value critic for ANN-to-SNN conversion workflows.

Implements an SNN critic that estimates state values by integrating
accumulated membrane current over a fixed simulation window (T steps).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.models.snn_utils import poisson_encode, init_lif_states
from src.models.snn_block import SNNBlock

class SNNSpikeValueCritic(nn.Module):
    """
    SNN Critic that outputs a value estimate based on accumulated current/spikes.
    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        *,
        beta: float = 0.95,
        V_th: float = 1.0,
        T: int = 32,
        poisson_encode: bool = False,
        rate_scale: float = 1.0,
    ):
        super().__init__()
        self.T = int(T)
        self.poisson_encode = poisson_encode
        self.rate_scale = rate_scale

        sg = surrogate.fast_sigmoid(slope=25.0)

        # Critic typically has 1 output neuron (Value)
        self.block1 = SNNBlock(nn.Linear(in_dim, hid_dim), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))
        self.block2 = SNNBlock(nn.Linear(hid_dim, hid_dim), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))
        self.block_out = SNNBlock(nn.Linear(hid_dim, 1), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        if self.poisson_encode:
            spk_in = poisson_encode(x, self.T, self.rate_scale)
        else:
            spk_in = x.unsqueeze(0).repeat(self.T, 1, 1)

        v1, v2, vout = init_lif_states(x, 
            self.block1.linear.out_features,
            self.block2.linear.out_features,
            self.block_out.linear.out_features
        )

        iout_accum = 0.0

        for t in range(self.T):
            spk1, v1, _ = self.block1.forward_step(spk_in[t], v1)
            spk2, v2, _ = self.block2.forward_step(spk1, v2)
            spk_out, vout, iout = self.block_out.forward_step(spk2, vout)

            iout_accum += iout

        # Average current over time acts as the Value estimate
        return iout_accum / self.T
    
    def reset_stats(self):
        self.block1.reset_stats()
        self.block2.reset_stats()
        self.block_out.reset_stats()