# import torch
# import torch.nn as nn

# import snntorch as snn

# from src.models.snn_utils import poisson_encode, init_lif_states
# from src.models.snn_block import SNNBlock
# from src.models.surrogates import cosh_surrogate


# class SNNSpikeValueCritic(nn.Module):
#     """
#     Spike-count value critic with cosh surrogate for ANN→SNN conversion.
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         hid_dim: int,
#         *,
#         beta: float = 0.95,
#         V_th: float = 1.0,
#         T: int = 32,
#         poisson_encode: bool = True,
#         rate_scale: float = 1.0,
#     ):
#         super().__init__()

#         sg = cosh_surrogate(alpha=10.0, beta=1.0)

#         self.T = int(T)
#         self.poisson_encode = poisson_encode
#         self.rate_scale = rate_scale

#         self.block1 = SNNBlock(
#             nn.Linear(in_dim, hid_dim),
#             snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
#         )
#         self.block2 = SNNBlock(
#             nn.Linear(hid_dim, hid_dim),
#             snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
#         )
#         self.block_out = SNNBlock(
#             nn.Linear(hid_dim, 1),
#             snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
#         )

#         self._last_reg = torch.tensor(0.0)
#         self._last_spike_count = 0.0

#     def forward(self, x):
#         value, _ = self.forward_T(x)
#         return value.squeeze(-1)

#     def forward_T(self, x, *, return_activations: bool = False):
#         B = x.size(0)

#         if self.poisson_encode:
#             spk_in = poisson_encode(x, self.T, self.rate_scale)
#         else:
#             spk_in = x.unsqueeze(0).repeat(self.T, 1, 1)

#         v1, v2, _ = init_lif_states(
#             x,
#             self.block1.linear.out_features,
#             self.block2.linear.out_features,
#             1, # Dummy out_features for vout, not used anymore
#         )

#         spike_mean = 0.0
#         spike_sum = 0.0
        
#         activations = {} if return_activations else None

#         accumulated_value = 0.0
        
#         for t in range(self.T):
#             spk1, v1, _ = self.block1.forward_step(spk_in[t], v1)
#             spk2, v2, _ = self.block2.forward_step(spk1, v2)
            
#             # --- FIX: Linear Readout (Integrate without Fire) ---
#             # We pass the hidden spikes directly to the linear layer.
#             # This allows the value to be negative.
#             current_t = self.block_out.linear(spk2)
#             accumulated_value += current_t
            
#             spike_mean += spk1.mean() + spk2.mean()
#             spike_sum += spk1.sum().item() + spk2.sum().item()
            
#             if return_activations:
#                 # Note: We detach() to save memory during inference/logging
#                 s1 = spk1.sum(dim=-1).detach()
#                 s2 = spk2.sum(dim=-1).detach()
                
#                 # For the output, we don't have spikes anymore, so we log the continuous value
#                 # or just skip logging 'output' spikes.
#                 out_val = current_t.flatten().detach()

#                 if t == 0:
#                     activations['layer_0'] = s1
#                     activations['layer_1'] = s2
#                     activations['output'] = out_val
#                 else:
#                     activations['layer_0'] += s1
#                     activations['layer_1'] += s2
#                     activations['output'] += out_val

#         # Calculate mean over time
#         final_value = accumulated_value / self.T
        
#         # If logging activations, we usually want the MEAN value for the output, 
#         # but the SUM for spikes. Let's keep it consistent with the accumulation loop.
#         if activations:
#             activations['output'] = activations['output'] / self.T

#         self._last_reg = spike_mean / self.T
#         self._last_spike_count = spike_sum

#         if return_activations:
#             return final_value, self._last_reg, activations
            
#         return final_value, self._last_reg



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