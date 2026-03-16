"""Spike-count based SNN actor using Leaky Integrate-and-Fire neurons and Poisson encoding."""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.models.snn_utils import poisson_encode, init_lif_states
from src.models.snn_block import SNNBlock

class SNNSpikeActor(nn.Module):
    """
    Spike-count based SNN actor, compatible with ANN BackboneNetwork interface.

    Inputs are Poisson-encoded into spike trains over T timesteps.  Output logits
    are derived from accumulated synaptic current across the simulation window.
    """

    def __init__(
            self,
            in_dim: int,
            hid_dim: int,
            out_dim: int,
            *,
            beta: float = 0.95,
            V_th: float = 1.0,
            T: int = 32,
            poisson_encode: bool = True,
            rate_scale: float = 1.0,
            logit_temp: float = 1.0,
            center_logits: bool = True,
            use_potential_fallback: bool = True,
            actor_surrogate_slope: float = 25.0,
    ):
        super().__init__()

        self.T = int(T)
        self.poisson_encode = poisson_encode
        self.rate_scale = rate_scale
        self.logit_temp = logit_temp
        self.center_logits = center_logits
        self.use_potential_fallback = use_potential_fallback

        sg = surrogate.fast_sigmoid(slope=int(actor_surrogate_slope))

        self.block1 = SNNBlock(
            nn.Linear(in_dim, hid_dim),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )
        self.block2 = SNNBlock(
            nn.Linear(hid_dim, hid_dim),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )
        self.block_out = SNNBlock(
            nn.Linear(hid_dim, out_dim),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )

        self._last_reg = torch.tensor(0.0)
        self._last_spike_count = 0.0
        self._last_latency = 0.0

    # -------------------------------------------------------------

    def forward(self, x: torch.Tensor, critic_value: torch.Tensor = None, return_activations: bool = False, **kwargs) -> torch.Tensor:
        """
        Standard forward pass compatible with ANN backbone signature.
        """
        return_temporal = bool(kwargs.get("return_temporal", False))
        # Call the time-stepped forward pass
        result = self.forward_T(
            x,
            critic_value=critic_value,
            return_activations=return_activations,
            return_temporal=return_temporal,
        )

        # Unpack based on what was requested
        if return_activations:
            logits, _, activations = result
            return logits, activations

        # Default behavior: return just logits
        logits, _ = result
        return logits

    def forward_T(
        self,
        x: torch.Tensor,
        critic_value: torch.Tensor = None,
        return_activations: bool = False,
        return_temporal: bool = False,
    ):
        B = x.size(0)

        # --- CRITIC INFORMS ACTOR LOGIC ---
        if critic_value is not None:
            if critic_value.dim() == 1:
                cv = critic_value.unsqueeze(-1)
            else:
                cv = critic_value
            x = torch.cat([x, cv], dim=-1)

        if self.poisson_encode:
            spk_in = poisson_encode(x, self.T, self.rate_scale)
        else:
            spk_in = x.unsqueeze(0).repeat(self.T, 1, 1)

        v1, v2, vout = init_lif_states(
            x,
            self.block1.linear.out_features,
            self.block2.linear.out_features,
            self.block_out.linear.out_features,
        )

        out_counts = x.new_zeros(B, self.block_out.linear.out_features)
        iout_accum = torch.zeros_like(out_counts)

        spike_mean = 0.0
        spike_sum = 0.0

        activations = {}
        if return_activations:
            activations = {
                "layer_0": torch.zeros(B, device=x.device),
                "layer_1": torch.zeros(B, device=x.device),
                "output": torch.zeros(B, device=x.device),
                "output_per_action": torch.zeros(B, self.block_out.linear.out_features, device=x.device),
            }
            if return_temporal:
                activations["output_potential_trace"] = []
                activations["output_spike_trace"] = []

        out_dim = self.block_out.linear.out_features
        first_spike_time = torch.full((B, out_dim), self.T, dtype=torch.float32, device=x.device)

        for t in range(self.T):
            spk1, v1, _ = self.block1.forward_step(spk_in[t], v1)
            spk2, v2, _ = self.block2.forward_step(spk1, v2)
            spk_out, vout, iout = self.block_out.forward_step(spk2, vout)

            out_counts += spk_out
            iout_accum += iout

            # Accumulate metrics
            spike_mean += spk1.mean() + spk2.mean() + spk_out.mean()
            spike_sum += spk1.sum().item() + spk2.sum().item() + spk_out.sum().item()

            if return_activations:
                activations['layer_0'] += spk1.sum(dim=-1).detach()
                activations['layer_1'] += spk2.sum(dim=-1).detach()
                activations['output'] += spk_out.sum(dim=-1).detach()
                activations['output_per_action'] += spk_out.detach()
                if return_temporal:
                    # Store per-timestep output traces for exact internal tau visualization.
                    activations["output_potential_trace"].append(iout.detach())
                    activations["output_spike_trace"].append(spk_out.detach())

            # Latency tracking
            just_spiked = (spk_out > 0) & (first_spike_time == self.T)
            first_spike_time[just_spiked] = t

        self._last_reg = spike_mean / self.T
        self._last_spike_count = spike_sum
        self._last_latency = first_spike_time.mean().item()

        logits = iout_accum

        # Potential fallback for zero-spike frames
        if self.use_potential_fallback:
            zero_mask = logits.sum(dim=-1) == 0
            if zero_mask.any():
                logits = logits.clone()
                logits[zero_mask] = iout_accum[zero_mask] / self.T

        if self.center_logits:
            logits -= logits.mean(dim=-1, keepdim=True)

        logits /= max(self.logit_temp, 1e-6)

        if return_activations:
            if return_temporal:
                if len(activations["output_potential_trace"]) > 0:
                    activations["output_potential_trace"] = torch.stack(activations["output_potential_trace"], dim=0)
                    activations["output_spike_trace"] = torch.stack(activations["output_spike_trace"], dim=0)
                else:
                    activations["output_potential_trace"] = torch.zeros(0, B, self.block_out.linear.out_features, device=x.device)
                    activations["output_spike_trace"] = torch.zeros(0, B, self.block_out.linear.out_features, device=x.device)
            return logits, self._last_reg, activations

        return logits, self._last_reg

    # -------------------------------------------------------------

    def regulariser(self):
        return self._last_reg

    def last_spike_count(self):
        return self._last_spike_count

    def last_latency(self):
        return self._last_latency

    def reset_stats(self):
        self.block1.reset_stats()
        self.block2.reset_stats()
        self.block_out.reset_stats()
    
    def get_spike_stats(self):
        """
        Returns cumulative spike statistics across all blocks since the last reset.
        """
        total_spikes = (
            self.block1.total_spikes + 
            self.block2.total_spikes + 
            self.block_out.total_spikes
        )
        
        total_timesteps = (
            self.block1.total_timesteps + 
            self.block2.total_timesteps + 
            self.block_out.total_timesteps
        )

        if isinstance(total_spikes, torch.Tensor):
            total_spikes = total_spikes.detach().cpu().item()
        if isinstance(total_timesteps, torch.Tensor):
            total_timesteps = total_timesteps.detach().cpu().item()

        sparsity = 0.0
        if total_timesteps > 0:
            # Sparsity = inactive fraction; density/firing rate = total_spikes / total_timesteps
            sparsity = 1.0 - (total_spikes / float(total_timesteps))

        return {
            "total_spikes": total_spikes,
            "sparsity": sparsity,
            "actual_steps": self.T,
            "total_timesteps": total_timesteps,
            "firing_rate": (total_spikes / float(total_timesteps)) if total_timesteps > 0 else 0.0,
            "mean_latency": float(self._last_latency),
        }
