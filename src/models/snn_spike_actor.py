"""Spike-count actor built from Leaky Integrate-and-Fire blocks.

Wraps snntorch layers to encode inputs over `T` time steps, log spike stats, and
optionally concatenate critic predictions for actor-critic workflows.
"""
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Optional

from src.models.snn_utils import poisson_encode, init_lif_states
from src.models.snn_block import SNNBlock


class SNNSpikeActor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        *,
        beta: float = 0.95,
        V_th: float = 1.0,
        T: int = 32,
        poisson_encode: bool = False,
        rate_scale: float = 1.0,
        logit_temp: float = 1.0,
        center_logits: bool = True,
        use_potential_fallback: bool = False,
        actor_surrogate_slope: float = 25.0,
    ):
        super().__init__()

        self.T = int(T)
        self.use_poisson_encode = bool(poisson_encode)
        self.rate_scale = float(rate_scale)
        self.logit_temp = float(logit_temp)
        self.center_logits = bool(center_logits)
        self.use_potential_fallback = bool(use_potential_fallback)

        sg = surrogate.fast_sigmoid(slope=int(actor_surrogate_slope))

        # Stack three SNN blocks to grow a deep spike-processing actor.
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
        self._last_spike_count_total = 0.0
        self._last_spike_count_per_env = None
        self._last_latency = 0.0

    def forward_T(
        self,
        x: torch.Tensor,
        critic_value: Optional[torch.Tensor] = None,
        return_activations: bool = False,
        return_temporal: bool = False,
    ):
        B = x.size(0)

        if critic_value is not None:
            cv = critic_value.unsqueeze(-1) if critic_value.dim() == 1 else critic_value
            x = torch.cat([x, cv], dim=-1)

        # Encode the input spikes across time steps with optional Poisson noise.
        if self.use_poisson_encode:
            spk_in = poisson_encode(x, self.T, self.rate_scale)
        else:
            spk_in = x.unsqueeze(0).repeat(self.T, 1, 1)

        v1, v2, vout = init_lif_states(
            x,
            self.block1.linear.out_features,
            self.block2.linear.out_features,
            self.block_out.linear.out_features,
        )

        out_dim = self.block_out.linear.out_features
        out_counts = x.new_zeros(B, out_dim)
        iout_accum = torch.zeros_like(out_counts)

        # Track cumulative spike statistics used for the regularizer and diagnostics.
        spike_mean = 0.0
        per_env_spike_sum = torch.zeros(B, device=x.device, dtype=torch.float32)

        # Optionally accumulate activation summaries per layer/action.
        activations: dict = {}
        if return_activations:
            activations = {
                "layer_0": torch.zeros(B, device=x.device),
                "layer_1": torch.zeros(B, device=x.device),
                "output": torch.zeros(B, device=x.device),
                "output_per_action": torch.zeros(B, out_dim, device=x.device),
            }
            if return_temporal:
                activations["output_potential_trace"] = []
                activations["output_spike_trace"] = []

        first_spike_time = torch.full(
            (B, out_dim), self.T, dtype=torch.float32, device=x.device
        )

        for t in range(self.T):
            spk1, v1, _ = self.block1.forward_step(spk_in[t], v1)
            spk2, v2, _ = self.block2.forward_step(spk1, v2)
            spk_out, vout, iout = self.block_out.forward_step(spk2, vout)

            out_counts += spk_out
            iout_accum += iout

            spike_mean += spk1.mean() + spk2.mean() + spk_out.mean()

            # Accumulate spikes per environment (once per timestep, reused below).
            spk1_sum = spk1.sum(dim=-1).float()
            spk2_sum = spk2.sum(dim=-1).float()
            spk_out_sum = spk_out.sum(dim=-1).float()

            env_spikes_t = spk1_sum + spk2_sum + spk_out_sum
            per_env_spike_sum += env_spikes_t

            if return_activations:
                activations["layer_0"] += spk1_sum.detach()
                activations["layer_1"] += spk2_sum.detach()
                activations["output"] += spk_out_sum.detach()
                activations["output_per_action"] += spk_out.detach()
                if return_temporal:
                    activations["output_potential_trace"].append(iout.detach())
                    activations["output_spike_trace"].append(spk_out.detach())

            # Track the first spike time per action for latency statistics.
            just_spiked = (spk_out > 0) & (first_spike_time == self.T)
            first_spike_time[just_spiked] = float(t)

        self._last_reg = spike_mean / self.T
        self._last_spike_count_total = float(per_env_spike_sum.sum().item())
        self._last_spike_count_per_env = per_env_spike_sum.detach()
        self._last_latency = first_spike_time.mean().item()

        logits = iout_accum

        # If the output never spikes, fall back to membrane potential as logits.
        if self.use_potential_fallback:
            no_output_spike = (out_counts.sum(dim=-1, keepdim=True) == 0)
            logits = torch.where(no_output_spike, vout, logits)

        if self.center_logits:
            logits = logits - logits.mean(dim=-1, keepdim=True)

        logits = logits / max(self.logit_temp, 1e-6)

        if return_activations:
            if return_temporal:
                if len(activations["output_potential_trace"]) > 0:
                    activations["output_potential_trace"] = torch.stack(
                        activations["output_potential_trace"], dim=0
                    )
                    activations["output_spike_trace"] = torch.stack(
                        activations["output_spike_trace"], dim=0
                    )
                else:
                    activations["output_potential_trace"] = torch.zeros(
                        0, B, out_dim, device=x.device
                    )
                    activations["output_spike_trace"] = torch.zeros(
                        0, B, out_dim, device=x.device
                    )
            return logits, self._last_reg, activations

        return logits, self._last_reg

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard PyTorch forward — delegates to forward_T, returns logits only."""
        logits, _ = self.forward_T(x, **kwargs)
        return logits

    def regulariser(self):
        return self._last_reg

    def last_spike_count(self):
        return self._last_spike_count_per_env

    def last_spike_count_total(self):
        return self._last_spike_count_total

    def last_latency(self):
        return self._last_latency

    def get_spike_stats(self) -> dict:
        """
        Returns cumulative spike statistics across all blocks since last reset.
        Called by energy_benchmark.get_spike_stats_safe() and get_cumulative_spikes().
        """
        total_spikes = (
            self.block1.total_spikes
            + self.block2.total_spikes
            + self.block_out.total_spikes
        )
        total_timesteps = (
            self.block1.total_timesteps
            + self.block2.total_timesteps
            + self.block_out.total_timesteps
        )

        # Each block tracks its own tensor counters; convert them to scalars for reporting.
        if isinstance(total_spikes,    torch.Tensor): total_spikes    = total_spikes.detach().cpu().item()
        if isinstance(total_timesteps, torch.Tensor): total_timesteps = total_timesteps.detach().cpu().item()

        # Average spikes per timestep indicates how active the SNN is; sparsity is its complement.
        firing_rate = (total_spikes / float(total_timesteps)) if total_timesteps > 0 else 0.0
        sparsity    = 1.0 - firing_rate

        return {
            "total_spikes":    total_spikes,
            "sparsity":        sparsity,
            "actual_steps":    self.T,
            "total_timesteps": total_timesteps,
            "firing_rate":     firing_rate,
            "mean_latency":    float(self._last_latency),
        }
