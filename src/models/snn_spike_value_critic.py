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
    SNN critic that outputs a value estimate based on accumulated synaptic current.

    The mean current over T timesteps acts as the value estimate.  This matches
    the spike-count encoding used by SNNSpikeActor so the two can be trained
    together under the same surrogate-gradient PPO loop.
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
        critic_surrogate_slope: float = 25.0,
    ):
        super().__init__()

        self.T = int(T)

        self.use_poisson_encode = bool(poisson_encode)
        self.rate_scale         = float(rate_scale)

        sg = surrogate.fast_sigmoid(slope=int(critic_surrogate_slope))

        self.block1 = SNNBlock(
            nn.Linear(in_dim, hid_dim),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )
        self.block2 = SNNBlock(
            nn.Linear(hid_dim, hid_dim),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )
        self.block_out = SNNBlock(
            nn.Linear(hid_dim, 1),
            snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg),
        )

        # Stat tracking — required by energy_benchmark.get_cumulative_spikes()
        # and the surrogate trainer's latency probe.
        self._last_reg         = torch.tensor(0.0)
        self._last_spike_count = 0.0
        self._last_latency     = 0.0

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard value forward pass.  Returns [B, 1] value estimates.
        Delegates to forward_detailed so stat tracking runs every call.
        """
        v, _tau = self.forward_detailed(x)
        return v

    def forward_detailed(self, x: torch.Tensor):
        """
        Returns (value, latency_proxy).

        value           : [B, 1]  mean accumulated current over T steps.
        latency_proxy   : [B, 1]  same as value — spike-count critics do not
                          have a meaningful tau, but this matches the interface
                          expected by the surrogate trainer's critic probe
                          (which calls forward_detailed and reads tau).

        The surrogate trainer probe calls forward_detailed to obtain tau for
        diagnostic logging.  For timing-based critics (SNNTimingCritic) tau
        carries latency semantics; here it is a placeholder that mirrors value
        so downstream code reading tau.mean() gets a sensible scalar.
        """
        B = x.size(0)

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

        iout_accum = x.new_zeros(B, 1)

        spike_mean = 0.0
        spike_sum  = 0.0

        for t in range(self.T):
            spk1, v1, _          = self.block1.forward_step(spk_in[t], v1)
            spk2, v2, _          = self.block2.forward_step(spk1, v2)
            spk_out, vout, iout  = self.block_out.forward_step(spk2, vout)

            iout_accum += iout

            spike_mean += spk1.mean() + spk2.mean() + spk_out.mean()
            spike_sum  += (
                spk1.sum().item() + spk2.sum().item() + spk_out.sum().item()
            )

        # Update stat tracking so energy_benchmark and trainer probes work
        self._last_reg         = spike_mean / self.T
        self._last_spike_count = spike_sum
        # Latency proxy: not meaningful for a spike-count critic, but must be
        # a finite float so get_last_latency() does not return stale zeros.
        self._last_latency     = 0.0

        value = iout_accum / self.T

        # Return value and a latency placeholder that matches the tau interface
        # expected by forward_detailed callers.
        return value, value.detach()

    # ------------------------------------------------------------------

    def regulariser(self):
        return self._last_reg

    def last_spike_count(self):
        return self._last_spike_count

    def last_latency(self):
        return float(self._last_latency)

    def reset_stats(self):
        self.block1.reset_stats()
        self.block2.reset_stats()
        self.block_out.reset_stats()

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

        if isinstance(total_spikes,    torch.Tensor): total_spikes    = total_spikes.detach().cpu().item()
        if isinstance(total_timesteps, torch.Tensor): total_timesteps = total_timesteps.detach().cpu().item()

        firing_rate = (total_spikes / float(total_timesteps)) if total_timesteps > 0 else 0.0

        return {
            "total_spikes":    total_spikes,
            "sparsity":        1.0 - firing_rate,
            "actual_steps":    self.T,
            "total_timesteps": total_timesteps,
            "firing_rate":     firing_rate,
            "mean_latency":    float(self._last_latency),
        }