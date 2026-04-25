"""Timing-based SNN critic that encodes value estimates via first-spike latency.

Higher-value states produce earlier output spikes; lower-value states produce
later spikes or no spike within the simulation window.
"""
import torch
import torch.nn as nn
import snntorch as snn

from src.models.snn_block import SNNBlock
from src.models.surrogates import make_surrogate
from src.models.snn_utils import poisson_encode


class SNNTimingCritic(nn.Module):
    """
    Timing-based SNN critic.

    Value is encoded by the FIRST spike timing of the output neuron: earlier
    spikes map to higher values; later (or absent) spikes map to lower values.

    ``forward()`` delegates to ``forward_detailed()`` and discards tau,
    so all stat tracking, latency recording, and vout storage happen in one
    place.  The surrogate trainer's critic probe calls ``forward_detailed``
    directly to obtain tau for diagnostic logging.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        *,
        beta: float = 0.95,
        V_th: float = 1.0,
        critic_T: int = 8,
        Rmax: float = 500.0,
        Rmin: float = 0.0,
        gamma: float = 0.99,
        spike_temp: float = 25.0,
        poisson_encode: bool = True,
        rate_scale: float = 1.0,
        critic_surrogate_type: str = "cosh",
        critic_surrogate_slope: float = 25.0,
        cosh_alpha: float = 10.0,
        cosh_beta: float = 1.0,
        use_hard_no_spike: bool = False,
        scale_by_discount: bool = False,
        value_affine: bool = False,
        lambda_tau: float = 0.0,
    ):
        super().__init__()

        self.T               = int(critic_T)
        self.V_th            = float(V_th)
        self.Rmax            = float(Rmax)
        self.Rmin            = float(Rmin)
        self.gamma           = float(gamma)
        self.spike_temp      = float(spike_temp)
        self.use_poisson_encode = bool(poisson_encode)
        self.rate_scale         = float(rate_scale)
        self.use_hard_no_spike  = bool(use_hard_no_spike)
        self.scale_by_discount  = bool(scale_by_discount)
        self.lambda_tau         = float(lambda_tau)

        sg = make_surrogate(
            critic_surrogate_type,
            slope=critic_surrogate_slope,
            alpha=cosh_alpha,
            beta=cosh_beta,
        )

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

        self.value_mapper = (
            nn.Linear(1, 1) if value_affine else nn.Identity()
        )
        if value_affine and isinstance(self.value_mapper, nn.Linear):
            nn.init.ones_(self.value_mapper.weight)
            nn.init.zeros_(self.value_mapper.bias)

        self._last_reg     = torch.tensor(0.0)
        self._last_latency = 0.0
        self._last_tau     = torch.tensor(float(self.T) / 2.0)  # live tensor for tau reg

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _soft_first_spike_time(self, v_traj: torch.Tensor) -> torch.Tensor:
        """Differentiable expected first spike time via soft sigmoid."""
        T, B, _ = v_traj.shape

        p = torch.sigmoid(self.spike_temp * (v_traj - self.V_th)).squeeze(-1)
        p = p.clamp(1e-6, 1.0 - 1e-6)

        no_prev = torch.cat(
            [torch.ones(1, B, device=p.device), torch.cumprod(1 - p, dim=0)[:-1]],
            dim=0,
        )
        q = no_prev * p
        q = q / q.sum(dim=0, keepdim=True)

        t_idx = torch.arange(T, device=p.device).float().unsqueeze(1)
        return (q * t_idx).sum(dim=0)

    def _map_tau_to_value(self, tau: torch.Tensor) -> torch.Tensor:
        """Map first-spike time to scalar value in [Rmin, Rmax]."""
        N    = float(self.T)
        half = N / 2.0

        scale     = 2.0 / N
        pos_scale = scale * self.Rmax   # early spikes  → high value
        neg_scale = scale * self.Rmin   # late spikes   → low value

        if self.scale_by_discount:
            denom = max(1e-6, 1.0 - self.gamma)
            pos_scale /= denom
            neg_scale /= denom

        v   = torch.full_like(tau, self.Rmin)
        pos = tau < half
        v[pos]  = (half - tau[pos])  * pos_scale
        v[~pos] = (tau[~pos] - half) * neg_scale
        return v

    # ------------------------------------------------------------------
    # Core simulation loop (shared by forward and forward_detailed)
    # ------------------------------------------------------------------

    def _run_timesteps(self, obs: torch.Tensor):
        """
        Execute the T-step LIF simulation.  Returns (v_traj, spike_energy).

        v_traj: [T, B, 1] membrane potential of the output neuron.
        spike_energy: scalar mean spike rate across all layers and timesteps.
        """
        B = obs.size(0)
        H = self.block1.linear.out_features

        v1   = obs.new_zeros(B, H)
        v2   = obs.new_zeros(B, H)
        vout = obs.new_zeros(B, 1)

        if self.use_poisson_encode:
            spk_in = poisson_encode(obs, self.T, self.rate_scale)
        else:
            spk_in = obs.unsqueeze(0).repeat(self.T, 1, 1)

        spike_energy = 0.0
        v_traj       = []

        for t in range(self.T):
            spk1, v1, _          = self.block1.forward_step(spk_in[t], v1)
            spk2, v2, _          = self.block2.forward_step(spk1, v2)
            spk_out, vout, _     = self.block_out.forward_step(spk2, vout)

            v_traj.append(vout.unsqueeze(0))
            spike_energy += spk1.mean() + spk2.mean() + spk_out.mean()

        return torch.cat(v_traj, dim=0), spike_energy   # [T, B, 1], scalar

    # ------------------------------------------------------------------
    # Public forward interface
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.  Calls forward_detailed and discards tau.
        """
        v, _tau = self.forward_detailed(obs)
        return v

    def forward_detailed(self, obs: torch.Tensor):
        """
        Returns (value, tau).

        value : [B, 1] scalar value estimate in [Rmin, Rmax].
        tau   : [B]    soft first-spike time for each batch element.

        Used by the surrogate trainer's critic probe for diagnostic logging.
        """
        v_traj, spike_energy = self._run_timesteps(obs)

        self._last_reg  = spike_energy / self.T
        self._last_vout = v_traj.detach()   # retained for downstream diagnostics

        tau = self._soft_first_spike_time(v_traj)
        self._last_tau     = tau                        # kept as tensor for tau regularisation
        self._last_latency = float(tau.mean().item())

        v = self._map_tau_to_value(tau)

        if self.use_hard_no_spike:
            no_spike = (v_traj.squeeze(-1) > self.V_th).any(dim=0) == 0
            v[no_spike] = self.Rmin

        v = v.clamp(self.Rmin, self.Rmax).unsqueeze(-1)
        v = self.value_mapper(v)
        v = v.clamp(self.Rmin, self.Rmax)

        return v, tau

    # ------------------------------------------------------------------
    # Stats & metrics
    # ------------------------------------------------------------------

    def regulariser(self):
        # Spike-count reg + optional tau penalty to prevent slow-mode locking.
        # Tau penalty penalises late spiking (large tau / T) and keeps gradients
        # flowing through the critic even when the value loss is near zero.
        if self.lambda_tau > 0.0 and isinstance(self._last_tau, torch.Tensor):
            tau_penalty = self.lambda_tau * self._last_tau.mean() / float(self.T)
            return self._last_reg + tau_penalty
        return self._last_reg

    def last_latency(self) -> float:
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
            self.block1.total_spikes + self.block2.total_spikes + self.block_out.total_spikes
        )
        total_timesteps = (
            self.block1.total_timesteps + self.block2.total_timesteps + self.block_out.total_timesteps
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