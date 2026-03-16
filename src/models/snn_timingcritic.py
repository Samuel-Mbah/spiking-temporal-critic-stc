import torch
import torch.nn as nn
import snntorch as snn

from src.models.snn_block import SNNBlock
from src.models.surrogates import cosh_surrogate
from src.models.snn_utils import poisson_encode

class SNNTimingCritic(nn.Module):
    """
    Timing-based SNN critic.
    Value is encoded by FIRST spike timing of output neuron.
    
    Merged Version:
    - Logic: Matches 'cartpole/src' (Working version)
    - Structure: Includes 'root' version utilities (reset_stats, forward_detailed)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        *,
        beta: float = 0.95,
        V_th: float = 1.0,
        critic_T: int = 8,        # RESTORED: Working default (8) instead of Root (32)
        Rmax: float = 500.0,      # RESTORED: Working default (500) instead of Root (100)
        Rmin: float = 0.0,
        gamma: float = 0.99,
        spike_temp: float = 25.0,
        poisson_encode: bool = True,
        rate_scale: float = 1.0,
        cosh_alpha: float = 10.0,
        cosh_beta: float = 1.0,
        use_hard_no_spike: bool = False, # RESTORED: Default False
        scale_by_discount: bool = False,
        value_affine: bool = False,
    ):
        super().__init__()

        self.T = int(critic_T)
        self.V_th = float(V_th)
        self.Rmax = float(Rmax)
        self.Rmin = float(Rmin)
        self.gamma = float(gamma)
        self.spike_temp = float(spike_temp)
        self.poisson_encode = poisson_encode
        self.rate_scale = float(rate_scale)
        self.use_hard_no_spike = bool(use_hard_no_spike)
        self.scale_by_discount = bool(scale_by_discount)

        sg = cosh_surrogate(alpha=cosh_alpha, beta=cosh_beta)

        # --- Network (Standard Init - No Positive Bias Hack) ---
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
        
        # Optional value scaling
        if value_affine:
            self.value_mapper = nn.Linear(1, 1)
            nn.init.ones_(self.value_mapper.weight)
            nn.init.zeros_(self.value_mapper.bias)
        else:
            self.value_mapper = nn.Identity()

        self._last_reg = torch.tensor(0.0)
        self._last_latency = 0.0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _soft_first_spike_time(self, v_traj: torch.Tensor) -> torch.Tensor:
        """Differentiable expected first spike time."""
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
        """Map timing to scalar value."""
        N = float(self.T)
        half = N / 2.0

        scale = (2.0 / N)
        pos_scale = scale * self.Rmax
        neg_scale = scale * self.Rmin

        if self.scale_by_discount:
            denom = max(1e-6, 1.0 - self.gamma)
            pos_scale /= denom
            neg_scale /= denom

        v = torch.full_like(tau, self.Rmin)
        pos = tau < half

        v[pos] = (half - tau[pos]) * pos_scale
        v[~pos] = (tau[~pos] - half) * neg_scale

        return v
        
        # alpha = tau / float(self.T)
        # v = (1.0 - alpha) * self.Rmax + alpha * self.Rmin
        # return v

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.size(0)
        H = self.block1.linear.out_features

        v1 = obs.new_zeros(B, H)
        v2 = obs.new_zeros(B, H)
        vout = obs.new_zeros(B, 1)

        spike_energy = 0.0
        v_traj = []

        if self.poisson_encode:
            spk_in = poisson_encode(obs, self.T, self.rate_scale)
        else:
            spk_in = obs.unsqueeze(0).repeat(self.T, 1, 1)

        for t in range(self.T):
            i1 = self.block1.linear(spk_in[t])
            s1, v1 = self.block1.lif(i1, v1)

            i2 = self.block2.linear(s1)
            s2, v2 = self.block2.lif(i2, v2)

            iout = self.block_out.linear(s2)
            sout, vout = self.block_out.lif(iout, vout)

            v_traj.append(vout.unsqueeze(0))
            spike_energy += s1.mean() + s2.mean() + sout.mean()

        self._last_reg = spike_energy / self.T

        v_traj = torch.cat(v_traj, dim=0)  # [T,B,1]
        self._last_vout = v_traj.detach()

        tau = self._soft_first_spike_time(v_traj)
        v = self._map_tau_to_value(tau)

        if self.use_hard_no_spike:
            no_spike = (v_traj.squeeze(-1) > self.V_th).any(dim=0) == 0
            v[no_spike] = self.Rmin

        v = v.clamp(self.Rmin, self.Rmax).unsqueeze(-1)
        v = self.value_mapper(v)
        return v.clamp(self.Rmin, self.Rmax)
    
    
    def forward_detailed(self, obs: torch.Tensor):
        """
        Returns both the Value and the First-Spike-Time (tau).
        """
        B = obs.size(0)
        H = self.block1.linear.out_features

        v1 = obs.new_zeros(B, H)
        v2 = obs.new_zeros(B, H)
        vout = obs.new_zeros(B, 1)

        spike_energy = 0.0
        v_traj = []

        # RESTORED: Standard Input Handling (No forced rate scaling on direct input)
        if self.poisson_encode:
            spk_in = poisson_encode(obs, self.T, self.rate_scale)
        else:
            spk_in = obs.unsqueeze(0).repeat(self.T, 1, 1)

        for t in range(self.T):
            i1 = self.block1.linear(spk_in[t])
            s1, v1 = self.block1.lif(i1, v1)

            i2 = self.block2.linear(s1)
            s2, v2 = self.block2.lif(i2, v2)

            iout = self.block_out.linear(s2)
            sout, vout = self.block_out.lif(iout, vout)

            v_traj.append(vout.unsqueeze(0))
            spike_energy += s1.mean() + s2.mean() + sout.mean()

        self._last_reg = spike_energy / self.T

        v_traj = torch.cat(v_traj, dim=0)  # [T,B,1]

        # Calculate Timing (Tau)
        tau = self._soft_first_spike_time(v_traj)
        self._last_latency = float(tau.mean().item())

        # Map Tau to Value
        v = self._map_tau_to_value(tau)

        if self.use_hard_no_spike:
            no_spike = (v_traj.squeeze(-1) > self.V_th).any(dim=0) == 0
            v[no_spike] = self.Rmin

        # Value Mapping & Clamping
        v = v.clamp(self.Rmin, self.Rmax).unsqueeze(-1)
        v = self.value_mapper(v)
        v = v.clamp(self.Rmin, self.Rmax)
        
        return v, tau

    # ------------------------------------------------------------------
    # Stats & Metrics (Adopted from Root for completeness)
    # ------------------------------------------------------------------

    def regulariser(self):
        return self._last_reg
    
    def last_latency(self) -> float:
        return float(self._last_latency)

    def reset_stats(self):
        self.block1.reset_stats()
        self.block2.reset_stats()
        self.block_out.reset_stats()





































































