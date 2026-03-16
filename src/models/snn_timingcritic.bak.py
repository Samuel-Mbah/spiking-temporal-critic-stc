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











































































# import torch
# import torch.nn as nn
# import snntorch as snn

# from src.models.snn_block import SNNBlock
# from src.models.surrogates import cosh_surrogate
# from src.models.snn_utils import poisson_encode, init_lif_states

# class SNNTimingCritic(nn.Module):
#     """
#     Timing-based SNN critic.
#     Value is encoded by FIRST spike timing of output neuron.
#     """

#     def __init__(
#         self,
#         in_dim: int,
#         hid_dim: int,
#         *,
#         beta: float = 0.95,
#         V_th: float = 1.0,
#         critic_T: int = 8,
#         Rmax: float = 500.0,
#         Rmin: float = 0.0,
#         gamma: float = 0.99,
#         spike_temp: float = 25.0,
#         poisson_encode: bool = True,
#         rate_scale: float = 1.0,
#         cosh_alpha: float = 10.0,
#         cosh_beta: float = 1.0,
#         use_hard_no_spike: bool = False,
#         scale_by_discount: bool = False,
#         value_affine: bool = False,
#     ):
#         super().__init__()

#         self.T = int(critic_T)
#         self.V_th = float(V_th)
#         self.Rmax = float(Rmax)
#         self.Rmin = float(Rmin)
#         self.gamma = float(gamma)
#         self.spike_temp = float(spike_temp)
#         self.poisson_encode = poisson_encode
#         self.rate_scale = float(rate_scale)
#         self.use_hard_no_spike = bool(use_hard_no_spike)
#         self.scale_by_discount = bool(scale_by_discount)

#         sg = cosh_surrogate(alpha=cosh_alpha, beta=cosh_beta)

#         # --- Network ---
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
        
#         # Optional value scaling
#         if value_affine:
#             self.value_mapper = nn.Linear(1, 1)
#             nn.init.ones_(self.value_mapper.weight)
#             nn.init.zeros_(self.value_mapper.bias)
#         else:
#             self.value_mapper = nn.Identity()

#         self._last_reg = torch.tensor(0.0)
#         self._last_vout = None

#     # ------------------------------------------------------------------
#     # Utilities
#     # ------------------------------------------------------------------

#     def _soft_first_spike_time(self, v_traj: torch.Tensor) -> torch.Tensor:
#         """Differentiable expected first spike time."""
#         T, B, _ = v_traj.shape

#         p = torch.sigmoid(self.spike_temp * (v_traj - self.V_th)).squeeze(-1)
#         p = p.clamp(1e-6, 1.0 - 1e-6)

#         no_prev = torch.cat(
#             [torch.ones(1, B, device=p.device), torch.cumprod(1 - p, dim=0)[:-1]],
#             dim=0,
#         )

#         q = no_prev * p
#         q = q / q.sum(dim=0, keepdim=True)

#         t_idx = torch.arange(T, device=p.device).float().unsqueeze(1)
#         return (q * t_idx).sum(dim=0)

#     def _map_tau_to_value(self, tau: torch.Tensor) -> torch.Tensor:
#         """Map timing to scalar value."""
#         N = float(self.T)
#         half = N / 2.0

#         scale = (2.0 / N)
#         pos_scale = scale * self.Rmax
#         neg_scale = scale * self.Rmin

#         if self.scale_by_discount:
#             denom = max(1e-6, 1.0 - self.gamma)
#             pos_scale /= denom
#             neg_scale /= denom

#         v = torch.full_like(tau, self.Rmin)
#         pos = tau < half

#         v[pos] = (half - tau[pos]) * pos_scale
#         v[~pos] = (tau[~pos] - half) * neg_scale

#         return v

#     # ------------------------------------------------------------------
#     # Forward
#     # ------------------------------------------------------------------

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         B = obs.size(0)
#         H = self.block1.linear.out_features

#         v1 = obs.new_zeros(B, H)
#         v2 = obs.new_zeros(B, H)
#         vout = obs.new_zeros(B, 1)

#         spike_energy = 0.0
#         v_traj = []

#         if self.poisson_encode:
#             spk_in = poisson_encode(obs, self.T, self.rate_scale)
#         else:
#             spk_in = obs.unsqueeze(0).repeat(self.T, 1, 1)

#         for t in range(self.T):
#             i1 = self.block1.linear(spk_in[t])
#             s1, v1 = self.block1.lif(i1, v1)

#             i2 = self.block2.linear(s1)
#             s2, v2 = self.block2.lif(i2, v2)

#             iout = self.block_out.linear(s2)
#             sout, vout = self.block_out.lif(iout, vout)

#             v_traj.append(vout.unsqueeze(0))
#             spike_energy += s1.mean() + s2.mean() + sout.mean()

#         self._last_reg = spike_energy / self.T

#         v_traj = torch.cat(v_traj, dim=0)  # [T,B,1]
#         self._last_vout = v_traj.detach()

#         tau = self._soft_first_spike_time(v_traj)
#         v = self._map_tau_to_value(tau)

#         if self.use_hard_no_spike:
#             no_spike = (v_traj.squeeze(-1) > self.V_th).any(dim=0) == 0
#             v[no_spike] = self.Rmin

#         v = v.clamp(self.Rmin, self.Rmax).unsqueeze(-1)
#         v = self.value_mapper(v)
#         return v.clamp(self.Rmin, self.Rmax)
    
    
    
#     def forward_detailed(self, obs: torch.Tensor):
#         """
#         Returns both the Value and the First-Spike-Time (tau).
#         Used for research/diagnostic plotting.
#         """
#         B = obs.size(0)
        
#         # 1. Run the SNN Forward Pass (Same as forward)
#         # Note: You could refactor this into a _forward_internal to avoid code duplication,
#         # but copying the body of forward() here is fine for safety.
#         H = self.block1.linear.out_features
#         v1 = obs.new_zeros(B, H)
#         v2 = obs.new_zeros(B, H)
#         vout = obs.new_zeros(B, 1)

#         v_traj = []

#         if self.poisson_encode:
#             spk_in = poisson_encode(obs, self.T, self.rate_scale)
#         else:
#             spk_in = obs.unsqueeze(0).repeat(self.T, 1, 1)

#         for t in range(self.T):
#             i1 = self.block1.linear(spk_in[t])
#             s1, v1 = self.block1.lif(i1, v1)
#             i2 = self.block2.linear(s1)
#             s2, v2 = self.block2.lif(i2, v2)
#             iout = self.block_out.linear(s2)
#             sout, vout = self.block_out.lif(iout, vout)
#             v_traj.append(vout.unsqueeze(0))

#         v_traj = torch.cat(v_traj, dim=0)

#         # 2. Calculate Timing (Tau) and Value (V)
#         tau = self._soft_first_spike_time(v_traj)
#         v = self._map_tau_to_value(tau)

#         # Apply value mapping/clamping
#         v = v.clamp(self.Rmin, self.Rmax).unsqueeze(-1)
#         v = self.value_mapper(v)
#         v = v.clamp(self.Rmin, self.Rmax)

#         # 3. Return Tuple
#         return v, tau

#     # ------------------------------------------------------------------

#     def regulariser(self):
#         return self._last_reg




























# import torch
# import torch.nn as nn
# import snntorch as snn

# from src.models.snn_block import SNNBlock
# from src.models.surrogates import cosh_surrogate
# from src.models.snn_utils import poisson_encode, init_lif_states

# class SNNTimingCritic(nn.Module):
#     """
#     SNN Critic where Value is encoded by the First-Spike Time (TTFS).
    
#     Earlier spike = Higher Value (typically).
#     Includes logic to map [0, T] -> [Rmax, Rmin].
#     """
#     def __init__(
#         self,
#         in_dim: int,
#         hid_dim: int,
#         *,
#         beta: float = 0.95,
#         V_th: float = 1.0,
#         critic_T: int = 32,
#         Rmax: float = 100.0,
#         Rmin: float = 0.0,
#         gamma: float = 0.99,
#         spike_temp: float = 25.0,
#         poisson_encode: bool = False,
#         rate_scale: float = 1.0,
#         cosh_alpha: float = 10.0,
#         cosh_beta: float = 1.0,
#         use_hard_no_spike: bool = True,
#         scale_by_discount: bool = False,
#         value_affine: bool = False,
#     ):
#         super().__init__()
#         self.T = int(critic_T)
#         self.V_th = V_th
#         self.Rmax = Rmax
#         self.Rmin = Rmin
#         self.spike_temp = spike_temp
#         self.poisson_encode = poisson_encode
#         self.rate_scale = rate_scale
#         self.use_hard_no_spike = use_hard_no_spike

#         # Use Cosh surrogate for better temporal gradients
#         sg = cosh_surrogate(alpha=cosh_alpha, beta=cosh_beta)

#         self.block1 = SNNBlock(nn.Linear(in_dim, hid_dim), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))
#         self.block2 = SNNBlock(nn.Linear(hid_dim, hid_dim), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))
#         self.block_out = SNNBlock(nn.Linear(hid_dim, 1), snn.Leaky(beta=beta, threshold=V_th, spike_grad=sg))
        
#         # --- FIX 1: Positive Bias Initialization ---
#         # Initialize biases to be > V_th. 
#         # This ensures that 0-input (Perfect State) causes immediate spiking (High Value).
#         # Bad states (large inputs) will learn to INHIBIT this activity.
#         init_bias = self.V_th * 1.05
#         nn.init.constant_(self.block1.linear.bias, init_bias)
#         nn.init.constant_(self.block2.linear.bias, init_bias)
        
        
#         nn.init.constant_(self.block_out.linear.bias, 0.0)
        
#         # Initialize weights to be small to let bias dominate initially
#         nn.init.orthogonal_(self.block1.linear.weight, gain=0.5)
#         nn.init.orthogonal_(self.block2.linear.weight, gain=0.5)
#         nn.init.orthogonal_(self.block_out.linear.weight, gain=0.5)
        
#         if value_affine:
#             self.value_mapper = nn.Linear(1, 1)
#         else:
#             self.value_mapper = nn.Identity()
#         self._last_latency = 0.0

#     def _soft_first_spike_time(self, v_traj: torch.Tensor) -> torch.Tensor:
#         """
#         Differentiable calculation of first spike time using Softmax over time.
#         Args:
#             v_traj: Membrane potentials over time [T, Batch, 1]
#         Returns:
#             tau: Expected first spike time [Batch]
#         """
#         T, B, _ = v_traj.shape
        
#         # Probability of spiking at each step (Sigmoid of membrane potential)
#         p = torch.sigmoid(self.spike_temp * (v_traj - self.V_th)).squeeze(-1) # [T, B]
#         p = p.clamp(1e-6, 1.0 - 1e-6)

#         # Probability of NOT spiking before t: cumprod(1-p)
#         no_spike_before = torch.cat([torch.ones(1, B, device=p.device), torch.cumprod(1 - p, dim=0)[:-1]], dim=0)
        
#         # Probability density of first spike at t
#         pdf = no_spike_before * p
        
#         # Normalize PDF
#         pdf = pdf / (pdf.sum(dim=0, keepdim=True) + 1e-8)

#         # Expected time
#         t_indices = torch.arange(T, device=p.device, dtype=torch.float).unsqueeze(1) # [T, 1]
#         tau = (pdf * t_indices).sum(dim=0) # [B]
        
#         return tau

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """Standard forward returning Value."""
#         return self.forward_detailed(obs)[0]

#     def forward_detailed(self, obs: torch.Tensor):
#         B = obs.size(0)
        
#         if self.poisson_encode:
#             spk_in = poisson_encode(obs, self.T, self.rate_scale)
#         else:
#             # APPLY SCALING TO DIRECT INPUTS
#             # This allows the network to see "strong" continuous values
#             # without losing sign information or silence at 0.
#             scaled_obs = obs * self.rate_scale
#             spk_in = scaled_obs.unsqueeze(0).repeat(self.T, 1, 1)

#         v1, v2, vout = init_lif_states(obs, 
#             self.block1.linear.out_features,
#             self.block2.linear.out_features,
#             self.block_out.linear.out_features
#         )

#         v_traj = []

#         # Run SNN
#         for t in range(self.T):
#             spk1, v1, _ = self.block1.forward_step(spk_in[t], v1)
#             spk2, v2, _ = self.block2.forward_step(spk1, v2)
#             _, vout, _ = self.block_out.forward_step(spk2, vout)
            
#             v_traj.append(vout.unsqueeze(0))

#         v_traj = torch.cat(v_traj, dim=0) # [T, B, 1]

#         # Calculate Tau (Timing)
#         tau = self._soft_first_spike_time(v_traj)
#         self._last_latency = float(tau.mean().item())

#         # Map Tau to Value
#         # Tau=0 -> Rmax, Tau=T -> Rmin
#         # Linear interpolation
#         alpha = tau / float(self.T)
#         val = (1.0 - alpha) * self.Rmax + alpha * self.Rmin
        
#         # Hard No-Spike Penalty
#         if self.use_hard_no_spike:
#             # Check if membrane never crossed threshold
#             max_v = v_traj.max(dim=0).values.squeeze(-1)
#             no_spike_mask = max_v < self.V_th
#             val[no_spike_mask] = self.Rmin

#         val = val.unsqueeze(-1) # [B, 1]
#         val = self.value_mapper(val)
        
#         return val, tau

#     def last_latency(self) -> float:
#         return float(self._last_latency)
    
#     def reset_stats(self):
#         self.block1.reset_stats()
#         self.block2.reset_stats()
#         self.block_out.reset_stats()
