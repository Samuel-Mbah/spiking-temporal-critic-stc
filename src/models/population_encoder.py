"""Population encoder for POPSan: Gaussian receptive-field coding of continuous observations.

Reference: Tang et al. (2021) "Deep Reinforcement Learning with Population-Coded
Spiking Neural Network for Continuous Control", NeurIPS Workshop.
"""
import torch
import torch.nn as nn


class PopulationEncoder(nn.Module):
    """Encodes each observation dimension into N neurons via Gaussian tuning curves.

    Neuron j for dimension d responds to input x with:
        g(x, μ_j) = exp(-(x - μ_j)^2 / (2σ^2))

    where μ_j are evenly spaced across [obs_low, obs_high] and σ is set to
    sigma_scale times the spacing between adjacent means.

    Output: [B, obs_dim * n_neurons_per_dim] with values in (0, 1].
    """

    def __init__(
        self,
        obs_dim: int,
        n_neurons_per_dim: int = 10,
        obs_low: float = -1.0,
        obs_high: float = 1.0,
        sigma_scale: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_neurons = n_neurons_per_dim
        self.encoded_dim = obs_dim * n_neurons_per_dim

        if n_neurons_per_dim == 1:
            mu = torch.tensor([(obs_low + obs_high) / 2.0])
            sigma_val = abs(obs_high - obs_low) + 1e-6
        else:
            mu = torch.linspace(obs_low, obs_high, n_neurons_per_dim)
            sigma_val = sigma_scale * (obs_high - obs_low) / (n_neurons_per_dim - 1)

        sigma = torch.full((n_neurons_per_dim,), max(sigma_val, 1e-6))

        # [1, 1, N] — broadcast over [B, obs_dim, 1]
        self.register_buffer("mu",    mu.view(1, 1, n_neurons_per_dim))
        self.register_buffer("sigma", sigma.view(1, 1, n_neurons_per_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, obs_dim]
        Returns:
            [B, obs_dim * n_neurons_per_dim]
        """
        x = obs.unsqueeze(-1)                                          # [B, obs_dim, 1]
        encoded = torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) # [B, obs_dim, N]
        return encoded.reshape(obs.size(0), -1)                        # [B, obs_dim * N]
