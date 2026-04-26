"""POPSan actor: population-coded SNN actor for RL.

Prepends a PopulationEncoder to an SNNSpikeActor and exposes the same
interface so the surrogate trainer requires no changes.

Reference: Tang et al. (2021) "Deep Reinforcement Learning with Population-Coded
Spiking Neural Network for Continuous Control", NeurIPS Workshop.
"""
import torch
import torch.nn as nn
from typing import Optional

from src.models.population_encoder import PopulationEncoder
from src.models.snn_spike_actor import SNNSpikeActor


class POPSanActor(nn.Module):
    """Population-coded SNN actor.

    Pipeline: raw obs → PopulationEncoder → encoded spike rates → SNNSpikeActor.

    The encoded dimension is obs_dim * n_neurons_per_dim.  When
    critic_informs_actor=True, the critic value is appended to the encoded
    observation (inside SNNSpikeActor.forward_T), so the SNN first-layer
    expects encoded_dim + 1 inputs — this is handled here at construction time.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hid_dim: int,
        *,
        n_neurons_per_dim: int = 10,
        obs_low: float = -1.0,
        obs_high: float = 1.0,
        sigma_scale: float = 1.0,
        critic_informs_actor: bool = False,
        **snn_kwargs,
    ):
        super().__init__()
        self.encoder = PopulationEncoder(
            obs_dim=obs_dim,
            n_neurons_per_dim=n_neurons_per_dim,
            obs_low=obs_low,
            obs_high=obs_high,
            sigma_scale=sigma_scale,
        )
        encoded_dim = obs_dim * n_neurons_per_dim
        # +1 makes room for the critic value concatenated inside SNNSpikeActor.forward_T
        snn_in_dim = encoded_dim + (1 if critic_informs_actor else 0)

        self.snn = SNNSpikeActor(
            in_dim=snn_in_dim,
            hid_dim=hid_dim,
            out_dim=act_dim,
            **snn_kwargs,
        )
        self.T = self.snn.T

    # ------------------------------------------------------------------
    # Forward interface — mirrors SNNSpikeActor exactly
    # ------------------------------------------------------------------

    def forward_T(
        self,
        x: torch.Tensor,
        critic_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.snn.forward_T(self.encoder(x), critic_value=critic_value, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.snn.forward(self.encoder(x), **kwargs)

    # ------------------------------------------------------------------
    # Stats & metrics — delegate to inner SNN
    # ------------------------------------------------------------------

    def regulariser(self):
        return self.snn.regulariser()

    def last_spike_count(self):
        return self.snn.last_spike_count()

    def last_spike_count_total(self):
        return self.snn.last_spike_count_total()

    def last_latency(self):
        return self.snn.last_latency()

    def get_spike_stats(self) -> dict:
        stats = self.snn.get_spike_stats()
        stats["encoded_dim"] = self.encoder.encoded_dim
        return stats

    def reset_stats(self):
        if hasattr(self.snn, "reset_stats"):
            self.snn.reset_stats()
