"""Agent factory: constructs actor-critic pairs for ANN, SNN, and hybrid architectures.

Exposes ``make_agent()``, ``build_actor()``, and ``build_critic()`` with type-safe
selection via the ``ActorType`` and ``CriticType`` enumerations.
"""
import torch
import torch.nn as nn
from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional

from src.models.ann import BackboneNetwork, Actor, Critic, orthogonal_init
from src.models.snn_spike_actor import SNNSpikeActor
from src.models.snn_spike_value_critic import SNNSpikeValueCritic
from src.models.snn_timing_critic import SNNTimingCritic
from src.models.popsan_actor import POPSanActor
from src.models.actor_critic import ActorCritic

class ActorType(Enum):
    ANN = auto()
    SNN_SPIKE = auto()
    SNN_POP = auto()
    # ANN_RECURRENT = auto()

class CriticType(Enum):
    ANN = auto()
    SNN_SPIKE = auto()
    SNN_TIMING = auto()


def resolve_actor_critic_params(
    *,
    T: int,
    V_th: float,
    actor_T: Optional[int] = None,
    critic_T: Optional[int] = None,
    poisson_encode: bool = False,
    actor_poisson_encode: Optional[bool] = None,
    critic_poisson_encode: Optional[bool] = None,
    rate_scale: float = 1.0,
    actor_rate_scale: Optional[float] = None,
    critic_rate_scale: Optional[float] = None,
    actor_V_th: Optional[float] = None,
    critic_V_th: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Resolves hierarchical configuration parameters for SNN components.
    Prioritizes specific (actor_*) params over global defaults.
    """
    return {
        "actor_T": actor_T or T,
        "critic_T": critic_T or T,
        "actor_poisson": actor_poisson_encode if actor_poisson_encode is not None else poisson_encode,
        "critic_poisson": critic_poisson_encode if critic_poisson_encode is not None else poisson_encode,
        "actor_rate": actor_rate_scale if actor_rate_scale is not None else rate_scale,
        "critic_rate": critic_rate_scale if critic_rate_scale is not None else rate_scale,
        "actor_V": actor_V_th if actor_V_th is not None else V_th,
        "critic_V": critic_V_th if critic_V_th is not None else V_th,
    }

def build_actor(
    actor_type: ActorType,
    *,
    in_dim: int,
    act_dim: int,
    hidden_dim: int,
    params: Dict[str, Any],
    beta: float,
    logit_temp: float = 1.0,
    center_logits: bool = True,
    critic_informs_actor: bool = False,
    actor_surrogate_slope: float = 25.0,
    actor_surrogate_type: str = "fast_sigmoid",
    actor_cosh_alpha: float = 10.0,
    actor_cosh_beta: float = 1.0,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Constructs the Actor network based on the specified type."""
    # Dispatch between the available actor implementations.
    if actor_type is ActorType.ANN:
        backbone = BackboneNetwork(
            in_features=in_dim,
            hidden_dims=hidden_dim,
            out_features=hidden_dim,
            dropout=dropout,
        )
        return Actor(
            backbone=backbone,
            latent_dim=hidden_dim,
            action_dim=act_dim,
            critic_informs_actor=critic_informs_actor
        )

    if actor_type is ActorType.SNN_SPIKE:
        # SNN actor handles the optional critic concatenation internally.
        effective_in_dim = in_dim + (1 if critic_informs_actor else 0)

        return SNNSpikeActor(
            in_dim=effective_in_dim,
            hid_dim=hidden_dim,
            out_dim=act_dim,
            beta=beta,
            V_th=params["actor_V"],
            T=params["actor_T"],
            poisson_encode=params["actor_poisson"],
            rate_scale=params["actor_rate"],
            logit_temp=logit_temp,
            center_logits=center_logits,
            actor_surrogate_slope=actor_surrogate_slope,
            actor_surrogate_type=actor_surrogate_type,
            actor_cosh_alpha=actor_cosh_alpha,
            actor_cosh_beta=actor_cosh_beta,
        )

    if actor_type is ActorType.SNN_POP:
        # Population encoder replaces Poisson/rate coding.
        # critic_informs_actor is handled inside POPSanActor: it adds +1 to the
        # SNN's input dim to make room for the value concatenated in forward_T.
        return POPSanActor(
            obs_dim=in_dim,
            act_dim=act_dim,
            hid_dim=hidden_dim,
            n_neurons_per_dim=int(kwargs.get("popsan_n_neurons", 10)),
            obs_low=float(kwargs.get("popsan_obs_low", -1.0)),
            obs_high=float(kwargs.get("popsan_obs_high", 1.0)),
            sigma_scale=float(kwargs.get("popsan_sigma_scale", 1.0)),
            critic_informs_actor=critic_informs_actor,
            beta=beta,
            V_th=params["actor_V"],
            T=params["actor_T"],
            poisson_encode=False,  # population encoding replaces Poisson
            rate_scale=params["actor_rate"],
            logit_temp=logit_temp,
            center_logits=center_logits,
            actor_surrogate_slope=actor_surrogate_slope,
            actor_surrogate_type=actor_surrogate_type,
            actor_cosh_alpha=actor_cosh_alpha,
            actor_cosh_beta=actor_cosh_beta,
        )

    raise ValueError(f"Unsupported actor type: {actor_type}")

def build_critic(
    critic_type: CriticType,
    *,
    in_dim: int,
    hidden_dim: int,
    params: Dict[str, Any],
    beta: float,
    gamma: float,
    Rmax: float,
    Rmin: float,
    critic_spike_temp: float = 25.0,
    critic_surrogate_type: str = "cosh",
    critic_surrogate_slope: float = 25.0,
    critic_cosh_alpha: float = 10.0,
    critic_cosh_beta: float = 1.0,
    critic_use_hard_no_spike: bool = False,
    critic_scale_by_discount: bool = False,
    critic_value_affine: bool = False,
    critic_lambda_tau: float = 0.0,
    dropout: float = 0.0,
    **kwargs,
) -> nn.Module:
    """Constructs the Critic network based on the specified type."""
    # Choose the matching critic implementation for the requested type.
    if critic_type is CriticType.ANN:
        backbone = BackboneNetwork(
            in_features=in_dim,
            hidden_dims=hidden_dim,
            out_features=hidden_dim,
            dropout=dropout,
        )
        return Critic(backbone=backbone, latent_dim=hidden_dim)

    if critic_type is CriticType.SNN_SPIKE:
        return SNNSpikeValueCritic(
            in_dim=in_dim,
            hid_dim=hidden_dim,
            beta=beta,
            V_th=params["critic_V"],
            T=params["critic_T"],
            poisson_encode=params["critic_poisson"],
            rate_scale=params["critic_rate"],
            critic_surrogate_type=critic_surrogate_type,
            critic_surrogate_slope=critic_surrogate_slope,
            critic_cosh_alpha=critic_cosh_alpha,
            critic_cosh_beta=critic_cosh_beta,
        )

    if critic_type is CriticType.SNN_TIMING:
        return SNNTimingCritic(
            in_dim=in_dim,
            hid_dim=hidden_dim,
            beta=beta,
            V_th=params["critic_V"],
            critic_T=params["critic_T"],
            gamma=gamma,
            Rmax=Rmax,
            Rmin=Rmin,
            spike_temp=critic_spike_temp,
            poisson_encode=params["critic_poisson"],
            rate_scale=params["critic_rate"],
            critic_surrogate_type=critic_surrogate_type,
            critic_surrogate_slope=critic_surrogate_slope,
            cosh_alpha=critic_cosh_alpha,
            cosh_beta=critic_cosh_beta,
            use_hard_no_spike=critic_use_hard_no_spike,
            scale_by_discount=critic_scale_by_discount,
            value_affine=critic_value_affine,
            lambda_tau=critic_lambda_tau,
        )

    raise ValueError(f"Unsupported critic type: {critic_type}")

def make_agent(
    *,
    actor_type: ActorType,
    critic_type: CriticType,
    hidden_dim: int = 64,
    in_dim: int = 4,
    act_dim: int = 2,
    T: int = 32,
    beta: float = 0.95,
    V_th: float = 1.0,
    gamma: float = 0.99,
    Rmax: float = 500.0,
    Rmin: float = -500.0,
    **kwargs,
) -> ActorCritic:
    """
    Main entry point to assemble an Actor-Critic agent.
    Applies orthogonal initialization to all created modules.
    After actor + critic are built, the helper applies a shared initialization wrapper.
    """
    params = resolve_actor_critic_params(T=T, V_th=V_th, **kwargs)

    actor = build_actor(
        actor_type, in_dim=in_dim, act_dim=act_dim, hidden_dim=hidden_dim,
        params=params, beta=beta, **kwargs
    )

    critic = build_critic(
        critic_type, in_dim=in_dim, hidden_dim=hidden_dim,
        params=params, beta=beta, gamma=gamma, Rmax=Rmax, Rmin=Rmin, **kwargs
    )

    # Apply Standard PPO Initialization
    actor.apply(orthogonal_init)
    critic.apply(orthogonal_init)

    # Break the zero-value fixed point for ANN critics on sparse-reward tasks.
    # orthogonal_init zeros all biases; on sparse-reward envs the critic can
    # trivially converge to predicting 0 for everything and never escape.
    # A small positive bias on the value head gives the optimizer a gradient
    # to climb before any positive rewards are seen.
    critic_value_init_bias = float(kwargs.get("critic_value_init_bias", 0.0))
    if critic_value_init_bias != 0.0 and hasattr(critic, "value_head"):
        nn.init.constant_(critic.value_head.bias, critic_value_init_bias)

    return ActorCritic(
        actor=actor,
        critic=critic,
        critic_informs_actor=kwargs.get("critic_informs_actor", False),
        detach_critic_for_actor=kwargs.get("detach_critic_for_actor", True),
        normalize_critic_for_actor=kwargs.get("normalize_critic_for_actor", True),
        critic_actor_value_clip=kwargs.get("critic_actor_value_clip", 5.0),
        critic_actor_norm_momentum=kwargs.get("critic_actor_norm_momentum", 0.01),
    )

def resolve_cartpole_types(mode: str) -> Tuple[ActorType, CriticType]:
    """Maps config string 'mode' to internal Enum types."""
    mode = mode.lower().strip()
    if mode == "ann":
        return ActorType.ANN, CriticType.ANN
    # if mode == "ann_recurrent":
    #     return ActorType.ANN_RECURRENT, CriticType.ANN
    if mode == "snn_actor_ann_critic":
        return ActorType.SNN_SPIKE, CriticType.ANN
    if mode == "snn_actor_snn_critic":
        return ActorType.SNN_SPIKE, CriticType.SNN_SPIKE
    if mode == "snn_actor_snn_timing_critic":
        return ActorType.SNN_SPIKE, CriticType.SNN_TIMING
    if mode in ("popsan", "popsan_ann_critic"):
        return ActorType.SNN_POP, CriticType.ANN
    if mode == "popsan_timing_critic":
        return ActorType.SNN_POP, CriticType.SNN_TIMING

    # Legacy fallback
    if mode == "snn":
        return ActorType.SNN_SPIKE, CriticType.ANN
        
    raise ValueError(f"Unknown agent mode: {mode}")
