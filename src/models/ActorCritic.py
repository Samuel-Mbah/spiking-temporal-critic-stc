from typing import Generic, TypeVar, Optional, Tuple
import torch
import torch.nn as nn

A = TypeVar("A", bound=nn.Module)  # Actor type
C = TypeVar("C", bound=nn.Module)  # Critic type

class ActorCritic(nn.Module, Generic[A, C]):
    """
    Actor-Critic wrapper with explicit control over policy and value evaluation.

    Supports:
        - decoupled actor-critic
        - critic-informed actors
        - ANN / SNN hybrid extensions
    """

    def __init__(
        self,
        actor: A,
        critic: C,
        *,
        critic_informs_actor: bool = False,
        detach_critic_for_actor: bool = True,
        normalize_critic_for_actor: bool = True,
        critic_actor_value_clip: Optional[float] = 5.0,
        critic_actor_norm_momentum: float = 0.01,
        mode: Optional[str] = None,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.critic_informs_actor = critic_informs_actor
        self.detach_critic_for_actor = detach_critic_for_actor
        self.normalize_critic_for_actor = normalize_critic_for_actor
        self.critic_actor_value_clip = critic_actor_value_clip
        self.critic_actor_norm_momentum = critic_actor_norm_momentum
        self.mode = mode or ""
        self.register_buffer("_critic_actor_norm_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("_critic_actor_norm_var", torch.tensor(1.0), persistent=False)

    def _prepare_critic_value_for_actor(self, critic_value: torch.Tensor) -> torch.Tensor:
        """Prepares critic value used as actor input without altering value-learning target."""
        value_for_actor = critic_value

        if self.detach_critic_for_actor:
            value_for_actor = value_for_actor.detach()

        if self.normalize_critic_for_actor:
            with torch.no_grad():
                batch_mean = value_for_actor.mean()
                batch_var = value_for_actor.var(unbiased=False)
                m = float(self.critic_actor_norm_momentum)
                self._critic_actor_norm_mean.mul_(1.0 - m).add_(m * batch_mean)
                self._critic_actor_norm_var.mul_(1.0 - m).add_(m * batch_var)

            denom = torch.sqrt(self._critic_actor_norm_var + 1e-6)
            value_for_actor = (value_for_actor - self._critic_actor_norm_mean) / denom

        if self.critic_actor_value_clip is not None:
            c = float(self.critic_actor_value_clip)
            value_for_actor = torch.clamp(value_for_actor, -c, c)

        return value_for_actor

    # ---- Explicit sub-forwards ---------------------------------

    def critic_forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute value estimate."""
        return self.critic(state)

    def actor_forward(
        self,
        state: torch.Tensor,
        *,
        critic_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute policy logits."""
        if self.critic_informs_actor:
            if critic_value is None:
                raise ValueError(
                    "critic_value must be provided when critic_informs_actor=True"
                )
            return self.actor(state, critic_value=critic_value, **kwargs)

        return self.actor(state, **kwargs)

    # ---- Main forward ------------------------------------------

    def forward(
        self,
        state: torch.Tensor,
        *,
        critic_value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard PPO-style forward pass.

        Returns:
            logits: policy output
            value: critic value estimate
        """
        value_raw = critic_value
        if value_raw is None:
            value_raw = self.critic_forward(state)

        value_for_actor = value_raw
        if self.critic_informs_actor:
            value_for_actor = self._prepare_critic_value_for_actor(value_raw)

        logits = self.actor_forward(state, critic_value=value_for_actor, **kwargs)

        return logits, value_raw
    
    # ---- Spike stats passthrough -------------------------------
    
    def get_spike_stats(self):
        """Pass-through to the actor."""
        if hasattr(self.actor, "get_spike_stats"):
            return self.actor.get_spike_stats()
        return {"total_spikes": 0, "sparsity": 0.0}
