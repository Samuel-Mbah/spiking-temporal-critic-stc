"""
Defines ANN backbone architectures.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(m: nn.Module) -> None:
    """Apply orthogonal initialisation to Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class BackboneNetwork(nn.Module):
    """
    Standard ANN MLP backbone (ReLU).
    Forces 2 hidden layers by default for SNN compatibility.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Union[int, List[int]],
        out_features: int,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()

        # FIX: Ensure we have exactly 2 hidden layers if a single dim is provided
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims, hidden_dims]
        elif isinstance(hidden_dims, list) and len(hidden_dims) == 1:
            # If passed as [64], expand to [64, 64]
            hidden_dims = [hidden_dims[0], hidden_dims[0]]

        dims = [in_features, *hidden_dims, out_features]

        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.apply(orthogonal_init)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_activations: bool = False,
        threshold: float = 0.1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        activations: Optional[Dict[str, torch.Tensor]] = (
            {} if return_activations else None
        )

        for idx, layer in enumerate(self.layers):
            x = layer(x)

            is_last = idx == len(self.layers) - 1
            if not is_last:
                x = F.relu(x)

                if activations is not None:
                    activations[f"layer_{idx}"] = (
                        (x > threshold).sum(dim=-1).detach()
                    )

                x = self.dropout(x)

        if activations is not None:
            activations["output"] = (x > threshold).sum(dim=-1).detach()
            return x, activations

        return x


# -----------------------------------------------------------------
# Actor & Critic Wrappers (Unchanged but included for completeness)
# -----------------------------------------------------------------

class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        latent_dim: int,
        action_dim: int,
        *,
        critic_informs_actor: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.critic_informs_actor = critic_informs_actor

        input_dim = latent_dim + (1 if critic_informs_actor else 0)
        self.policy_head = nn.Linear(input_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        *,
        critic_value: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        z = self.backbone(state, return_activations=return_activations)
        
        activations = None
        if return_activations and isinstance(z, tuple):
            z, activations = z

        if self.critic_informs_actor:
            if critic_value is None:
                raise ValueError("critic_value must be provided when critic_informs_actor=True")
            z = torch.cat([z, critic_value.unsqueeze(-1)], dim=-1)

        logits = self.policy_head(z)
        
        if return_activations and activations is not None:
            return logits, activations
        
        return logits


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(latent_dim, 1)

    def forward(self, state: torch.Tensor, *, return_activations: bool = False):
        z = self.backbone(state, return_activations=return_activations)
        
        activations = None
        if return_activations and isinstance(z, tuple):
            z, activations = z
            
        value = self.value_head(z).squeeze(-1)
        
        if return_activations and activations is not None:
            return value, activations
            
        return value