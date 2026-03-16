"""Recurrent ANN backbone using LSTM for partial-observability environments."""
import torch
import torch.nn as nn

class RecurrentBackbone(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)

    def forward(self, x, hx=None):
        # x shape: [batch, seq_len, features]
        # For PPO rollout, seq_len is 1. For updates, it is your BPTT length.
        out, hx_next = self.lstm(x, hx)
        return out, hx_next

class RecurrentActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, act_dim):
        super().__init__()
        self.backbone = RecurrentBackbone(in_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, x, hx=None):
        # Ensure input has sequence dimension: [batch, 1, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        latent, hx_next = self.backbone(x, hx)
        logits = self.action_head(latent.squeeze(1))
        return logits, hx_next