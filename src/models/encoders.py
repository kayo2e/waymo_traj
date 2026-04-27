"""Shared encoder building blocks."""

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel


class TambaMambaEncoder(nn.Module):
    """Wraps HuggingFace MambaModel for sequence-to-sequence encoding."""

    def __init__(self, d_model=128, n_layers=2):
        super().__init__()
        cfg = MambaConfig(
            d_model=d_model,
            n_layers=n_layers,
            expand=2,
            d_conv=4,
            d_state=16,
            bos_token_id=0,
            eos_token_id=0,
            hidden_size=d_model,
        )
        self.mamba = MambaModel(cfg)

    def forward(self, x):
        return self.mamba(inputs_embeds=x).last_hidden_state


class JointPolylineEncoder(nn.Module):
    """
    Encodes a batch of polylines via MLP + max-pool.

    3-D input  [B, N, F]        → output [B, N, D]
    4-D input  [B, N, pts, F]   → max-pool over pts → output [B, N, D]
    """

    def __init__(self, input_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        feat = self.net(x)
        if feat.dim() == 4:
            feat = feat.max(dim=2).values
        return feat
