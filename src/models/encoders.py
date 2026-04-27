"""Shared encoder building blocks."""

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel


# ── 헬퍼: MambaConfig 생성 ─────────────────────────────────────────────────────
def _mamba_cfg(d_model, n_layers):
    return MambaConfig(
        d_model=d_model, n_layers=n_layers,
        expand=2, d_conv=4, d_state=16,
        bos_token_id=0, eos_token_id=0, hidden_size=d_model,
    )


class TambaMambaEncoder(nn.Module):
    """Wraps HuggingFace MambaModel for sequence-to-sequence encoding."""

    def __init__(self, d_model=128, n_layers=2):
        super().__init__()
        self.mamba = MambaModel(_mamba_cfg(d_model, n_layers))

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


class MultiStreamMambaEncoder(nn.Module):
    """
    3개 독립 Mamba 스트림 → GlobalMamba 통합.

    TemporalMamba : 에고 시계열          [B, T,  agent_dim] → [B, D]
    TrafficMamba  : 주변 에이전트 상호작용 [B, N,  agent_dim] → [B, D]
    SceneMamba    : 맵 + 신호 토큰       [B, L,  map_dim]   → [B, D]
    GlobalMamba   : 3 스트림 통합        [B, 3,  D]         → [B, D]

    Returns global_feat [B, D]
    """

    def __init__(self, agent_dim=6, map_dim=3, d_model=128, n_layers=2):
        super().__init__()
        D = d_model
        self.agent_proj = nn.Linear(agent_dim, D)
        self.map_proj   = nn.Linear(map_dim,   D)

        self.temporal_mamba = TambaMambaEncoder(D, n_layers)
        self.traffic_mamba  = TambaMambaEncoder(D, n_layers)
        self.scene_mamba    = TambaMambaEncoder(D, n_layers)
        self.global_mamba   = TambaMambaEncoder(D, n_layers)

    def forward(self, ego_hist, social_agents, map_tokens):
        """
        ego_hist      : [B, T,  agent_dim]
        social_agents : [B, N,  agent_dim]
        map_tokens    : [B, L,  map_dim]
        """
        e = self.agent_proj(ego_hist)       # [B, T, D]
        s = self.agent_proj(social_agents)  # [B, N, D]
        m = self.map_proj(map_tokens)       # [B, L, D]

        e_feat = self.temporal_mamba(e).mean(dim=1)  # [B, D]
        s_feat = self.traffic_mamba(s).mean(dim=1)   # [B, D]
        m_feat = self.scene_mamba(m).mean(dim=1)     # [B, D]

        combined  = torch.stack([e_feat, s_feat, m_feat], dim=1)  # [B, 3, D]
        global_out = self.global_mamba(combined)                    # [B, 3, D]
        return global_out.mean(dim=1)                               # [B, D]
