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

    TemporalMamba : 에고 시계열           [B, T,  6]        → [B, D]
    TrafficMamba  : 주변 에이전트 (시계열) [B, N,  T, 6]    → [B, N, D] → [B, D]
    SceneMamba    : 맵 폴리라인 + 신호    [B, 50, 10, 3]
                                          + [B, 6, 1]       → [B, D]
    GlobalMamba   : 3 스트림 통합         [B, 3,  D]        → [B, D]

    JointPolylineEncoder(PointNet-style)로 폴리라인 형상 정보를 보존합니다.
    Returns global_feat [B, D]
    """

    def __init__(self, agent_dim=6, map_dim=3, d_model=128, n_layers=2):
        super().__init__()
        D = d_model
        # ego: 시계열 Linear 투영
        self.ego_proj   = nn.Linear(agent_dim, D)
        # social: per-agent 시계열을 PointNet 방식으로 인코딩 [B, N, T, 6] → [B, N, D]
        self.social_enc = JointPolylineEncoder(agent_dim, D)
        # map: per-lane 폴리라인 인코딩 [B, 50, 10, 3] → [B, 50, D]
        self.map_enc    = JointPolylineEncoder(map_dim, D)
        # traffic signal: scalar → D
        self.traf_proj  = nn.Linear(1, D)

        self.temporal_mamba = TambaMambaEncoder(D, n_layers)
        self.traffic_mamba  = TambaMambaEncoder(D, n_layers)
        self.scene_mamba    = TambaMambaEncoder(D, n_layers)
        self.global_mamba   = TambaMambaEncoder(D, n_layers)

    def forward(self, ego_hist, social_agents, map_scene, traf):
        """
        ego_hist      : [B, T,  6]        에고 과거 시계열
        social_agents : [B, N,  T,  6]    주변 에이전트별 시계열 (PointNet 인코딩)
        map_scene     : [B, 50, 10, 3]    차선별 10개 폴리라인 포인트
        traf          : [B, 6,  1]        신호등 상태
        """
        # ── ego 시계열 스트림 ───────────────────────────────────────────────────
        e = self.ego_proj(ego_hist)                      # [B, T, D]
        e_feat = self.temporal_mamba(e).mean(dim=1)      # [B, D]

        # ── social 에이전트 스트림: 시계열 shape 보존 → PointNet max-pool ────────
        s = self.social_enc(social_agents)               # [B, N, D]
        s_feat = self.traffic_mamba(s).mean(dim=1)       # [B, D]

        # ── 맵 스트림: 폴리라인 형상 보존 → PointNet max-pool ───────────────────
        m = self.map_enc(map_scene)                      # [B, 50, D]
        t = self.traf_proj(traf)                         # [B, 6,  D]
        mt = torch.cat([m, t], dim=1)                    # [B, 56, D]
        m_feat = self.scene_mamba(mt).mean(dim=1)        # [B, D]

        # ── GlobalMamba 통합 ───────────────────────────────────────────────────
        combined   = torch.stack([e_feat, s_feat, m_feat], dim=1)  # [B, 3, D]
        global_out = self.global_mamba(combined)                     # [B, 3, D]
        return global_out.mean(dim=1)                                # [B, D]
