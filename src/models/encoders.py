"""Shared encoder building blocks."""

import torch
import torch.nn as nn


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


class SocialTemporalEncoder(nn.Module):
    """Per-agent GRU temporal encoder: [B, N, T, F] → [B, N, D]"""
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.gru  = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        B, N, T, F = x.shape
        h = self.proj(x.reshape(B * N, T, F))
        _, h_n = self.gru(h)
        return h_n.squeeze(0).reshape(B, N, -1)


class MultiStreamMambaEncoder(nn.Module):
    """
    Joint Transformer 인코더.

    모든 토큰(위험 prefix + 에고 시계열 + 주변 에이전트 + 맵 폴리라인 + 신호등)을
    하나의 시퀀스로 합쳐 Self-Attention을 수행합니다.

    use_risk_prefix=True (기본):
      [risk(1) | ego(11) | social(31) | map(50) | traf(6)] = 99토큰
      risk_label [B,3] → Linear(3,D) → risk_token [B,1,D]로 prepend
    use_risk_prefix=False (ablation):
      [ego(11) | social(31) | map(50) | traf(6)] = 98토큰

    입력:
      ego_hist      [B, T,  10]       에고 과거 시계열
      social_agents [B, N,  T,  10]   주변 에이전트별 시계열
      map_scene     [B, 50, 10, 6]    차선별 10개 폴리라인 포인트
      traf          [B, 6,  1]        신호등 상태
      risk_label    [B, 3]            위험 레이블 (use_risk_prefix=True 시)

    출력: (global_feat [B, D], lane_feat [B, 50, D])
    """

    def __init__(self, agent_dim=10, map_dim=6, d_model=128, n_layers=2, n_heads=4,
                 use_risk_prefix=True, use_traj_fix=True, cond_dim=3,
                 use_social_temporal=False):
        super().__init__()
        D = d_model
        self.use_risk_prefix = use_risk_prefix
        self.use_traj_fix    = use_traj_fix

        # ── 각 모달리티 → D차원 토큰 ────────────────────────────────────────────
        self.ego_proj   = nn.Linear(agent_dim, D)
        self.social_enc = (SocialTemporalEncoder(agent_dim, D)
                           if use_social_temporal
                           else JointPolylineEncoder(agent_dim, D))
        self.map_enc    = JointPolylineEncoder(map_dim, D)
        self.traf_proj  = nn.Linear(1, D)

        # ── Condition prefix token 투영 ──────────────────────────────────────
        if use_risk_prefix:
            self.risk_proj = nn.Linear(cond_dim, D)

        # ── Positional encoding: (x, y) → D ────────────────────────────────
        self.pos_enc = nn.Linear(2, D)

        self.T_hist = 11   # ego 시계열 길이 (현재 프레임 인덱스 = T-1 = 10)

        # ── Temporal positional encoding (use_traj_fix=True 시만) ────────────
        if use_traj_fix:
            self.time_emb = nn.Embedding(self.T_hist, D)

        # ── Joint Self-Attention ─────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads,
            dim_feedforward=D * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, ego_hist, social_agents, map_scene, traf, risk_label=None):
        e = self.ego_proj(ego_hist)
        s = self.social_enc(social_agents)
        m = self.map_enc(map_scene)
        t = self.traf_proj(traf)

        e = e + self.pos_enc(ego_hist[:, :, 0:2])
        s = s + self.pos_enc(social_agents[:, :, -1, 0:2])
        m = m + self.pos_enc(map_scene[:, :, :, 0:2].mean(dim=2))

        if self.use_traj_fix:
            t_idx = torch.arange(self.T_hist, device=ego_hist.device)
            e = e + self.time_emb(t_idx).unsqueeze(0)

        if self.use_risk_prefix and risk_label is not None:
            risk_tok = self.risk_proj(risk_label.float()).unsqueeze(1)
            tokens   = torch.cat([risk_tok, e, s, m, t], dim=1)
            offset   = 1
        else:
            tokens = torch.cat([e, s, m, t], dim=1)
            offset = 0

        out = self.transformer(tokens)

        if self.use_traj_fix:
            global_feat = out[:, offset + self.T_hist - 1, :]
        else:
            global_feat = out[:, offset : offset + self.T_hist, :].mean(dim=1)

        lane_start = offset + self.T_hist + s.shape[1]
        lane_feat  = out[:, lane_start : lane_start + 50, :]

        return global_feat, lane_feat
