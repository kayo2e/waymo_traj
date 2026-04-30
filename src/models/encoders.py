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
    Joint Transformer 인코더.

    모든 토큰(에고 시계열 + 주변 에이전트 + 맵 폴리라인 + 신호등)을
    하나의 시퀀스로 합쳐 Self-Attention을 수행합니다.
    에이전트가 각 차선 토큰을 직접 어텐션할 수 있어 커브/교차로 예측이 개선됩니다.

    입력:
      ego_hist      [B, T,  6]        에고 과거 시계열
      social_agents [B, N,  T,  6]    주변 에이전트별 시계열
      map_scene     [B, 50, 10, 3]    차선별 10개 폴리라인 포인트
      traf          [B, 6,  1]        신호등 상태

    출력: global_feat [B, D]  (에고 현재 프레임 토큰)
    """

    def __init__(self, agent_dim=6, map_dim=3, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        D = d_model

        # ── 각 모달리티 → D차원 토큰 ────────────────────────────────────────────
        self.ego_proj   = nn.Linear(agent_dim, D)             # [B, T, D]
        self.social_enc = JointPolylineEncoder(agent_dim, D)  # [B, N, D]
        self.map_enc    = JointPolylineEncoder(map_dim, D)    # [B, 50, D]
        self.traf_proj  = nn.Linear(1, D)                     # [B, 6, D]

        # ── Positional encoding: (x, y) → D ────────────────────────────────
        self.pos_enc = nn.Linear(2, D)

        # ── Joint Self-Attention ─────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=n_heads,
            dim_feedforward=D * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.T_hist = 11   # ego 시계열 길이 (현재 프레임 인덱스 = T-1 = 10)

    def forward(self, ego_hist, social_agents, map_scene, traf):
        # ── 각 모달리티 인코딩 ───────────────────────────────────────────────────
        e = self.ego_proj(ego_hist)         # [B, T, D]
        s = self.social_enc(social_agents)  # [B, N, D]
        m = self.map_enc(map_scene)         # [B, 50, D]
        t = self.traf_proj(traf)            # [B, 6, D]

        # ── Positional encoding 주입 ─────────────────────────────────────────
        # 에고: 각 프레임의 (x, y)
        e = e + self.pos_enc(ego_hist[:, :, 0:2])

        # 주변 에이전트: 마지막 프레임 (x, y)
        s = s + self.pos_enc(social_agents[:, :, -1, 0:2])

        # 맵 폴리라인: 10개 포인트의 중심 (x, y)
        m = m + self.pos_enc(map_scene[:, :, :, 0:2].mean(dim=2))

        # 신호등: 위치 정보 없으므로 생략

        # ── 전체 토큰 concat: [B, T+N+50+6, D] = [B, 98, D] ─────────────────
        tokens = torch.cat([e, s, m, t], dim=1)

        # ── Joint Self-Attention ─────────────────────────────────────────────
        out = self.transformer(tokens)    # [B, 98, D]

        # ── 에고 현재 프레임 토큰 추출 (인덱스 T-1 = 10) ────────────────────────
        return out[:, self.T_hist - 1, :]  # [B, D]
