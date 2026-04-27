"""
위험 조건부 궤적 예측 모델.

RiskConditionedModel (메인):
  MultiStreamMambaEncoder (Temporal / Traffic / Scene / Global)
  + RiskFusion (위험 레이블 조건화)
  + K=6 KeypointDecoder + TrajectoryRefiner

WaymoMotionModel (하위 호환용):
  단일 Mamba + K=6, 위험 레이블 없음
"""

import torch
import torch.nn as nn

from src.models.encoders import (
    TambaMambaEncoder, JointPolylineEncoder, MultiStreamMambaEncoder,
)
from src.models.risk_fusion import RiskFusion


class WaymoMotionModel(nn.Module):
    """
    Full k-gen pipeline:
      1. Encode agents / map / traffic → context tokens [B, N_tok, D]
      2. Mamba global interaction
      3. K mode queries × ego token → K keypoint sets  [B, K, 3, 2]
      4. K mode queries × ego token + keypoints → K trajectories  [B, K, 80, 2]
    """

    def __init__(self, d_model=128, K=6):
        super().__init__()
        self.d_model = d_model
        self.K = K

        # ── Encoders ──────────────────────────────────────────────────────────
        self.agent_encoder   = JointPolylineEncoder(input_dim=6,  d_model=d_model)
        self.map_encoder     = JointPolylineEncoder(input_dim=3,  d_model=d_model)
        self.traffic_encoder = nn.Linear(1, d_model)

        # ── Global context ────────────────────────────────────────────────────
        self.mamba = TambaMambaEncoder(d_model=d_model, n_layers=2)

        # ── K mode queries (learned per-mode diversity embeddings) ────────────
        self.mode_queries = nn.Embedding(K, d_model)

        # ── Stage 3-A: KeypointDecoder  input: [ego_ctx | mode_emb] ─────────
        self.kp_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3 * 2),      # 3 keypoints × (x, y)
        )

        # ── Stage 3-B: TrajectoryRefiner  input: [ego_ctx | mode_emb | kp] ──
        self.refiner = nn.Sequential(
            nn.Linear(d_model * 2 + 3 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 80 * 2),
        )

    def forward(self, agents, scene, traffic):
        """
        agents  : [B, N_agents, T_hist, 6]
        scene   : [B, N_map,   10,     3]
        traffic : [B, N_traf,          1]

        Returns dict:
          keypoints  : [B, K, 3,  2]
          trajectory : [B, K, 80, 2]
          context    : [B, N_tok, D]
        """
        B = agents.shape[0]

        a_emb = self.agent_encoder(agents)    # [B, N_agents, D]
        m_emb = self.map_encoder(scene)       # [B, N_map,    D]
        t_emb = self.traffic_encoder(traffic) # [B, N_traf,   D]

        tokens  = torch.cat([a_emb, m_emb, t_emb], dim=1)  # [B, N_tok, D]
        context = self.mamba(tokens)                         # [B, N_tok, D]

        ego_ctx = context[:, 0, :]                           # [B, D]

        # Expand ego_ctx and mode queries to [B, K, D]
        mode_idx = torch.arange(self.K, device=agents.device)
        mode_emb = self.mode_queries(mode_idx)               # [K, D]
        ego_exp  = ego_ctx.unsqueeze(1).expand(B, self.K, -1)   # [B, K, D]
        mode_exp = mode_emb.unsqueeze(0).expand(B, -1, -1)      # [B, K, D]

        combined = torch.cat([ego_exp, mode_exp], dim=-1)    # [B, K, D*2]

        # K keypoint sets
        kp = self.kp_head(combined).reshape(B, self.K, 3, 2)    # [B, K, 3, 2]

        # K dense trajectories
        kp_flat    = kp.reshape(B, self.K, -1)                   # [B, K, 6]
        refiner_in = torch.cat([combined, kp_flat], dim=-1)      # [B, K, D*2+6]
        traj = self.refiner(refiner_in).reshape(B, self.K, 80, 2)  # [B, K, 80, 2]

        return {"keypoints": kp, "trajectory": traj, "context": context}


class RiskConditionedModel(nn.Module):
    """
    위험 조건부 K=6 궤적 예측 모델.

    forward 입력:
      ego_hist      : [B, T,  6]   에고 과거 궤적
      social_agents : [B, N,  6]   주변 에이전트 (T 차원 평균 등)
      map_tokens    : [B, L,  3]   맵 폴리라인 + 신호 (L = N_MAP + N_TRAF)
      risk_label    : [B, 3] float — 학습 시 GT 이진 레이블,
                                      추론 시 None → 자체 예측값 사용

    forward 출력:
      keypoints   : [B, K, 3,  2]  1s/3s/5s 웨이포인트
      trajectory  : [B, K, 80, 2]  8초 전체 궤적
      risk_logits : [B, 3]         위험 분류 로짓 (sigmoid → 확률)
    """

    def __init__(self, agent_dim: int = 6, map_dim: int = 3,
                 d_model: int = 128, K: int = 6, n_layers: int = 2):
        super().__init__()
        self.K = K
        D = d_model

        # ── 인코더 ─────────────────────────────────────────────────────────────
        self.encoder   = MultiStreamMambaEncoder(agent_dim, map_dim, D, n_layers)

        # ── 위험 분류 헤드 (global_feat → logits) ─────────────────────────────
        self.risk_head = nn.Linear(D, 3)

        # ── 위험 조건화 Fusion ─────────────────────────────────────────────────
        self.fusion = RiskFusion(risk_dim=3, d_model=D)  # output: [B, D*2]

        # ── K mode queries [B, K, D] ───────────────────────────────────────────
        self.mode_queries = nn.Embedding(K, D)

        # decoder input dim = D*2 (fused) + D (mode) = D*3
        dec_dim = D * 3

        # ── 키포인트 디코더 ────────────────────────────────────────────────────
        self.kp_head = nn.Sequential(
            nn.Linear(dec_dim, D),
            nn.ReLU(),
            nn.Linear(D, 3 * 2),         # 3 keypoints × 2D
        )

        # ── 궤적 리파이너 ──────────────────────────────────────────────────────
        self.refiner = nn.Sequential(
            nn.Linear(dec_dim + 3 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 80 * 2),
        )

    def forward(self, ego_hist, social_agents, map_tokens, risk_label=None):
        B = ego_hist.shape[0]

        # ── 1. 씬 인코딩 ───────────────────────────────────────────────────────
        global_feat = self.encoder(ego_hist, social_agents, map_tokens)  # [B, D]

        # ── 2. 위험 분류 ───────────────────────────────────────────────────────
        risk_logits = self.risk_head(global_feat)                         # [B, 3]

        # ── 3. 위험 조건화 ─────────────────────────────────────────────────────
        if risk_label is None:
            # 추론 시: 자체 예측 확률 사용 (detach → decoder gradient 분리)
            risk_input = torch.sigmoid(risk_logits).detach()
        else:
            risk_input = risk_label.float()

        fused = self.fusion(global_feat, risk_input)  # [B, D*2]

        # ── 4. K 모드 디코딩 ───────────────────────────────────────────────────
        mode_idx = torch.arange(self.K, device=ego_hist.device)
        mode_emb = self.mode_queries(mode_idx)                           # [K, D]

        fused_exp = fused.unsqueeze(1).expand(B, self.K, -1)            # [B, K, D*2]
        mode_exp  = mode_emb.unsqueeze(0).expand(B, -1, -1)             # [B, K, D]
        combined  = torch.cat([fused_exp, mode_exp], dim=-1)            # [B, K, D*3]

        kp      = self.kp_head(combined).reshape(B, self.K, 3, 2)       # [B, K, 3, 2]
        kp_flat = kp.reshape(B, self.K, -1)                             # [B, K, 6]
        traj    = self.refiner(
            torch.cat([combined, kp_flat], dim=-1)
        ).reshape(B, self.K, 80, 2)                                      # [B, K, 80, 2]

        return {
            "keypoints":   kp,           # [B, K, 3, 2]
            "trajectory":  traj,         # [B, K, 80, 2]
            "risk_logits": risk_logits,  # [B, 3]
        }
