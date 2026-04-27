"""Risk conditioning module."""

import torch
import torch.nn as nn


class RiskFusion(nn.Module):
    """
    위험 레이블을 d_model 차원 임베딩으로 변환한 뒤 global_feat와 concat.

    Input
      global_feat : [B, D]
      risk_label  : [B, 3]  float (0/1 이진 또는 sigmoid 확률)

    Output
      fused : [B, D*2]  (global_feat ‖ risk_emb)
    """

    def __init__(self, risk_dim: int = 3, d_model: int = 128):
        super().__init__()
        self.risk_embed = nn.Sequential(
            nn.Linear(risk_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, global_feat: torch.Tensor,
                risk_label: torch.Tensor) -> torch.Tensor:
        risk_emb = self.risk_embed(risk_label.float())          # [B, D]
        return torch.cat([global_feat, risk_emb], dim=-1)       # [B, D*2]
