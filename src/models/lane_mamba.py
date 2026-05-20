"""Risk-aware Lane Mamba: 위험 조건부 차선 안전도 순차 추정."""

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel


def _mamba_cfg(d_model: int, n_layers: int) -> MambaConfig:
    return MambaConfig(
        d_model=d_model, n_layers=n_layers,
        expand=2, d_conv=4, d_state=16,
        bos_token_id=0, eos_token_id=0, hidden_size=d_model,
    )


class RiskAwareLaneMamba(nn.Module):
    """
    위험 이벤트(급정지/근접/차선변경) 조건 하에
    어느 차선이 안전한지를 Mamba로 순차 추정.

    차선을 ego 기준 lateral y 로 정렬(좌 → 현재 → 우) 후
    HuggingFace MambaModel로 스캔.

    Inputs
      risk_label : [B, risk_dim]    이진 위험 레이블
      lane_feat  : [B, L, D]        Transformer 어텐션 후 차선 토큰
      map_scene  : [B, L, 10, 3]   ego-relative 차선 폴리라인 (정렬 기준)

    Outputs
      lane_context : [B, D]   safety-weighted 차선 컨텍스트 → global_feat에 더해짐
      lane_prob    : [B, L]   각 차선 softmax 안전 확률 (로깅/시각화용)
    """

    def __init__(self, d_model: int = 128, risk_dim: int = 3, n_layers: int = 1):
        super().__init__()
        D = d_model
        self.risk_proj  = nn.Sequential(nn.Linear(risk_dim, D), nn.ReLU())
        self.input_proj = nn.Linear(D * 2, D)
        self.mamba      = MambaModel(_mamba_cfg(D, n_layers))
        self.lane_head  = nn.Linear(D, 1)

    @staticmethod
    def _sort_lanes(lane_feat: torch.Tensor, map_scene: torch.Tensor):
        """lateral y 기준 좌(+y)→우(-y) 정렬."""
        lane_y   = map_scene[:, :, :, 1].mean(dim=2)          # [B, L]
        sort_idx = lane_y.argsort(dim=1, descending=True)
        B, L, D  = lane_feat.shape
        sorted_feat = lane_feat.gather(
            1, sort_idx.unsqueeze(-1).expand(B, L, D)
        )
        return sorted_feat, sort_idx

    def forward(
        self,
        risk_label : torch.Tensor,   # [B, risk_dim]
        lane_feat  : torch.Tensor,   # [B, L, D]
        map_scene  : torch.Tensor,   # [B, L, 10, 3]
    ):
        B, L, D = lane_feat.shape

        # 1. lateral 정렬
        lane_sorted, _ = self._sort_lanes(lane_feat, map_scene)

        # 2. risk 임베딩 broadcast + concat → D
        risk_emb = self.risk_proj(risk_label)                    # [B, D]
        risk_exp = risk_emb.unsqueeze(1).expand(B, L, D)        # [B, L, D]
        x = self.input_proj(
            torch.cat([risk_exp, lane_sorted], dim=-1)           # [B, L, 2D]
        )                                                         # [B, L, D]

        # 3. Mamba 순차 스캔 (좌→우 차선)
        x = self.mamba(inputs_embeds=x).last_hidden_state        # [B, L, D]

        # 4. softmax 가중합 → lane_context
        lane_logits = self.lane_head(x).squeeze(-1)              # [B, L]
        lane_prob   = lane_logits.softmax(dim=-1)                # [B, L]
        lane_context = (lane_prob.unsqueeze(-1) * lane_sorted).sum(dim=1)  # [B, D]

        return lane_context, lane_prob
