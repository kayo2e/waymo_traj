"""위험 조건부 궤적 예측 모델 (RiskConditionedModel)."""

import torch
import torch.nn as nn

from src.models.encoders    import MultiStreamMambaEncoder
from src.models.risk_fusion import RiskFusion


class RiskConditionedModel(nn.Module):
    """
    K=6 궤적 예측 모델.

    use_ar=False (기본, Non-AR MLP 디코더):
      context + mode → lane cross-attn → MLP → [B, K, 80, 2] 한 번에 예측
      teacher forcing 불필요, exposure bias 없음

    use_ar=True (AR LSTM 디코더, ablation용):
      80 timestep autoregressive decoding

    forward 입력:
      ego_hist      : [B, T,  10]
      social_agents : [B, N,  T,  10]
      map_scene     : [B, 50, 10, 6]
      traf          : [B, 6,  1]
      risk_label    : [B, cond_dim]  조건 레이블 (GT 사용)
      gt_traj       : [B, 80, 2]    AR 모드 teacher-forcing 용 (non-AR 시 무시)
      tf_prob       : float          AR 모드 scheduled sampling (non-AR 시 무시)

    forward 출력:
      trajectory   : [B, K, 80, 2]
      risk_logits  : [B, cond_dim] or None
      lane_prob    : [B, 50] or None
    """

    def __init__(self, agent_dim: int = 10, map_dim: int = 6,
                 d_model: int = 128, K: int = 6, n_layers: int = 2,
                 ar_hidden: int = 256,
                 use_lane_mamba: bool = True,
                 use_risk_prefix: bool = True,
                 use_traj_fix: bool = True,
                 use_map_per_step: bool = False,
                 use_ar: bool = False,
                 cond_dim: int = 3):
        super().__init__()
        self.K                = K
        self.use_lane_mamba   = use_lane_mamba
        self.use_risk_prefix  = use_risk_prefix
        self.use_traj_fix     = use_traj_fix
        self.use_map_per_step = use_map_per_step
        self.use_ar           = use_ar
        self.cond_dim         = cond_dim
        D = d_model
        H = ar_hidden

        # ── 인코더 → (global_feat [B,D], lane_feat [B,50,D]) ──────────────────
        self.encoder = MultiStreamMambaEncoder(
            agent_dim, map_dim, D, n_layers,
            use_risk_prefix=use_risk_prefix,
            use_traj_fix=use_traj_fix,
            cond_dim=cond_dim,
        )

        if use_risk_prefix:
            self.context_proj = nn.Linear(D, D * 2)
        else:
            self.risk_head = nn.Linear(D, cond_dim)
            self.fusion    = RiskFusion(risk_dim=cond_dim, d_model=D)

        # ── vel_proj (use_traj_fix=True 시만): vx,vy,cos_h,sin_h → context ────
        if use_traj_fix:
            self.vel_proj = nn.Linear(4, D * 2)

        # ── Risk-aware Lane Mamba (optional) ──────────────────────────────────
        if use_lane_mamba:
            from src.models.lane_mamba import RiskAwareLaneMamba
            self.lane_mamba = RiskAwareLaneMamba(d_model=D, risk_dim=cond_dim, n_layers=1)

        # ── K mode queries ────────────────────────────────────────────────────
        self.mode_queries = nn.Embedding(K, D)

        if use_ar:
            # ── AR LSTM 디코더 ─────────────────────────────────────────────────
            self.init_proj = nn.Linear(D * 2 + D, H * 2)
            if use_map_per_step:
                self.pos_proj  = nn.Linear(2, D)
                self.map_attn  = nn.MultiheadAttention(D, num_heads=4,
                                                       batch_first=True, dropout=0.0)
                lstm_in = 2 + D * 2 + D
            else:
                lstm_in = 2 + D * 2
            self.lstm_cell = nn.LSTMCell(input_size=lstm_in, hidden_size=H)
            self.out_proj  = nn.Linear(H, 2)
        else:
            # ── Non-AR MLP 디코더 ──────────────────────────────────────────────
            # context[D*2] + mode[D] → query[D] → cross-attn(lane) → MLP → [80*2]
            self.lane_query_proj = nn.Linear(D * 2 + D, D)
            self.lane_attn       = nn.MultiheadAttention(D, num_heads=4,
                                                         batch_first=True, dropout=0.0)
            self.traj_mlp = nn.Sequential(
                nn.Linear(D * 2, H),
                nn.ReLU(),
                nn.Linear(H, H),
                nn.ReLU(),
                nn.Linear(H, 80 * 2),
            )

    def _mlp_decode(self, context, mode_emb, lane_feat, T=80):
        """
        Non-AR MLP 디코더.

        context   : [B, D*2]
        mode_emb  : [K, D]
        lane_feat : [B, 50, D]
        반환      : [B, K, T, 2]  ego-relative 절대 좌표
        """
        B = context.shape[0]
        K = self.K
        L = lane_feat.shape[1]

        ctx_bk  = context.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)        # [B*K, D*2]
        mode_bk = mode_emb.unsqueeze(0).expand(B, K, -1).reshape(B*K, -1)       # [B*K, D]
        lane_bk = lane_feat.unsqueeze(1).expand(B, K, L, -1).reshape(B*K, L, -1)  # [B*K, 50, D]

        # context + mode → query, 차선 cross-attention
        query = self.lane_query_proj(torch.cat([ctx_bk, mode_bk], dim=-1))       # [B*K, D]
        lane_ctx, _ = self.lane_attn(query.unsqueeze(1), lane_bk, lane_bk)       # [B*K, 1, D]
        lane_ctx = lane_ctx.squeeze(1)                                            # [B*K, D]

        # MLP → 80 timestep 한 번에 예측 (절대 좌표)
        feat = torch.cat([query, lane_ctx], dim=-1)                               # [B*K, D*2]
        traj = self.traj_mlp(feat).reshape(B*K, T, 2)                            # [B*K, T, 2]
        return traj.reshape(B, K, T, 2)

    def _ar_decode(self, context, mode_emb, lane_feat, T=80, gt_traj=None, tf_prob=1.0):
        """
        context   : [B, D*2]
        mode_emb  : [K, D]
        lane_feat : [B, 50, D]   transformer-encoded lane tokens
        반환      : [B, K, T, 2]  절대 좌표 (ego-relative)

        매 스텝:
          vel    = cur_pos - prev_pos          (속도 벡터)
          map_c  = cross-attn(cur_pos → lane)  (공간 맥락)
          LSTM( [vel | context | map_c] )
        """
        B = context.shape[0]
        K = self.K
        L = lane_feat.shape[1]

        ctx_bk  = context.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)
        mode_bk = mode_emb.unsqueeze(0).expand(B, K, -1).reshape(B*K, -1)
        if self.use_map_per_step:
            lane_bk = lane_feat.unsqueeze(1).expand(B, K, L, -1).reshape(B*K, L, -1)

        hc0  = self.init_proj(torch.cat([ctx_bk, mode_bk], dim=-1))
        H    = hc0.shape[-1] // 2
        h, c = hc0[:, :H], hc0[:, H:]

        prev_pos  = torch.zeros(B*K, 2, device=context.device)
        cur_pos   = torch.zeros(B*K, 2, device=context.device)
        positions = []

        for t in range(T):
            if self.use_map_per_step:
                # 속도 벡터 + 매 스텝 map cross-attention (386dim)
                vel      = cur_pos - prev_pos
                pos_q    = self.pos_proj(cur_pos).unsqueeze(1)
                map_c, _ = self.map_attn(pos_q, lane_bk, lane_bk)
                map_c    = map_c.squeeze(1)
                lstm_in  = torch.cat([vel, ctx_bk, map_c], dim=-1)
            else:
                # 이전/중간 아키텍처: [prev_xy | context] (258dim)
                lstm_in = torch.cat([prev_pos, ctx_bk], dim=-1)

            # LSTM
            h, c  = self.lstm_cell(lstm_in, (h, c))
            delta = self.out_proj(h)                             # [B*K, 2]

            next_pos = cur_pos + delta
            positions.append(next_pos)

            # 4. prev/cur 업데이트 (teacher forcing)
            prev_pos = cur_pos.detach()
            if gt_traj is not None and tf_prob > 0.0:
                gt_bk = gt_traj[:, t, :].unsqueeze(1).expand(B, K, 2).reshape(B*K, 2)
                if tf_prob >= 1.0:
                    cur_pos = gt_bk
                else:
                    mask    = (torch.rand(B*K, 1, device=context.device) < tf_prob)
                    cur_pos = torch.where(mask, gt_bk, next_pos.detach())
            else:
                cur_pos = next_pos.detach()

        return torch.stack(positions, dim=1).reshape(B, K, T, 2)

    def forward(self, ego_hist, social_agents, map_scene, traf,
                risk_label=None, gt_traj=None, tf_prob=1.0):

        # 1. 씬 인코딩 (use_risk_prefix=True이면 risk prefix token 포함)
        global_feat, lane_feat = self.encoder(
            ego_hist, social_agents, map_scene, traf,
            risk_label=risk_label,
        )

        # 2. risk conditioning 방식 분기
        if self.use_risk_prefix:
            risk_logits = None
            risk_input = risk_label.float() if risk_label is not None \
                         else torch.zeros(global_feat.shape[0], self.cond_dim,
                                          device=global_feat.device)
        else:
            risk_logits = self.risk_head(global_feat)
            risk_input  = risk_label.float() if risk_label is not None \
                          else torch.sigmoid(risk_logits).detach()

        # 3. Risk-aware Lane Mamba
        lane_prob = None
        if self.use_lane_mamba:
            lane_context, lane_prob = self.lane_mamba(risk_input, lane_feat, map_scene)
            enhanced_feat = global_feat + lane_context                      # [B, D]
        else:
            enhanced_feat = global_feat

        # 4. context [B, D*2] 생성
        if self.use_risk_prefix:
            context = self.context_proj(enhanced_feat)                      # [B, D*2]
        else:
            context = self.fusion(enhanced_feat, risk_input)                # [B, D*2]

        # vel_proj (use_traj_fix=True 시만)
        if self.use_traj_fix:
            cur_kin = ego_hist[:, -1, 2:6]                                  # [B, 4]
            context = context + self.vel_proj(cur_kin)                      # [B, D*2]

        # 5. 디코딩
        mode_idx = torch.arange(self.K, device=ego_hist.device)
        mode_emb = self.mode_queries(mode_idx)                              # [K, D]
        if self.use_ar:
            traj = self._ar_decode(context, mode_emb, lane_feat, T=80,
                                   gt_traj=gt_traj, tf_prob=tf_prob)       # [B, K, 80, 2]
        else:
            traj = self._mlp_decode(context, mode_emb, lane_feat, T=80)   # [B, K, 80, 2]

        return {
            "trajectory":  traj,          # [B, K, 80, 2]
            "risk_logits": risk_logits,   # [B, 3] or None
            "lane_prob":   lane_prob,     # [B, 50] or None
        }
