"""위험 조건부 궤적 예측 모델 (RiskConditionedModel)."""

import torch
import torch.nn as nn

from src.models.encoders    import MultiStreamMambaEncoder, LaneGraphEncoder
from src.models.risk_fusion import RiskFusion


class GoalHead(nn.Module):
    """Context → K goal positions [B, K, 2]."""
    def __init__(self, context_dim, K):
        super().__init__()
        self.K = K
        self.head = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, K * 2),
        )

    def forward(self, context):
        return self.head(context).reshape(context.shape[0], self.K, 2)


class LaneGoalHead(nn.Module):
    """
    Lane-anchored goal via cross-attention.

    K mode queries cross-attend to lane tokens → K goal positions.
    각 mode가 독립적으로 관련 차선을 선택 → fully differentiable.

    별도 lane_scorer가 [B, K, 50] logits를 출력해 auxiliary supervision 제공:
    winner mode는 GT endpoint에 가장 가까운 차선에 높은 점수를 줘야 함.

    context (maneuver-conditioned global_feat)를 query에 더해
    Turn/LC 같은 방향 전환 시 올바른 목적지 차선을 선택하도록 안내한다.

    Input : lane_feat   [B, 50, D]
            context     [B, context_dim]  (optional, maneuver info 포함)
    Output: goals       [B, K, 2]   ego-relative goal positions
            lane_logits [B, K, 50]  per-mode lane scoring logits (for loss)
    """
    def __init__(self, d_model, K, n_heads=4, context_dim=None, use_turn_emb=False):
        super().__init__()
        self.K = K
        self.mode_queries = nn.Embedding(K, d_model)
        self.cross_attn   = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1)
        self.out_proj     = nn.Linear(d_model, 2)
        self.lane_scorer  = nn.Linear(d_model, K)   # [B,50,D] → [B,50,K] → [B,K,50]
        self.cond_proj = nn.Linear(context_dim, d_model) if context_dim is not None else None
        # 4-class: 0=none, 1=left-turn, 2=right-turn, 3=u-turn
        self.turn_emb = nn.Embedding(4, d_model) if use_turn_emb else None

    @staticmethod
    def _lane_turn_type(map_scene):
        """Classify each lane from polyline geometry: 0=none,1=left,2=right,3=u-turn."""
        dx = map_scene[:, :, :, 2]                                       # [B, 50, 10]
        dy = map_scene[:, :, :, 3]
        a0 = torch.atan2(dy[:, :, 0],  dx[:, :, 0])                     # [B, 50]
        a1 = torch.atan2(dy[:, :, -1], dx[:, :, -1])
        delta = (a1 - a0 + torch.pi) % (2 * torch.pi) - torch.pi        # [-π, π]
        t = torch.zeros(map_scene.shape[:2], dtype=torch.long,
                        device=map_scene.device)
        t[delta >  0.5] = 1   # left  (> ~29°)
        t[delta < -0.5] = 2   # right
        t[delta.abs() > 2.0] = 3   # u-turn (> ~115°)
        is_lane = map_scene[:, :, 0, 4] > 0.5                            # type_lane flag
        t = t * is_lane.long()
        return t

    def forward(self, lane_feat, context=None, map_scene=None):
        B = lane_feat.shape[0]
        q = self.mode_queries.weight.unsqueeze(0).expand(B, -1, -1)   # [B, K, D]
        if context is not None and self.cond_proj is not None:
            q = q + self.cond_proj(context).unsqueeze(1)               # [B, K, D]
        kv = lane_feat
        if map_scene is not None and self.turn_emb is not None:
            kv = lane_feat + self.turn_emb(self._lane_turn_type(map_scene))
        out, _ = self.cross_attn(q, kv, kv)                            # [B, K, D]
        goals       = self.out_proj(out)                                # [B, K, 2]
        lane_logits = self.lane_scorer(kv).permute(0, 2, 1)            # [B, K, 50]
        return goals, lane_logits


class CausalTrajDecoder(nn.Module):
    """
    GPT-style causal trajectory decoder.

    T time-step queries attend causally to each other (causal self-attn)
    and cross-attend to lane features.  Context (scene + mode) is injected
    as an additive bias broadcast across all time steps.

    Single parallel forward pass — efficient, yet temporally causal.

    Args:
        d_model     : hidden dim (must match lane_feat dim)
        context_dim : dim of the per-mode context vector fed as bias
        n_heads     : attention heads
        n_layers    : number of TransformerDecoderLayer stacks
        T           : future trajectory length (timesteps)
    """

    def __init__(self, d_model: int, context_dim: int,
                 n_heads: int = 4, n_layers: int = 2, T: int = 80):
        super().__init__()
        D = d_model
        self.T = T
        self.time_queries = nn.Embedding(T, D)
        self.ctx_proj     = nn.Linear(context_dim, D)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=n_heads, dim_feedforward=D * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(D, 2)

    def forward(self, context: torch.Tensor, lane_feat: torch.Tensor) -> torch.Tensor:
        """
        context  : [BK, context_dim]  per-mode scene+mode context
        lane_feat: [BK, L, D]         encoded lane tokens (memory)
        Returns  : [BK, T, 2]         ego-relative future positions
        """
        BK = context.shape[0]
        t_idx = torch.arange(self.T, device=context.device)
        tgt = self.time_queries(t_idx).unsqueeze(0).expand(BK, -1, -1)  # [BK, T, D]
        tgt = tgt + self.ctx_proj(context).unsqueeze(1)                  # inject context
        causal_mask = torch.triu(
            torch.full((self.T, self.T), float('-inf'), device=context.device),
            diagonal=1,
        )
        out = self.decoder(tgt, lane_feat, tgt_mask=causal_mask)         # [BK, T, D]
        return self.out_proj(out)                                         # [BK, T, 2]


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

    def __init__(self, agent_dim: int = 10, map_dim: int = 8, traf_dim: int = 1,
                 d_model: int = 128, K: int = 6, n_layers: int = 2,
                 n_heads: int = 4,
                 ar_hidden: int = 256,
                 use_lane_mamba: bool = True,
                 use_risk_prefix: bool = True,
                 use_traj_fix: bool = True,
                 use_map_per_step: bool = False,
                 use_ar: bool = False,
                 use_causal_attn: bool = False,
                 use_lane_anchor: bool = False,
                 use_goal_cond: bool = False,
                 use_lane_goal: bool = False,
                 use_cond_query: bool = False,
                 use_turn_emb: bool = False,
                 use_social_temporal: bool = False,
                 use_lane_graph: bool = False,
                 use_goal_gate: bool = False,
                 cond_dim: int = 3):
        super().__init__()
        self.K                = K
        self.use_lane_mamba   = use_lane_mamba
        self.use_risk_prefix  = use_risk_prefix
        self.use_traj_fix     = use_traj_fix
        self.use_map_per_step = use_map_per_step
        self.use_ar             = use_ar
        self.use_causal_attn    = use_causal_attn
        self.use_lane_anchor    = use_lane_anchor
        self.use_goal_cond      = use_goal_cond
        self.use_lane_goal      = use_lane_goal
        self.use_cond_query     = use_cond_query
        self.use_turn_emb       = use_turn_emb
        self.use_lane_graph     = use_lane_graph
        self.use_goal_gate      = use_goal_gate
        self.cond_dim           = cond_dim
        D = d_model
        H = ar_hidden

        # ── 인코더 → (global_feat [B,D], lane_feat [B,50,D]) ──────────────────
        self.encoder = MultiStreamMambaEncoder(
            agent_dim, map_dim, traf_dim, D, n_layers,
            n_heads=n_heads,
            use_risk_prefix=use_risk_prefix,
            use_traj_fix=use_traj_fix,
            cond_dim=cond_dim,
            use_social_temporal=use_social_temporal,
        )

        if use_lane_graph:
            self.lane_graph_enc = LaneGraphEncoder(D, n_heads=n_heads, n_layers=2)

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
        elif use_causal_attn:
            # context[D*2] + mode[D] (+ goal[D] if any goal conditioning) → context_dim
            use_any_goal   = use_goal_cond or use_lane_goal
            causal_ctx_dim = D * 4 if use_any_goal else D * 3
            self.causal_decoder = CausalTrajDecoder(
                d_model=D, context_dim=causal_ctx_dim, n_heads=n_heads, n_layers=2,
            )
            if use_goal_cond:
                self.goal_head = GoalHead(D * 2, K)
            if use_lane_goal:
                ctx_dim = D * 2 if use_cond_query else None
                self.lane_goal_head = LaneGoalHead(
                    D, K, n_heads=n_heads, context_dim=ctx_dim,
                    use_turn_emb=use_turn_emb)
            if use_goal_cond and use_lane_goal and use_goal_gate:
                self.goal_gate = nn.Linear(D * 2, 1)
            if use_any_goal:
                self.goal_proj = nn.Linear(2, D)
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

    def _mlp_decode(self, context, mode_emb, lane_feat, lane_pts=None, T=80):
        """
        Non-AR MLP 디코더.

        context   : [B, D*2]
        mode_emb  : [K, D]
        lane_feat : [B, 50, D]
        lane_pts  : [B, 50, 2]  lane endpoint xy (use_lane_anchor=True 시)
        반환      : ([B, K, T, 2], [B, K, 50])
        """
        B = context.shape[0]
        K = self.K
        L = lane_feat.shape[1]

        ctx_bk  = context.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)          # [B*K, D*2]
        mode_bk = mode_emb.unsqueeze(0).expand(B, K, -1).reshape(B*K, -1)         # [B*K, D]
        lane_bk = lane_feat.unsqueeze(1).expand(B, K, L, -1).reshape(B*K, L, -1)  # [B*K, 50, D]

        # context + mode → query, 차선 cross-attention
        query = self.lane_query_proj(torch.cat([ctx_bk, mode_bk], dim=-1))         # [B*K, D]
        lane_ctx, attn_w = self.lane_attn(query.unsqueeze(1), lane_bk, lane_bk)    # [B*K, 1, D], [B*K, 1, L]
        lane_ctx  = lane_ctx.squeeze(1)                                             # [B*K, D]
        attn_w_sq = attn_w.squeeze(1)                                              # [B*K, L]

        # MLP → 80 timestep 한 번에 예측 (절대 좌표)
        feat = torch.cat([query, lane_ctx], dim=-1)                                 # [B*K, D*2]
        traj = self.traj_mlp(feat).reshape(B*K, T, 2)                              # [B*K, T, 2]

        # lane endpoint anchor: 각 mode를 attention-weighted lane endpoint로 편향
        # mode k가 lane l에 집중 → 그 lane의 공간 방향이 traj에 반영됨
        if self.use_lane_anchor and lane_pts is not None:
            lane_pts_bk = lane_pts.unsqueeze(1).expand(B, K, L, 2).reshape(B*K, L, 2)
            anchor = (attn_w_sq.unsqueeze(-1) * lane_pts_bk).sum(dim=1)            # [B*K, 2]
            t_ratio = torch.linspace(0, 1, T, device=traj.device).view(1, T, 1)   # ramp 0→1
            traj = traj + anchor.unsqueeze(1) * t_ratio                            # t=0: no offset, t=T-1: full anchor

        return traj.reshape(B, K, T, 2), attn_w_sq.reshape(B, K, L)

    def _causal_decode(self, context, mode_emb, lane_feat, goals=None, T=80):
        """
        context  : [B, D*2]
        mode_emb : [K, D]
        lane_feat: [B, 50, D]
        goals    : [B, K, 2] or None
        반환     : ([B, K, T, 2], None)
        """
        B = context.shape[0]
        K = self.K
        L = lane_feat.shape[1]

        ctx_bk  = context.unsqueeze(1).expand(B, K, -1).reshape(B*K, -1)
        mode_bk = mode_emb.unsqueeze(0).expand(B, K, -1).reshape(B*K, -1)
        lane_bk = lane_feat.unsqueeze(1).expand(B, K, L, -1).reshape(B*K, L, -1)

        if goals is not None and (self.use_goal_cond or self.use_lane_goal):
            goal_bk  = goals.reshape(B * K, 2)
            goal_emb = self.goal_proj(goal_bk)                          # [B*K, D]
            cond = torch.cat([ctx_bk, mode_bk, goal_emb], dim=-1)      # [B*K, D*4]
        else:
            cond = torch.cat([ctx_bk, mode_bk], dim=-1)                 # [B*K, D*3]

        traj = self.causal_decoder(cond, lane_bk)      # [B*K, T, 2]
        return traj.reshape(B, K, T, 2), None

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

        if self.use_lane_graph:
            lane_feat = self.lane_graph_enc(lane_feat, map_scene)

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
            lane_attn_w = None
        elif self.use_causal_attn:
            lane_goal_logits = None
            goals = None
            if self.use_goal_cond:
                goals = self.goal_head(context)                          # [B, K, 2]
            if self.use_lane_goal:
                lane_goals, lane_goal_logits = self.lane_goal_head(
                    lane_feat, context=context, map_scene=map_scene)
                if goals is not None and hasattr(self, "goal_gate"):
                    gate = torch.sigmoid(self.goal_gate(context)).unsqueeze(1)  # [B,1,1]
                    goals = gate * goals + (1 - gate) * lane_goals
                elif goals is not None:
                    goals = (goals + lane_goals) / 2
                else:
                    goals = lane_goals
            traj, lane_attn_w = self._causal_decode(
                context, mode_emb, lane_feat, goals=goals, T=80)           # [B, K, 80, 2], None
        else:
            lane_pts = map_scene[:, :, -1, :2] if self.use_lane_anchor else None
            traj, lane_attn_w = self._mlp_decode(
                context, mode_emb, lane_feat, lane_pts=lane_pts, T=80)    # [B, K, 80, 2], [B, K, 50]

        return {
            "trajectory":       traj,                                   # [B, K, 80, 2]
            "risk_logits":      risk_logits,                            # [B, 3] or None
            "lane_prob":        lane_prob,                              # [B, 50] or None
            "lane_attn_w":      lane_attn_w,                           # [B, K, 50] or None
            "goals":            goals if self.use_causal_attn else None,  # [B, K, 2] or None
            "lane_goal_logits": lane_goal_logits,                      # [B, K, 50] or None
        }
