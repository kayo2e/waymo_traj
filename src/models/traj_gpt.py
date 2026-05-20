"""
GPT-style 인과적 궤적 예측 모델 (TrajGPT).

LLM의 next-token prediction을 궤적에 적용:
  [p_0, ..., p_10] → causal self-attn + scene cross-attn → p_11, ..., p_90

핵심 설계:
  Anchor prediction  : 현재 속도 기반 등속 궤적을 anchor로 설정,
                       모델은 그 잔차(residual)만 학습 → 랜덤 초기화에서도 안정
  Causal attention   : 과거 위치만 참조, 미래 leakage 없음 (LLM과 동일)
  Scene cross-attn   : RCM Joint Transformer encoder (risk-conditioned) 출력 사용
  K modes            : mode embedding을 각 토큰에 더해 다양한 미래 생성
  Risk conditioning  : risk_label [B,3] → encoder prefix token → 모든 어텐션에 반영

학습 (teacher forcing):
  input  = [ego_hist_feat | shift(gt_future_feat)]
  output = residual on top of constant-velocity anchor

추론 (autoregressive):
  히스토리 토큰으로 시작 → 잔차 예측 → anchor + residual = 실제 예측 위치
"""

import torch
import torch.nn as nn

from src.models.encoders import MultiStreamMambaEncoder

_DT = 0.1   # WOMD 10 Hz


class TrajGPT(nn.Module):
    """
    GPT-style causal 궤적 예측 + RCM Joint Transformer 씬 인코더.

    forward 입력:
      ego_hist      : [B, T_hist, agent_dim]
      social_agents : [B, N, T_hist, agent_dim]
      map_scene     : [B, 50, 10, map_dim]
      traf          : [B, 6, 1]
      risk_label    : [B, 3]   위험 레이블 (optional, encoder prefix token)
      gt_future     : [B, T_future, 2]   학습 시 teacher-forcing GT
      tf_prob       : float

    forward 출력:
      trajectory : [B, K, T_future, 2]   절대 좌표 (ego-relative)
    """

    def __init__(self, agent_dim: int = 10, map_dim: int = 6,
                 d_model: int = 128, n_layers: int = 4, n_heads: int = 4,
                 enc_layers: int = 2, ar_hidden: int = 256,
                 decoder_type: str = 'transformer',
                 K: int = 6, T_hist: int = 11, T_future: int = 80):
        super().__init__()
        D = d_model
        H = ar_hidden
        self.K            = K
        self.T_hist       = T_hist
        self.T_future     = T_future
        self.decoder_type = decoder_type

        # ── Scene encoder: RCM Joint Transformer (risk-conditioned) ─────────
        self.encoder = MultiStreamMambaEncoder(
            agent_dim=agent_dim, map_dim=map_dim, d_model=D,
            n_layers=enc_layers, n_heads=n_heads,
            use_risk_prefix=True, use_traj_fix=True,
        )

        # ── K mode embeddings ────────────────────────────────────────────────
        self.mode_queries = nn.Embedding(K, D)

        if decoder_type == 'gru':
            # ── GRU AR 디코더 ─────────────────────────────────────────────────
            # vel_proj: kinematic context 주입 (encoder와 동일 방식)
            self.vel_proj  = nn.Linear(4, D * 2)
            # (context D*2) + (mode D) → h0
            self.init_proj = nn.Linear(D * 2 + D, H)
            # 매 스텝: [prev_pos(2) | context(D*2)] → H
            self.gru_cell  = nn.GRUCell(input_size=2 + D * 2, hidden_size=H)
            self.out_proj  = nn.Linear(H, 2)
            nn.init.normal_(self.out_proj.weight, std=0.01)
            nn.init.zeros_(self.out_proj.bias)
        else:
            # ── Transformer 디코더 ────────────────────────────────────────────
            self.ego_proj = nn.Linear(agent_dim, D)
            self.fut_proj = nn.Linear(2, D)
            self.time_emb = nn.Embedding(T_hist + T_future, D)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=D, nhead=n_heads,
                dim_feedforward=D * 4, dropout=0.1,
                batch_first=True, norm_first=True,
            )
            self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
            self.out_proj = nn.Linear(D, 2)
            nn.init.normal_(self.out_proj.weight, std=0.01)
            nn.init.zeros_(self.out_proj.bias)

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool),
                          diagonal=1)

    def _constant_velocity_anchor(self, ego_hist, T_future):
        """
        현재 속도(vx, vy)로 등속 직선 이동하는 anchor 궤적.
        shape: [B, T_future, 2]
        """
        vx = ego_hist[:, -1, 2]
        vy = ego_hist[:, -1, 3]
        steps = torch.arange(1, T_future + 1,
                             dtype=ego_hist.dtype,
                             device=ego_hist.device) * _DT
        anchor_x = vx.unsqueeze(1) * steps
        anchor_y = vy.unsqueeze(1) * steps
        return torch.stack([anchor_x, anchor_y], dim=-1)

    def _make_hist_tokens(self, ego_hist, mode_bk_h):
        B, T_h, _ = ego_hist.shape
        t_idx = torch.arange(T_h, device=ego_hist.device)
        e = self.ego_proj(ego_hist) + self.time_emb(t_idx)
        e_bk = e.unsqueeze(1).expand(B, self.K, T_h, -1).reshape(B * self.K, T_h, -1)
        return e_bk + mode_bk_h

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, ego_hist, social_agents, map_scene, traf,
                gt_future=None, tf_prob: float = 1.0,
                risk_label=None, **kwargs):
        B = ego_hist.shape[0]
        K = self.K
        device = ego_hist.device

        # 1. Constant-velocity anchor [B, T_future, 2]
        anchor = self._constant_velocity_anchor(ego_hist, self.T_future)

        # 2. Scene encoding (risk-conditioned)
        global_feat, lane_feat = self.encoder(
            ego_hist, social_agents, map_scene, traf, risk_label=risk_label
        )

        # 3. Mode embeddings [K, D]
        mode_emb = self.mode_queries(torch.arange(K, device=device))

        if self.decoder_type == 'gru':
            return self._forward_gru(
                ego_hist, global_feat, mode_emb, anchor, gt_future, tf_prob, B, K, device
            )

        # ── Transformer 경로 ──────────────────────────────────────────────────
        scene    = torch.cat([global_feat.unsqueeze(1), lane_feat], dim=1)  # [B,51,D]
        scene_bk = (scene.unsqueeze(1)
                        .expand(B, K, *scene.shape[1:])
                        .reshape(B * K, *scene.shape[1:]))

        mode_bk_h = (mode_emb
                     .unsqueeze(1).expand(K, self.T_hist, -1)
                     .unsqueeze(0).expand(B, K, self.T_hist, -1)
                     .reshape(B * K, self.T_hist, -1))
        hist_tok = self._make_hist_tokens(ego_hist, mode_bk_h)

        if gt_future is not None and tf_prob > 0.0:
            return self._forward_train(
                hist_tok, scene_bk, mode_emb, anchor, gt_future, B, K, device
            )
        else:
            return self._forward_infer(
                hist_tok, scene_bk, mode_emb, anchor, B, K, device
            )

    # ── GRU AR 디코더 ─────────────────────────────────────────────────────

    def _forward_gru(self, ego_hist, global_feat, mode_emb, anchor,
                     gt_future, tf_prob, B, K, device):
        """
        context [B, D*2] = context_proj(global_feat) + vel_proj(kinematic)
        각 mode k:  h0 = init_proj([context | mode_emb[k]])
        매 스텝:    GRUCell([prev_pos | context]) → delta → anchor + delta
        """
        T_f = self.T_future

        # context [B, D*2]
        context = self.vel_proj(ego_hist[:, -1, 2:6])   # kinematic만으로 D*2 생성

        # [B*K, D*2]
        ctx_bk  = context.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        mode_bk = mode_emb.unsqueeze(0).expand(B, K, -1).reshape(B * K, -1)

        # GRU 초기 hidden state [B*K, H]
        h = self.init_proj(torch.cat([ctx_bk, mode_bk], dim=-1))

        # anchor [B*K, T_f, 2]
        anchor_bk = (anchor.unsqueeze(1)
                           .expand(B, K, T_f, 2)
                           .reshape(B * K, T_f, 2))

        prev_pos  = torch.zeros(B * K, 2, device=device)
        positions = []

        for t in range(T_f):
            gru_in   = torch.cat([prev_pos, ctx_bk], dim=-1)   # [B*K, 2+D*2]
            h        = self.gru_cell(gru_in, h)
            residual = self.out_proj(h)                          # [B*K, 2]
            next_pos = anchor_bk[:, t, :] + residual
            positions.append(next_pos)

            # teacher forcing
            if gt_future is not None and tf_prob > 0.0:
                gt_bk = (gt_future[:, t, :]
                         .unsqueeze(1).expand(B, K, 2)
                         .reshape(B * K, 2))
                if tf_prob >= 1.0:
                    prev_pos = gt_bk
                else:
                    mask     = torch.rand(B * K, 1, device=device) < tf_prob
                    prev_pos = torch.where(mask, gt_bk, next_pos.detach())
            else:
                prev_pos = next_pos.detach()

        traj = torch.stack(positions, dim=1).reshape(B, K, T_f, 2)
        return {"trajectory": traj, "risk_logits": None, "lane_prob": None}

    # ── Teacher-forcing (병렬) ─────────────────────────────────────────────

    def _forward_train(self, hist_tok, scene_bk, mode_emb, anchor,
                       gt_future, B, K, device):
        T_f = self.T_future

        gt_bk = (gt_future.unsqueeze(1)
                           .expand(B, K, T_f, 2)
                           .reshape(B * K, T_f, 2))

        t_idx_fut = torch.arange(self.T_hist, self.T_hist + T_f, device=device)
        mode_bk_f = (mode_emb
                     .unsqueeze(1).expand(K, T_f, -1)
                     .unsqueeze(0).expand(B, K, T_f, -1)
                     .reshape(B * K, T_f, -1))

        fut_tok = self.fut_proj(gt_bk) + self.time_emb(t_idx_fut) + mode_bk_f

        zero   = torch.zeros(B * K, 1, fut_tok.shape[-1], device=device)
        fut_in = torch.cat([zero, fut_tok[:, :-1]], dim=1)

        seq    = torch.cat([hist_tok, fut_in], dim=1)
        causal = self._causal_mask(seq.shape[1], device)
        out    = self.decoder(seq, scene_bk, tgt_mask=causal)

        fut_out  = out[:, self.T_hist:, :]
        residual = self.out_proj(fut_out)

        anchor_bk = (anchor.unsqueeze(1)
                           .expand(B, K, T_f, 2)
                           .reshape(B * K, T_f, 2))
        traj = (anchor_bk + residual).reshape(B, K, T_f, 2)

        return {"trajectory": traj, "risk_logits": None, "lane_prob": None}

    # ── Autoregressive 추론 ───────────────────────────────────────────────

    def _forward_infer(self, hist_tok, scene_bk, mode_emb, anchor, B, K, device):
        T_f     = self.T_future
        tokens  = hist_tok
        mode_bk1 = (mode_emb.unsqueeze(0)
                             .expand(B, K, -1)
                             .reshape(B * K, 1, -1))

        anchor_bk = (anchor.unsqueeze(1)
                           .expand(B, K, T_f, 2)
                           .reshape(B * K, T_f, 2))

        cur_pos   = torch.zeros(B * K, 2, device=device)
        positions = []

        for t in range(T_f):
            t_idx = torch.tensor([self.T_hist + t], device=device)
            tok   = (self.fut_proj(cur_pos.unsqueeze(1))
                     + self.time_emb(t_idx)
                     + mode_bk1)
            tokens = torch.cat([tokens, tok], dim=1)

            causal   = self._causal_mask(tokens.shape[1], device)
            out      = self.decoder(tokens, scene_bk, tgt_mask=causal)
            residual = self.out_proj(out[:, -1, :])

            cur_pos = anchor_bk[:, t, :] + residual
            positions.append(cur_pos)

        traj = torch.stack(positions, dim=1).reshape(B, K, T_f, 2)
        return {"trajectory": traj, "risk_logits": None, "lane_prob": None}
