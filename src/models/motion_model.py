"""
WaymoMotionModel — true k-gen trajectory prediction (K diverse hypotheses).

Token layout after concatenation (dim=1):
  [0]          ego agent
  [1..31]      nearby agents (sorted by distance)
  [32..81]     map polylines
  [82..87]     traffic signals

Ego token (index 0) + K mode queries → K keypoint sets + K trajectories.
"""

import torch
import torch.nn as nn

from src.models.encoders import TambaMambaEncoder, JointPolylineEncoder


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
