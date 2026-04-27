"""
Stage 2: Bridge functions and MultiMambaEncoder.

bridge_stage1_to_stage2  — DataFrame/list → PyTorch tensors
MultiMambaEncoder        — parallel Mamba encoding of agents / map / traffic
bridge_stage2_to_stage3  — Mamba context + risk labels → Stage 3 package
"""

import torch
import torch.nn as nn

from waymo_traj.src.models.encoders import TambaMambaEncoder, JointPolylineEncoder


def bridge_stage1_to_stage2(stage1_output: dict, waymo_raw_data) -> dict:
    """Convert Stage 1 DataFrames and lists to PyTorch tensors for Stage 2."""
    import numpy as np

    agent_df       = stage1_output["temporal_mamba_input"]
    agent_features = agent_df[["distance", "accel"]].values.astype(np.float32)
    temporal_tensor = torch.from_numpy(agent_features)

    map_list     = stage1_output["scene_mamba_input"]
    scene_tensor = torch.zeros((1, len(map_list), 10, 3))   # [B, Lanes, Points, XYZ]

    signal_list     = stage1_output["traffic_mamba_input"]
    traffic_tensor  = torch.zeros((1, len(signal_list), 1)) # [B, Signals, 1]

    return {
        "temporal_tensor": temporal_tensor,
        "scene_tensor":    scene_tensor,
        "traffic_tensor":  traffic_tensor,
    }


class MultiMambaEncoder(nn.Module):
    """
    Parallel Mamba encoding of agents, map polylines, and traffic signals,
    followed by global cross-context interaction.

    agents    : [B, N, 2]      (distance, accel per agent)
    map_lanes : [B, L, 10, 3]  polyline XYZ
    traffic   : [B, S, 1]      signal state
    output    : [B, N+L+S, D]
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.agent_embed   = JointPolylineEncoder(input_dim=2, d_model=d_model)
        self.map_embed     = JointPolylineEncoder(input_dim=3, d_model=d_model)
        self.traffic_embed = nn.Linear(1, d_model)

        self.temporal_mamba = TambaMambaEncoder(d_model=d_model)
        self.scene_mamba    = TambaMambaEncoder(d_model=d_model)
        self.traffic_mamba  = TambaMambaEncoder(d_model=d_model)
        self.global_mamba   = TambaMambaEncoder(d_model=d_model)

    def forward(self, agents: torch.Tensor, traffic: torch.Tensor,
                map_lanes: torch.Tensor) -> torch.Tensor:
        a_emb = self.agent_embed(agents)    # [B, N, D]
        m_emb = self.map_embed(map_lanes)   # [B, L, D]
        t_emb = self.traffic_embed(traffic) # [B, S, D]

        a_feat = self.temporal_mamba(a_emb)
        m_feat = self.scene_mamba(m_emb)
        t_feat = self.traffic_mamba(t_emb)

        combined = torch.cat([a_feat, m_feat, t_feat], dim=1)
        return self.global_mamba(combined)


def bridge_stage2_to_stage3(mamba_context, risk_events_df, scenario_data: dict) -> dict:
    """Bundle Mamba context + risk text conditions for Stage 3."""
    current_idx       = scenario_data["current_time_index"]
    current_conditions = risk_events_df[risk_events_df["frame"] == current_idx]

    text_conditions = [
        f"Target {row['track_id']} is in [{row['event_label']}] state "
        f"(Distance: {row['distance']:.1f}m, Accel: {row['accel']:.2f}m/s²)."
        for _, row in current_conditions.iterrows()
    ]

    return {
        "mamba_embedding":   mamba_context,
        "qa_prompt_context": text_conditions,
        "scenario_id":       scenario_data["scenario_id"],
    }
