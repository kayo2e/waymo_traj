"""
Stage 1: Agent filtering, risk labeling, and map/signal extraction.

Operates on the dict-based scenario_data format produced by the legacy
bridge layer (not the proto-based extract_features in src/data/features.py).
"""

import numpy as np
import pandas as pd


def filter_agents_by_connectivity(scenario_data: dict) -> pd.DataFrame:
    """Return a DataFrame of valid ego + predicted-agent states up to current_time_index."""
    tracks        = scenario_data["tracks"]
    predict_dict  = scenario_data["tracks_to_predict"]
    predict_idxs  = [item["track_index"] for item in predict_dict]
    current_idx   = scenario_data["current_time_index"]

    rows = []
    for t in range(current_idx + 1):
        for trk_idx in predict_idxs:
            state = tracks[trk_idx]["states"][t]
            if state["valid"]:
                rows.append({
                    "frame":    t,
                    "track_id": trk_idx,
                    "x":        state["center_x"],
                    "y":        state["center_y"],
                    "vx":       state["velocity_x"],
                    "vy":       state["velocity_y"],
                    "heading":  state["heading"],
                })
    return pd.DataFrame(rows)


def label_risk_events(filtered_agents_df: pd.DataFrame, scenario_data: dict) -> pd.DataFrame:
    """
    Attach risk labels (Proximity / Sudden Braking / Lane Change / Normal)
    to each agent-frame row.
    """
    tracks      = scenario_data["tracks"]
    sdc_idx     = scenario_data["sdc_track_index"]
    timestamps  = scenario_data["timestamps_seconds"]

    rows = []
    for _, row in filtered_agents_df.iterrows():
        t      = int(row["frame"])
        trk_id = int(row["track_id"])

        ego_state        = tracks[sdc_idx]["states"][t]
        ego_x, ego_y     = ego_state["center_x"], ego_state["center_y"]
        distance         = np.sqrt((row["x"] - ego_x) ** 2 + (row["y"] - ego_y) ** 2)

        accel        = 0.0
        heading_diff = 0.0
        prev_data    = filtered_agents_df[
            (filtered_agents_df["frame"] == t - 1) &
            (filtered_agents_df["track_id"] == trk_id)
        ]
        if not prev_data.empty:
            p_row  = prev_data.iloc[0]
            v_curr = np.sqrt(row["vx"] ** 2 + row["vy"] ** 2)
            v_prev = np.sqrt(p_row["vx"] ** 2 + p_row["vy"] ** 2)
            dt     = timestamps[t] - timestamps[t - 1]
            accel  = (v_curr - v_prev) / dt
            heading_diff = abs(row["heading"] - p_row["heading"])

        if distance <= 7.0:
            label = "Proximity"
        elif accel <= -6.0:
            label = "Sudden Braking"
        elif heading_diff > 0.15:
            label = "Lane Change"
        else:
            label = "Normal"

        rows.append({
            "frame":       t,
            "track_id":    trk_id,
            "distance":    distance,
            "accel":       accel,
            "event_label": label,
        })
    return pd.DataFrame(rows)


def extract_map_and_signals(scenario_data: dict) -> dict:
    """Return map polylines and current traffic signal states."""
    current_idx      = scenario_data["current_time_index"]
    map_topology     = scenario_data.get("map_features", [])[:50]
    traffic_signals  = scenario_data.get("dynamic_map_states", [])
    current_signals  = traffic_signals[current_idx] if len(traffic_signals) > current_idx else []
    return {"map_topology": map_topology, "traffic_signals": current_signals}


def run_stage1_pipeline(scenario_data: dict) -> dict:
    """Run all Stage 1 steps and return a Stage-2-ready data bundle."""
    print("=== [Stage 1] 파이프라인 가동 ===")

    filtered_agents = filter_agents_by_connectivity(scenario_data)
    risk_events     = label_risk_events(filtered_agents, scenario_data)
    map_and_signals = extract_map_and_signals(scenario_data)

    stage2_inputs = {
        "temporal_mamba_input": risk_events,
        "scene_mamba_input":    map_and_signals["map_topology"],
        "traffic_mamba_input":  map_and_signals["traffic_signals"],
    }
    print("→ Stage 2 Mamba 인코더로 전달할 3가지 데이터셋 준비 완료.")
    return stage2_inputs
