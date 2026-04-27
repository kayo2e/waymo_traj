"""
Feature extraction for Waymo Open Motion Dataset.

Per-scenario output (all ego-relative at current_time_index):
  agent_tensor   : [N_AGENTS, T_HIST, 6]  (x, y, vx, vy, cos_h, sin_h)
  agent_mask     : [N_AGENTS]             bool — valid slot
  scene_tensor   : [N_MAP, 10, 3]         map polyline xyz
  traffic_tensor : [N_TRAF, 1]            signal state float
  gt_trajectory  : [N_FUTURE, 2]          ego future x,y (nan if invalid)
  gt_valid       : [N_FUTURE]             bool
  gt_keypoints   : [3, 2]                 1s/3s/5s keypoints
  kp_valid       : [3]                    bool
"""

import numpy as np

N_AGENTS = 32
T_HIST   = 11    # frames 0..10  (current_time_index = 10)
N_MAP    = 50
N_TRAF   = 6
N_FUTURE = 80    # frames 11..90
KP_STEPS = [9, 29, 49]   # 1s / 3s / 5s  (0-indexed into N_FUTURE)


def _world_to_ego(wx, wy, x0, y0, cos_h, sin_h):
    dx, dy = wx - x0, wy - y0
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _vel_to_ego(vx_w, vy_w, cos_h, sin_h):
    return vx_w * cos_h + vy_w * sin_h, -vx_w * sin_h + vy_w * cos_h


def extract_features(scenario):
    """
    scenario : waymo_open_dataset.protos.scenario_pb2.Scenario
    Returns  : dict of numpy arrays (see module docstring)
    """
    t0      = scenario.current_time_index   # always 10 in WOMD
    sdc_idx = scenario.sdc_track_index

    # ── ego pose at t0 → coordinate origin ──────────────────────────────────
    ego_s = scenario.tracks[sdc_idx].states[t0]
    x0, y0 = ego_s.center_x, ego_s.center_y
    theta   = ego_s.heading
    cos_h, sin_h = float(np.cos(theta)), float(np.sin(theta))

    # ── sort other agents by distance at t0 ─────────────────────────────────
    others = []
    for i, track in enumerate(scenario.tracks):
        if i == sdc_idx:
            continue
        s = track.states[t0]
        if s.valid:
            dist = (s.center_x - x0) ** 2 + (s.center_y - y0) ** 2
            others.append((dist, i))
    others.sort()
    agent_idxs = [sdc_idx] + [i for _, i in others[: N_AGENTS - 1]]

    # ── agent features ───────────────────────────────────────────────────────
    agent_tensor = np.zeros((N_AGENTS, T_HIST, 6), dtype=np.float32)
    agent_mask   = np.zeros(N_AGENTS, dtype=bool)

    for slot, trk_idx in enumerate(agent_idxs):
        track = scenario.tracks[trk_idx]
        for t in range(T_HIST):
            s = track.states[t]
            if not s.valid:
                continue
            ex, ey   = _world_to_ego(s.center_x, s.center_y, x0, y0, cos_h, sin_h)
            evx, evy = _vel_to_ego(s.velocity_x, s.velocity_y, cos_h, sin_h)
            rel_h    = s.heading - theta
            agent_tensor[slot, t] = [ex, ey, evx, evy, np.cos(rel_h), np.sin(rel_h)]
        agent_mask[slot] = True

    # ── map polylines ────────────────────────────────────────────────────────
    scene_tensor = np.zeros((N_MAP, 10, 3), dtype=np.float32)
    for i, feat in enumerate(list(scenario.map_features)[:N_MAP]):
        pts = None
        try:
            if feat.HasField("lane"):
                pts = feat.lane.polyline
            elif feat.HasField("road_edge"):
                pts = feat.road_edge.polyline
            elif feat.HasField("road_line"):
                pts = feat.road_line.polyline
        except Exception:
            pass
        if pts is None or len(pts) == 0:
            continue
        raw = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        if len(raw) >= 10:
            idx = np.linspace(0, len(raw) - 1, 10, dtype=int)
            arr = raw[idx]
        else:
            arr = np.zeros((10, 3), dtype=np.float32)
            arr[: len(raw)] = raw
        ex, ey = _world_to_ego(arr[:, 0], arr[:, 1], x0, y0, cos_h, sin_h)
        arr[:, 0], arr[:, 1] = ex, ey
        scene_tensor[i] = arr

    # ── traffic signals ──────────────────────────────────────────────────────
    traffic_tensor = np.zeros((N_TRAF, 1), dtype=np.float32)
    if len(scenario.dynamic_map_states) > t0:
        for j, ls in enumerate(scenario.dynamic_map_states[t0].lane_states[:N_TRAF]):
            traffic_tensor[j, 0] = float(ls.state)

    # ── GT future trajectory (ego-relative) ─────────────────────────────────
    sdc_track     = scenario.tracks[sdc_idx]
    gt_trajectory = np.full((N_FUTURE, 2), np.nan, dtype=np.float32)
    gt_valid      = np.zeros(N_FUTURE, dtype=bool)

    for k in range(N_FUTURE):
        t = t0 + 1 + k
        if t >= len(sdc_track.states):
            break
        s = sdc_track.states[t]
        if s.valid:
            ex, ey = _world_to_ego(s.center_x, s.center_y, x0, y0, cos_h, sin_h)
            gt_trajectory[k] = [ex, ey]
            gt_valid[k] = True

    # ── GT keypoints: 1s / 3s / 5s ──────────────────────────────────────────
    gt_keypoints = np.zeros((3, 2), dtype=np.float32)
    kp_valid     = np.zeros(3, dtype=bool)
    for ki, step in enumerate(KP_STEPS):
        if gt_valid[step]:
            gt_keypoints[ki] = gt_trajectory[step]
            kp_valid[ki] = True

    return {
        "agent_tensor":   agent_tensor,    # [32, 11, 6]
        "agent_mask":     agent_mask,      # [32]
        "scene_tensor":   scene_tensor,    # [50, 10, 3]
        "traffic_tensor": traffic_tensor,  # [6, 1]
        "gt_trajectory":  gt_trajectory,   # [80, 2]
        "gt_valid":       gt_valid,        # [80]
        "gt_keypoints":   gt_keypoints,    # [3, 2]
        "kp_valid":       kp_valid,        # [3]
    }
