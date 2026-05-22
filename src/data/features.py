"""
Feature extraction for Waymo Open Motion Dataset.

Per-scenario output (all ego-relative at current_time_index):
  agent_tensor   : [N_AGENTS, T_HIST, 10]
                   (x, y, vx, vy, cos_h, sin_h, valid_t, t_norm, type_v, type_c)
  agent_mask     : [N_AGENTS]             bool — valid slot
  scene_tensor   : [N_MAP, 10, 6]
                   (x, y, dx_norm, dy_norm, is_lane, is_special)
  traffic_tensor : [N_TRAF, 1]            signal state float
  gt_trajectory  : [N_FUTURE, 2]          ego future x,y (nan if invalid)
  gt_valid       : [N_FUTURE]             bool
  gt_keypoints   : [3, 2]                 1s/3s/5s keypoints
  kp_valid       : [3]                    bool
"""

import numpy as np

N_AGENTS  = 32
T_HIST    = 11    # frames 0..10  (current_time_index = 10)
N_MAP     = 50
N_TRAF    = 6
N_FUTURE  = 80    # frames 11..90
KP_STEPS  = [9, 29, 49]   # 1s / 3s / 5s  (0-indexed into N_FUTURE)

AGENT_DIM = 10   # x, y, vx, vy, cos_h, sin_h, valid_t, t_norm, type_v, type_c
MAP_DIM   = 6    # x, y, dx_norm, dy_norm, is_lane, is_special


def _world_to_ego(wx, wy, x0, y0, cos_h, sin_h):
    dx, dy = wx - x0, wy - y0
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


def _vel_to_ego(vx_w, vy_w, cos_h, sin_h):
    return vx_w * cos_h + vy_w * sin_h, -vx_w * sin_h + vy_w * cos_h


def _direction_vecs(pts_xy: np.ndarray) -> np.ndarray:
    """Normalized segment direction vectors for a polyline [N, 2] → [N, 2]."""
    dirs  = np.diff(pts_xy, axis=0)         # [N-1, 2]
    dirs  = np.vstack([dirs, dirs[-1:]])     # [N, 2]  duplicate last
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(norms, 1e-6)   # [N, 2]


def _polyline_row(arr_xy: np.ndarray,
                  is_lane: float, is_special: float) -> np.ndarray:
    """Build [10, 6] map feature row from ego-relative xy [10, 2]."""
    dirs  = _direction_vecs(arr_xy)          # [10, 2]
    flags = np.full((10, 2), [is_lane, is_special], dtype=np.float32)
    return np.concatenate([arr_xy, dirs, flags], axis=1)  # [10, 6]


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

    # ── agent features [N_AGENTS, T_HIST, AGENT_DIM] ────────────────────────
    agent_tensor = np.zeros((N_AGENTS, T_HIST, AGENT_DIM), dtype=np.float32)
    agent_mask   = np.zeros(N_AGENTS, dtype=bool)

    for slot, trk_idx in enumerate(agent_idxs):
        track  = scenario.tracks[trk_idx]
        # TYPE_VEHICLE=1, TYPE_PEDESTRIAN=2, TYPE_CYCLIST=3
        type_v = float(track.object_type == 1)
        type_c = float(track.object_type == 3)
        for t in range(T_HIST):
            s = track.states[t]
            if not s.valid:
                continue   # invalid frames stay zero (valid_t=0 signals this)
            ex, ey   = _world_to_ego(s.center_x, s.center_y, x0, y0, cos_h, sin_h)
            evx, evy = _vel_to_ego(s.velocity_x, s.velocity_y, cos_h, sin_h)
            rel_h    = s.heading - theta
            t_norm   = t / (T_HIST - 1)   # 0.0 → 1.0
            agent_tensor[slot, t] = [
                ex, ey, evx, evy,
                np.cos(rel_h), np.sin(rel_h),
                1.0,           # valid_t — model knows this frame has real data
                t_norm,
                type_v, type_c,
            ]
        agent_mask[slot] = True

    # ── map polylines [N_MAP, 10, MAP_DIM] ───────────────────────────────────
    scene_tensor = np.zeros((N_MAP, 10, MAP_DIM), dtype=np.float32)
    for i, feat in enumerate(list(scenario.map_features)[:N_MAP]):
        pts        = None
        is_lane    = 0.0
        is_special = 0.0
        handled    = False

        try:
            if feat.HasField("lane"):
                pts     = feat.lane.polyline
                is_lane = 1.0

            elif feat.HasField("road_edge"):
                pts = feat.road_edge.polyline

            elif feat.HasField("road_line"):
                pts = feat.road_line.polyline

            elif feat.HasField("stop_sign"):
                pos    = feat.stop_sign.position
                ex, ey = _world_to_ego(pos.x, pos.y, x0, y0, cos_h, sin_h)
                arr_xy = np.full((10, 2), [ex, ey], dtype=np.float32)
                # stop sign has no direction → zero vectors
                scene_tensor[i] = _polyline_row(arr_xy, 0.0, 1.0)
                handled = True

            elif feat.HasField("crosswalk"):
                poly = feat.crosswalk.polygon
                if len(poly) > 0:
                    pts_w  = np.array([[p.x, p.y] for p in poly], dtype=np.float32)
                    cx, cy = pts_w.mean(axis=0)
                    ex, ey = _world_to_ego(cx, cy, x0, y0, cos_h, sin_h)
                    arr_xy = np.full((10, 2), [ex, ey], dtype=np.float32)
                    scene_tensor[i] = _polyline_row(arr_xy, 0.0, 1.0)
                handled = True

            elif feat.HasField("speed_bump"):
                poly = feat.speed_bump.polygon
                if len(poly) > 0:
                    pts_w  = np.array([[p.x, p.y] for p in poly], dtype=np.float32)
                    cx, cy = pts_w.mean(axis=0)
                    ex, ey = _world_to_ego(cx, cy, x0, y0, cos_h, sin_h)
                    arr_xy = np.full((10, 2), [ex, ey], dtype=np.float32)
                    scene_tensor[i] = _polyline_row(arr_xy, 0.0, 1.0)
                handled = True

        except Exception:
            pass

        if handled or pts is None or len(pts) == 0:
            continue

        raw = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        if len(raw) >= 10:
            idx    = np.linspace(0, len(raw) - 1, 10, dtype=int)
            arr_xy = raw[idx]
        else:
            arr_xy = np.zeros((10, 2), dtype=np.float32)
            arr_xy[: len(raw)] = raw

        ex, ey           = _world_to_ego(arr_xy[:, 0], arr_xy[:, 1], x0, y0, cos_h, sin_h)
        arr_xy[:, 0]     = ex
        arr_xy[:, 1]     = ey
        scene_tensor[i]  = _polyline_row(arr_xy, is_lane, is_special)

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
            gt_valid[k]      = True

    # ── GT keypoints: 1s / 3s / 5s ──────────────────────────────────────────
    gt_keypoints = np.zeros((3, 2), dtype=np.float32)
    kp_valid     = np.zeros(3, dtype=bool)
    for ki, step in enumerate(KP_STEPS):
        if gt_valid[step]:
            gt_keypoints[ki] = gt_trajectory[step]
            kp_valid[ki]     = True

    return {
        "agent_tensor":   agent_tensor,    # [32, 11, 10]
        "agent_mask":     agent_mask,      # [32]
        "scene_tensor":   scene_tensor,    # [50, 10, 6]
        "traffic_tensor": traffic_tensor,  # [6, 1]
        "gt_trajectory":  gt_trajectory,   # [80, 2]
        "gt_valid":       gt_valid,        # [80]
        "gt_keypoints":   gt_keypoints,    # [3, 2]
        "kp_valid":       kp_valid,        # [3]
    }


# ── Condition label constants ─────────────────────────────────────────────────
MANEUVER_NAMES = ["Stop", "GoStraight", "LaneChangeLeft", "LaneChangeRight", "TurnLeft", "TurnRight"]
SPEED_NAMES    = ["Stopped", "Slow", "Fast"]
COND_DIM       = 9   # 6 maneuver + 3 speed

# ── Risk label constants ──────────────────────────────────────────────────────
_PROXIMITY_M     = 7.0    # 근접 판단 거리 (m)
_HARD_BRAKE_MS2  = -6.0   # 급정지 가속도 임계값 (m/s²)
_LANE_CHG_VEL    = 1.0    # 차선변경 판단 횡방향 속도 (m/s)
_DT              = 0.1    # WOMD 샘플링 주기 (10 Hz)


def extract_risk_label(scenario) -> np.ndarray:
    """
    에고 차량의 T_HIST 구간에서 위험 이벤트를 이진 레이블로 추출.

    Returns [3] float32
      [0] Proximity  — 임의 에이전트와 거리 <= 7 m
      [1] HardBrake  — 종방향 감속도 <= -6.0 m/s²
      [2] LaneChange — 횡방향 속도 > 1.0 m/s (차선변경 근사)
    """
    t0      = scenario.current_time_index
    sdc_idx = scenario.sdc_track_index
    label   = np.zeros(3, dtype=np.float32)
    ego_trk = scenario.tracks[sdc_idx]

    # ── [0] Proximity ─────────────────────────────────────────────────────────
    for t in range(max(0, t0 - T_HIST + 1), t0 + 1):
        es = ego_trk.states[t]
        if not es.valid:
            continue
        for i, track in enumerate(scenario.tracks):
            if i == sdc_idx:
                continue
            s = track.states[t]
            if not s.valid:
                continue
            if np.hypot(s.center_x - es.center_x, s.center_y - es.center_y) <= _PROXIMITY_M:
                label[0] = 1.0
                break
        if label[0]:
            break

    # ── [1] HardBrake  [2] LaneChange ────────────────────────────────────────
    valid_states = [
        ego_trk.states[t]
        for t in range(max(0, t0 - T_HIST + 1), t0 + 1)
        if ego_trk.states[t].valid
    ]
    for i in range(1, len(valid_states)):
        prev, curr = valid_states[i - 1], valid_states[i]
        cos_h = float(np.cos(curr.heading))
        sin_h = float(np.sin(curr.heading))

        v_long_cur  = curr.velocity_x * cos_h + curr.velocity_y * sin_h
        v_long_prev = prev.velocity_x * cos_h + prev.velocity_y * sin_h
        if (v_long_cur - v_long_prev) / _DT <= _HARD_BRAKE_MS2:
            label[1] = 1.0

        if abs(-curr.velocity_x * sin_h + curr.velocity_y * cos_h) > _LANE_CHG_VEL:
            label[2] = 1.0

        if label[1] and label[2]:
            break

    return label


def extract_condition_label(scenario) -> np.ndarray:
    """
    GT 궤적 + 에고 히스토리에서 조건 레이블 추출.

    Returns [9] float32
      [0:6] Maneuver intent (6-class one-hot, GT에서 역산)
             Stop / GoStraight / LaneChangeLeft / LaneChangeRight / TurnLeft / TurnRight
      [6:9] Speed state (3-class one-hot, 현재 프레임 속도)
             Stopped (v<0.5m/s) / Slow (0.5≤v<5) / Fast (v≥5)
    """
    t0      = scenario.current_time_index
    sdc_idx = scenario.sdc_track_index
    ego_trk = scenario.tracks[sdc_idx]
    label   = np.zeros(COND_DIM, dtype=np.float32)

    ego_s = ego_trk.states[t0]

    # ── Speed state ───────────────────────────────────────────────────────────
    v = np.hypot(ego_s.velocity_x, ego_s.velocity_y)
    if v < 0.5:
        label[6] = 1.0   # Stopped
    elif v < 5.0:
        label[7] = 1.0   # Slow
    else:
        label[8] = 1.0   # Fast

    # ── Maneuver intent from GT future ────────────────────────────────────────
    x0, y0   = ego_s.center_x, ego_s.center_y
    cos_h    = float(np.cos(ego_s.heading))
    sin_h    = float(np.sin(ego_s.heading))

    valid_xy = []
    for k in range(N_FUTURE):
        t = t0 + 1 + k
        if t >= len(ego_trk.states):
            break
        s = ego_trk.states[t]
        if s.valid:
            ex, ey = _world_to_ego(s.center_x, s.center_y, x0, y0, cos_h, sin_h)
            valid_xy.append((ex, ey))

    if not valid_xy:
        label[0] = 1.0   # Stop (GT 없으면 정지로 간주)
        return label

    final_x, final_y = valid_xy[-1]
    total_dist = np.hypot(final_x, final_y)

    if total_dist < 2.0:
        label[0] = 1.0   # Stop
    elif abs(final_y) < 3.0:
        label[1] = 1.0   # GoStraight
    elif 3.0 <= final_y < 10.0:
        label[2] = 1.0   # LaneChangeLeft
    elif -10.0 < final_y <= -3.0:
        label[3] = 1.0   # LaneChangeRight
    elif final_y >= 10.0:
        label[4] = 1.0   # TurnLeft
    else:
        label[5] = 1.0   # TurnRight

    return label
