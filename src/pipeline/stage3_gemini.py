"""
Stage 3-C: Gemini-based trajectory explanation and refinement.

generate_gemini_explanation  — safety analyst explanation given risk + keypoints
mamba_context_to_text        — convert Mamba attention norms to natural language
gemini_refine_trajectory     — ask Gemini to output a physically consistent 80-step traj
"""

import os
import json
import re

import numpy as np

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

_API_KEY_ENV = "GOOGLE_API_KEY"
_FALLBACK_KEY = "AIzaSyALDaEkZSghM3iG2VrOfuIbly3IAzGKw64"


def _get_api_key() -> str:
    return os.environ.get(_API_KEY_ENV, _FALLBACK_KEY)


def generate_gemini_explanation(text_conditions, keypoints_np: np.ndarray) -> str:
    """
    Ask Gemini to explain the predicted trajectory in [Subject]+[Action]+[Constraint] format.

    text_conditions : list[str]  risk descriptions from Stage 1
    keypoints_np    : [3, 2]     predicted 1s/3s/5s waypoints
    """
    prompt = (
        "You are an autonomous driving safety analyst.\n"
        f"Current risk conditions: {text_conditions}\n"
        f"Predicted waypoints (1s, 3s, 5s): {keypoints_np.tolist()}\n\n"
        "In 2-3 sentences, explain why the vehicle will follow this trajectory "
        "considering the risk conditions. Use [Subject]+[Action]+[Constraint] format."
    )
    try:
        import google.generativeai as genai
        genai.configure(api_key=_get_api_key())
        for model_name in GEMINI_MODELS:
            try:
                m = genai.GenerativeModel(model_name)
                return m.generate_content(prompt).text
            except Exception:
                pass
        return "[Mock] Gemini unavailable."
    except Exception as e:
        return f"[Mock - {e}]"


def mamba_context_to_text(context_tensor) -> str:
    """
    Summarise which tokens the Mamba encoder attended to most.

    context_tensor : [1, N_tok, D]  from model forward()["context"]
    """
    ctx   = context_tensor[0].detach().cpu().numpy()   # [N_tok, D]
    norms = np.linalg.norm(ctx, axis=1)                # [N_tok]

    agent_norms = norms[:32]
    map_norms   = norms[32:82]
    traf_norm   = norms[82] if len(norms) > 82 else 0.0

    lines = ["[Mamba Context Attention]"]
    for rank, idx in enumerate(agent_norms.argsort()[::-1][:3]):
        role = "ego" if idx == 0 else f"agent_{idx}"
        lines.append(f"  Agent #{rank+1}: token {idx} ({role})  activation={agent_norms[idx]:.2f}")
    for rank, idx in enumerate(map_norms.argsort()[::-1][:3]):
        lines.append(f"  Map #{rank+1}: polyline_{idx}  activation={map_norms[idx]:.2f}")
    lines.append(f"  Traffic token: activation={traf_norm:.2f}")
    dom = "agents" if agent_norms.max() > map_norms.max() else "map polylines"
    lines.append(f"  → Dominant attention: {dom}")
    return "\n".join(lines)


def gemini_refine_trajectory(feats: dict, result: dict, traj_best: np.ndarray):
    """
    Ask Gemini to refine the best-mode ML trajectory for physical plausibility.

    feats     : output of extract_features()
    result    : output of WaymoMotionModel.forward()
    traj_best : [80, 2]  best-mode ML trajectory (ego-relative, metres)

    Returns (traj: np.ndarray [80,2] or None, raw_response: str)
    """
    kp      = result["keypoints"][0].detach().cpu().numpy()   # [K, 3, 2]
    kp_best = kp[0]                                           # use mode 0 for prompt
    ctx_txt = mamba_context_to_text(result["context"])

    kp_lines = []
    for ki, label in enumerate(["1s", "3s", "5s"]):
        px, py = kp_best[ki]
        if feats["kp_valid"][ki]:
            gx, gy = feats["gt_keypoints"][ki]
            kp_lines.append(f"  {label} → x={px:+.3f}m, y={py:+.3f}m  (GT: x={gx:+.3f}, y={gy:+.3f})")
        else:
            kp_lines.append(f"  {label} → x={px:+.3f}m, y={py:+.3f}m")

    traj_lines = [
        f"  t={i*0.1:.1f}s → x={traj_best[i,0]:+.3f}m, y={traj_best[i,1]:+.3f}m"
        for i in range(0, 80, 10)
    ]

    prompt = (
        "You are Stage 3-C of an autonomous driving trajectory prediction pipeline.\n"
        "The upstream ML model (WaymoMotionModel) has been TRAINED on real Waymo data,\n"
        "so its keypoints and trajectory are meaningful anchors.\n\n"
        "=== STAGE 2: Mamba Context Attention ===\n"
        + ctx_txt + "\n\n"
        "=== STAGE 3-A: Trained Keypoint Predictions (1s/3s/5s) ===\n"
        + "\n".join(kp_lines) + "\n\n"
        "=== STAGE 3-B: ML Dense Trajectory (sampled every 1s) ===\n"
        + "\n".join(traj_lines) + "\n\n"
        "=== YOUR TASK ===\n"
        "Refine the 80-step trajectory for physical plausibility.\n"
        "- Ego-relative: x=forward, y=left (metres)\n"
        "- 0.1s per step, smooth motion, max 3m/step\n"
        "- Pass near the ML keypoints (they are trained predictions, not random)\n\n"
        "CRITICAL: output EXACTLY 80 waypoints (indices 0..79).\n\n"
        'Respond ONLY with valid JSON: {"trajectory":[[x0,y0],...,[x79,y79]],"reasoning":"one sentence"}'
    )

    try:
        import google.generativeai as genai
        genai.configure(api_key=_get_api_key())
        last_err = None
        for model_name in GEMINI_MODELS:
            try:
                m = genai.GenerativeModel(
                    model_name,
                    generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
                )
                raw = m.generate_content(prompt).text.strip()
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    raw = match.group(0)
                parsed = json.loads(raw)
                traj   = np.array(parsed["trajectory"], dtype=np.float32)
                if traj.shape[0] > 80:
                    traj = traj[:80]
                elif traj.shape[0] < 80:
                    traj = np.concatenate([traj, np.tile(traj[-1:], (80 - traj.shape[0], 1))])
                print(f"[OK] {model_name}")
                print(f"[Gemini] {parsed.get('reasoning', '')}")
                return traj, raw
            except Exception as e:
                last_err = e
                print(f"[FAIL] {model_name}: {e}")
        return None, str(last_err)
    except Exception as e:
        return None, str(e)
