"""
더미 데이터로 RiskConditionedModel forward + multi-task loss 동작 확인.

Run:
    cd waymo_traj
    python smoke_test.py
"""

import os, sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.motion_model import RiskConditionedModel
from src.eval.metrics import compute_minADE_FDE, compute_MR
import numpy as np

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
B        = 2      # 배치 크기
T_HIST   = 11     # 과거 타임스텝
N_SOCIAL = 31     # 주변 에이전트 수
N_MAP    = 56     # 맵 토큰 수 (50 폴리라인 + 6 신호)
AGENT_DIM = 6
MAP_DIM   = 3
D_MODEL  = 128
K        = 6
LAM      = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── 더미 입력 생성 ─────────────────────────────────────────────────────────────
torch.manual_seed(42)
ego_hist      = torch.randn(B, T_HIST,   AGENT_DIM, device=device)
social_agents = torch.randn(B, N_SOCIAL, AGENT_DIM, device=device)
map_tokens    = torch.randn(B, N_MAP,    MAP_DIM,   device=device)
risk_label    = torch.randint(0, 2, (B, 3), device=device).float()  # 이진 GT

gt_traj  = torch.randn(B, 80, 2, device=device)
gt_kp    = torch.randn(B, 3,  2, device=device)
kp_valid = torch.ones(B, 3, dtype=torch.bool, device=device)
gt_valid = torch.ones(B, 80, dtype=torch.bool, device=device)

# ── 모델 초기화 ────────────────────────────────────────────────────────────────
model = RiskConditionedModel(
    agent_dim=AGENT_DIM, map_dim=MAP_DIM,
    d_model=D_MODEL, K=K, n_layers=2
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"파라미터 수: {n_params:,}")

# ── [1] Forward (학습 모드 — risk_label 제공) ─────────────────────────────────
model.train()
out = model(ego_hist, social_agents, map_tokens, risk_label=risk_label)

assert out["keypoints"].shape  == (B, K, 3, 2),  f"keypoints shape 오류: {out['keypoints'].shape}"
assert out["trajectory"].shape == (B, K, 80, 2), f"trajectory shape 오류: {out['trajectory'].shape}"
assert out["risk_logits"].shape == (B, 3),        f"risk_logits shape 오류: {out['risk_logits'].shape}"
print(f"[1] forward (train) 통과  keypoints={out['keypoints'].shape}  "
      f"trajectory={out['trajectory'].shape}  risk_logits={out['risk_logits'].shape}")

# ── [2] Forward (추론 모드 — risk_label=None) ─────────────────────────────────
model.eval()
with torch.no_grad():
    out_infer = model(ego_hist, social_agents, map_tokens, risk_label=None)
assert out_infer["trajectory"].shape == (B, K, 80, 2)
print(f"[2] forward (infer) 통과  risk_label=None → 자체 예측 사용")

# ── [3] Multi-task loss 계산 ──────────────────────────────────────────────────
model.train()
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

batch_losses = []
for b in range(B):
    pred_traj = out["trajectory"][b]   # [K, 80, 2]
    pred_kp   = out["keypoints"][b]    # [K, 3, 2]
    valid_idx = gt_valid[b].nonzero(as_tuple=True)[0]

    # WTA
    with torch.no_grad():
        ades = torch.stack([
            torch.linalg.norm(pred_traj[k][valid_idx] - gt_traj[b][valid_idx], dim=1).mean()
            for k in range(K)
        ])
        best_k = int(ades.argmin())

    kp_loss   = F.huber_loss(pred_kp[best_k][kp_valid[b]], gt_kp[b][kp_valid[b]], delta=2.0)
    traj_loss = F.l1_loss(pred_traj[best_k][valid_idx], gt_traj[b][valid_idx])
    L_risk    = bce(out["risk_logits"][b], risk_label[b])
    loss      = (kp_loss + traj_loss) + LAM * L_risk
    batch_losses.append(loss)

total_loss = torch.stack(batch_losses).mean()
optimizer.zero_grad()
total_loss.backward()
optimizer.step()
print(f"[3] multi-task loss={total_loss.item():.4f}  backward 통과")

# ── [4] 지표 계산 ─────────────────────────────────────────────────────────────
traj_np  = out_infer["trajectory"][0].numpy() if device.type == "cpu" \
           else out_infer["trajectory"][0].cpu().numpy()
gt_np    = gt_traj[0].cpu().numpy()
valid_np = gt_valid[0].cpu().numpy().astype(bool)

ade, fde = compute_minADE_FDE(traj_np, gt_np, valid_np)
mr       = compute_MR(traj_np, gt_np, valid_np, threshold=2.0)
print(f"[4] minADE={ade:.3f}m  minFDE={fde:.3f}m  MR={mr:.1f}")

print("\n✓ 모든 smoke test 통과")
