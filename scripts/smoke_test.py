"""
더미 데이터로 RiskConditionedModel forward + loss 동작 확인.

Run:
    cd waymo_traj
    python smoke_test.py
"""

import math, os, sys
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from src.models.motion_model import RiskConditionedModel
from src.eval.metrics import compute_minADE_FDE, compute_MR

B         = 2
T_HIST    = 11
N_SOCIAL  = 31
N_MAP     = 50
N_MAP_PTS = 10
N_TRAF    = 6
AGENT_DIM = 10
MAP_DIM   = 6
D_MODEL   = 128
K         = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

torch.manual_seed(42)
ego_hist      = torch.randn(B, T_HIST,   AGENT_DIM,           device=device)
social_agents = torch.randn(B, N_SOCIAL, T_HIST,   AGENT_DIM, device=device)
map_scene     = torch.randn(B, N_MAP,    N_MAP_PTS, MAP_DIM,   device=device)
traf          = torch.randn(B, N_TRAF,   1,                    device=device)
risk_label    = torch.randint(0, 2, (B, 3), device=device).float()
gt_traj       = torch.randn(B, 80, 2, device=device)
gt_valid      = torch.ones(B, 80, dtype=torch.bool, device=device)

# ── [1] use_risk_prefix=True (기본) ──────────────────────────────────────────
model = RiskConditionedModel(
    agent_dim=AGENT_DIM, map_dim=MAP_DIM,
    d_model=D_MODEL, K=K, n_layers=2,
    use_lane_mamba=True, use_risk_prefix=True,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"파라미터 수 (risk_prefix=True):  {n_params:,}")

model.train()
out = model(ego_hist, social_agents, map_scene, traf,
            risk_label=risk_label, gt_traj=gt_traj, tf_prob=1.0)

assert out["trajectory"].shape == (B, K, 80, 2), f"traj shape 오류: {out['trajectory'].shape}"
assert out["risk_logits"] is None, "use_risk_prefix=True이면 risk_logits=None 이어야 함"
assert out["lane_prob"].shape == (B, N_MAP),      f"lane_prob shape 오류: {out['lane_prob'].shape}"
print(f"[1] forward (train, risk_prefix=True) 통과  "
      f"trajectory={out['trajectory'].shape}  lane_prob={out['lane_prob'].shape}")

# Laplace NLL loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
pred_traj = out["trajectory"][0]   # [K, 80, 2]
valid_idx = gt_valid[0].nonzero(as_tuple=True)[0]
traj_losses = torch.stack([
    F.l1_loss(pred_traj[k][valid_idx], gt_traj[0][valid_idx]) for k in range(K)
])
b_scale = 1.0
loss = math.log(K) - torch.logsumexp(-traj_losses / b_scale, dim=0)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"[2] Laplace NLL loss={loss.item():.4f}  backward 통과")

# 추론 (risk_label 항상 전달)
model.eval()
with torch.no_grad():
    out_infer = model(ego_hist, social_agents, map_scene, traf, risk_label=risk_label)
assert out_infer["trajectory"].shape == (B, K, 80, 2)
print(f"[3] forward (infer, risk_label 전달) 통과")

# ── [2] use_risk_prefix=False (ablation) ─────────────────────────────────────
model2 = RiskConditionedModel(
    agent_dim=AGENT_DIM, map_dim=MAP_DIM,
    d_model=D_MODEL, K=K, n_layers=2,
    use_lane_mamba=True, use_risk_prefix=False,
).to(device)

n_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"\n파라미터 수 (risk_prefix=False): {n_params2:,}")

model2.train()
out2 = model2(ego_hist, social_agents, map_scene, traf,
              risk_label=risk_label, gt_traj=gt_traj, tf_prob=1.0)

assert out2["trajectory"].shape == (B, K, 80, 2)
assert out2["risk_logits"].shape == (B, 3), f"ablation: risk_logits shape 오류"
print(f"[4] forward (train, risk_prefix=False) 통과  "
      f"risk_logits={out2['risk_logits'].shape}")

# ── [3] 지표 계산 ─────────────────────────────────────────────────────────────
traj_np  = out_infer["trajectory"][0].cpu().numpy()
gt_np    = gt_traj[0].cpu().numpy()
valid_np = gt_valid[0].cpu().numpy().astype(bool)

ade, fde = compute_minADE_FDE(traj_np, gt_np, valid_np)
mr       = compute_MR(traj_np, gt_np, valid_np, threshold=2.0)
print(f"\n[5] 지표  minADE={ade:.3f}m  minFDE={fde:.3f}m  MR={mr:.1f}")

print("\n모든 smoke test 통과")
