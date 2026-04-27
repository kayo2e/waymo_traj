"""Simple trajectory prediction baselines for comparison."""

import numpy as np
import torch
import torch.nn as nn


class ConstantVelocityBaseline:
    """
    Extends the ego velocity (vx, vy) at the last history frame linearly.

    agent_hist : [T_hist, 6]  — ego row from agent_tensor (x,y,vx,vy,cos_h,sin_h)
    Returns    : [80, 2]  ego-relative predicted trajectory
    """

    def predict(self, agent_hist: np.ndarray) -> np.ndarray:
        vx, vy = float(agent_hist[-1, 2]), float(agent_hist[-1, 3])
        steps  = np.arange(1, 81, dtype=np.float32) * 0.1   # 0.1s per step
        traj   = np.stack([vx * steps, vy * steps], axis=1)  # [80, 2]
        return traj


class LSTMBaseline(nn.Module):
    """
    Simple LSTM that maps past (x, y) history to 80-step future trajectory.

    Input : [B, T_hist, 2]  — ego (x, y) from agent_tensor[:, :, :2]
    Output: [B, 80, 2]
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 80 * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)                   # h: [num_layers, B, hidden]
        return self.head(h[-1]).reshape(x.shape[0], 80, 2)
