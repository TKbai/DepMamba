import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceSelector(nn.Module):
    """
    输入:  x ∈ (B, D, L)  —— Batch, Dim, Time
    输出:  gate ∈ (B, 1, L) —— 每个时间步的保留概率
    """
    def __init__(self, d_in, d_hidden=128, tau=1.0, hard=False):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, d_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(d_hidden, 2, kernel_size=1)  # 2 类：keep / drop

        self.tau = tau
        self.hard = hard

    def forward(self, x):
        # x: (B, D, L)
        h = self.relu(self.conv1(x))       # (B, d_hidden, L)
        logits = self.conv2(h)             # (B, 2, L)

        # 变成 (B, L, 2) 以便在最后一维上做 softmax
        logits_t = logits.permute(0, 2, 1)  # (B, L, 2)

        # Gumbel-Softmax
        y = F.gumbel_softmax(
            logits_t, tau=self.tau, hard=self.hard, dim=-1
        )                                   # (B, L, 2)

        # 取“keep”那一列作为 gate 概率
        gate = y[..., 1]                    # (B, L)
        gate = gate.unsqueeze(1)            # (B, 1, L)

        return gate, logits