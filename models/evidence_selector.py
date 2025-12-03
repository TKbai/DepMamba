import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        logits = torch.clamp(logits, min=-10.0, max=10.0)

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

class VideoSelector(nn.Module):
    """
    Slave gate for the visual stream.

    输入:
        x: (B, D, L)，D 是通道维，L 是时间步。
    输出:
        gate:   (B, 1, L) 经过 Sigmoid 的软门
        logits: (B, 1, L) 未归一化的 logit
    """
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_hidden, 1, kernel_size=3, padding=1),
        )

        # 简单初始化
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, D, L)
        """
        logits = self.net(x)          # (B, 1, L)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        gate = torch.sigmoid(logits)  # (B, 1, L)
        return gate, logits