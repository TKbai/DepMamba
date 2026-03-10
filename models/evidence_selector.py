import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceSelector(nn.Module):
    """
    输入:
        x: (B, D, L)
    输出:
        gate:   (B, 1, L)
        logits: (B, 2, L)
    """
    def __init__(self, d_in, d_hidden=128, tau=1.0, hard=False, use_gumbel=False):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, d_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(d_hidden, 2, kernel_size=1)  # drop / keep

        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel

        nn.init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # x: (B, D, L)
        h = self.relu(self.conv1(x))
        logits = self.conv2(h)                    # (B, 2, L)
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        logits_t = logits.permute(0, 2, 1)       # (B, L, 2)

        if self.use_gumbel:
            probs = F.gumbel_softmax(
                logits_t, tau=self.tau, hard=self.hard, dim=-1
            )                                     # (B, L, 2)
        else:
            probs = F.softmax(logits_t, dim=-1)   # 更稳定

        gate = probs[..., 1].unsqueeze(1)         # (B, 1, L)
        return gate, logits


# 这版新 DepMamba 已经不再需要 VideoSelector 了
# 你可以先保留这个类不删，但主路径里不要再用它
class VideoSelector(nn.Module):
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_hidden, 1, kernel_size=3, padding=1),
        )

        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        gate = torch.sigmoid(logits)
        return gate, logits