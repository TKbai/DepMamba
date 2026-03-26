import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceDiscoveryHead(nn.Module):
    """
    输入:
        x: (B, D, T)
    输出:
        proposal_score: (B, T)
        aux: {"proposal_logit": (B, T)}

    改动点：
    1) 输入 LayerNorm
    2) score logit 做 masked centering
    3) 负 bias 初始化，鼓励 sparse proposal
    4) temperature + clamp，避免 sigmoid 饱和
    """
    def __init__(
        self,
        d_in,
        d_hidden=128,
        dropout=0.1,
        score_temp=2.5,
        logit_clip=8.0,
        init_bias=-2.0,
    ):
        super().__init__()

        self.in_norm = nn.LayerNorm(d_in)

        self.branch3 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(
            d_in,
            d_hidden - 2 * (d_hidden // 3),
            kernel_size=15,
            padding=7,
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.score_head = nn.Conv1d(d_hidden, 1, kernel_size=1, bias=True)

        self.score_temp = score_temp
        self.logit_clip = logit_clip
        self.init_bias = init_bias

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 很关键：给 proposal 一个“默认偏低”的先验
        if self.score_head.bias is not None:
            nn.init.constant_(self.score_head.bias, self.init_bias)

    @staticmethod
    def _masked_center(logit, padding_mask=None):
        # logit: (B, T)
        if padding_mask is None:
            return logit - logit.mean(dim=-1, keepdim=True)

        m = padding_mask.float()
        mean = (logit * m).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        centered = (logit - mean) * m
        return centered

    def forward(self, x, padding_mask=None):
        # x: (B, D, T)
        x = self.in_norm(x.transpose(1, 2)).transpose(1, 2)

        h3 = self.branch3(x)
        h7 = self.branch7(x)
        h15 = self.branch15(x)

        h = torch.cat([h3, h7, h15], dim=1)
        h = self.fuse(h)

        raw_logit = self.score_head(h).squeeze(1)   # (B, T)

        # 先做 sample 内部的相对中心化，避免整段一起变大
        proposal_logit = self._masked_center(raw_logit, padding_mask)

        # 再做温度缩放 + clip，防止进入 sigmoid 饱和区
        proposal_logit = proposal_logit / self.score_temp
        proposal_logit = proposal_logit.clamp(min=-self.logit_clip, max=self.logit_clip)

        if padding_mask is not None:
            proposal_logit = proposal_logit.masked_fill(padding_mask == 0, -1e4)

        proposal_score = torch.sigmoid(proposal_logit)
        if padding_mask is not None:
            proposal_score = proposal_score * padding_mask.float()

        aux = {
            "proposal_logit": proposal_logit,
            "raw_logit": raw_logit,
        }
        return proposal_score, aux