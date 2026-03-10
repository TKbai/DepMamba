import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceSelector(nn.Module):
    """
    Proposal-aware ASG v2 (lightweight)
    输入:
        x: (B, D, L)
    输出:
        gate:   (B, 1, L)
        logits: (B, 2, L)   # 兼容原接口：drop / keep
    额外:
        self.last_aux: 保存 raw score / completeness / final score，供 loss 使用
    """
    def __init__(
        self,
        d_in,
        d_hidden=128,
        tau=1.0,
        hard=False,
        use_gumbel=False,
        win_sizes=(7, 15, 31),
        comp_weight=0.3,
    ):
        super().__init__()

        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel
        self.win_sizes = win_sizes
        self.comp_weight = comp_weight

        # -------- 多尺度时序建模 --------
        self.branch3 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(d_in, d_hidden - 2 * (d_hidden // 3), kernel_size=15, padding=7)

        self.fuse = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # raw evidence score
        self.score_head = nn.Conv1d(d_hidden, 1, kernel_size=1)

        # keep/drop logits head（兼容旧接口）
        self.logit_head = nn.Conv1d(1, 2, kernel_size=1)

        self.last_aux = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _same_avg_pool(x, k):
        # x: (B,1,L)
        pad = k // 2
        return F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=k, stride=1)

    def _completeness_score(self, raw_score):
        """
        raw_score: (B,1,L)
        轻量 completeness:
        内部窗口均值 - 左右更大上下文均值
        """
        comp_all = []
        for k in self.win_sizes:
            inside = self._same_avg_pool(raw_score, k)

            outer_k = min(2 * k + 1, raw_score.size(-1) if raw_score.size(-1) % 2 == 1 else raw_score.size(-1) - 1)
            outer_k = max(outer_k, k + 2 if (k + 2) % 2 == 1 else k + 3)
            outer = self._same_avg_pool(raw_score, outer_k)

            comp = inside - outer
            comp_all.append(comp)

        comp_score = torch.stack(comp_all, dim=0).mean(dim=0)  # (B,1,L)
        return comp_score

    def forward(self, x):
        # x: (B, D, L)
        h3 = self.branch3(x)
        h7 = self.branch7(x)
        h15 = self.branch15(x)

        h = torch.cat([h3, h7, h15], dim=1)      # (B, d_hidden, L)
        h = self.fuse(h)

        raw_score = self.score_head(h)           # (B,1,L)
        raw_score = torch.clamp(raw_score, min=-10.0, max=10.0)

        comp_score = self._completeness_score(raw_score)   # (B,1,L)
        comp_score = torch.clamp(comp_score, min=-10.0, max=10.0)

        final_score = raw_score + self.comp_weight * comp_score
        final_score = torch.clamp(final_score, min=-10.0, max=10.0)

        # 兼容原接口：生成 2 类 logits
        logits = self.logit_head(final_score)    # (B,2,L)
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        logits_t = logits.permute(0, 2, 1)       # (B,L,2)

        if self.use_gumbel:
            probs = F.gumbel_softmax(
                logits_t, tau=self.tau, hard=self.hard, dim=-1
            )
        else:
            probs = F.softmax(logits_t, dim=-1)

        gate = probs[..., 1].unsqueeze(1)        # (B,1,L)

        # 保存辅助量，供 main.py 做 top-k MIL / completeness loss
        self.last_aux = {
            "raw_score": raw_score,      # (B,1,L)
            "comp_score": comp_score,    # (B,1,L)
            "final_score": final_score,  # (B,1,L)
        }

        return gate, logits


class VideoSelector(nn.Module):
    # 保留占位，当前主路径不用
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