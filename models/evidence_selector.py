import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceSelector(nn.Module):
    """
    Proposal-aware ASG v2 (cleaned, mask-aware)
    输入:
        x: (B, D, L)
        padding_mask: (B, L), 1=valid, 0=pad
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
        eps=1e-6,
    ):
        super().__init__()

        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel
        self.win_sizes = win_sizes
        self.comp_weight = comp_weight
        self.eps = eps

        # -------- 多尺度时序建模 --------
        self.branch3 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=3, padding=1)
        self.branch7 = nn.Conv1d(d_in, d_hidden // 3, kernel_size=7, padding=3)
        self.branch15 = nn.Conv1d(
            d_in, d_hidden - 2 * (d_hidden // 3), kernel_size=15, padding=7
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # raw evidence score
        self.score_head = nn.Conv1d(d_hidden, 1, kernel_size=1)

        # 兼容旧接口：保留 logits 输出，但不再额外学习一个独立 gate 头
        self.last_aux = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _to_valid_mask(padding_mask, x):
        """
        padding_mask: (B, L) -> (B, 1, L)
        """
        if padding_mask is None:
            return None
        return padding_mask.unsqueeze(1).float().to(x.device)

    def _masked_same_avg_pool(self, x, valid_mask, k):
        """
        x:          (B,1,L)
        valid_mask: (B,1,L) or None
        返回长度不变的 masked avg pooling
        """
        if valid_mask is None:
            pad = k // 2
            return F.avg_pool1d(
                F.pad(x, (pad, pad), mode="replicate"),
                kernel_size=k,
                stride=1,
            )

        pad = k // 2

        # 用 0 padding，靠 mask 做有效均值
        x_pad = F.pad(x * valid_mask, (pad, pad), mode="constant", value=0.0)
        m_pad = F.pad(valid_mask, (pad, pad), mode="constant", value=0.0)

        # avg_pool * k 等价于 sum_pool
        num = F.avg_pool1d(x_pad, kernel_size=k, stride=1) * k
        den = F.avg_pool1d(m_pad, kernel_size=k, stride=1) * k

        out = num / den.clamp(min=self.eps)
        return out

    def _completeness_score(self, raw_score, valid_mask=None):
        """
        raw_score:  (B,1,L)
        valid_mask: (B,1,L) or None

        轻量 completeness:
        内部窗口均值 - 外部更大上下文均值
        """
        comp_all = []
        L = raw_score.size(-1)

        for k in self.win_sizes:
            inside = self._masked_same_avg_pool(raw_score, valid_mask, k)

            outer_k = min(2 * k + 1, L if L % 2 == 1 else L - 1)
            outer_k = max(outer_k, k + 2 if (k + 2) % 2 == 1 else k + 3)

            if outer_k < 3:
                outer_k = 3
            if outer_k % 2 == 0:
                outer_k += 1
            outer_k = min(outer_k, L if L % 2 == 1 else max(1, L - 1))

            if outer_k < k:
                outer_k = k

            outer = self._masked_same_avg_pool(raw_score, valid_mask, outer_k)
            comp = inside - outer

            if valid_mask is not None:
                comp = comp * valid_mask

            comp_all.append(comp)

        comp_score = torch.stack(comp_all, dim=0).mean(dim=0)  # (B,1,L)
        return comp_score

    def forward(self, x, padding_mask=None):
        """
        x: (B, D, L)
        padding_mask: (B, L), 1=valid, 0=pad
        """
        valid_mask = self._to_valid_mask(padding_mask, x)  # (B,1,L) or None

        if valid_mask is not None:
            x = x * valid_mask

        h3 = self.branch3(x)
        h7 = self.branch7(x)
        h15 = self.branch15(x)

        h = torch.cat([h3, h7, h15], dim=1)  # (B, d_hidden, L)
        h = self.fuse(h)

        if valid_mask is not None:
            h = h * valid_mask

        raw_score = self.score_head(h)  # (B,1,L)
        raw_score = torch.clamp(raw_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            raw_score = raw_score * valid_mask

        comp_score = self._completeness_score(raw_score, valid_mask=valid_mask)
        comp_score = torch.clamp(comp_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            comp_score = comp_score * valid_mask

        final_score = raw_score + self.comp_weight * comp_score
        final_score = torch.clamp(final_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            final_score = final_score * valid_mask

        # -------- 统一 gate 与 final_score --------
        # gate 直接由 final_score 决定，tau 这时真正生效
        if self.use_gumbel:
            logits = torch.cat([-final_score, final_score], dim=1)  # (B,2,L)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            logits_t = logits.permute(0, 2, 1)  # (B,L,2)
            probs = F.gumbel_softmax(
                logits_t,
                tau=self.tau,
                hard=self.hard,
                dim=-1,
            )
            gate = probs[..., 1].unsqueeze(1)  # (B,1,L)
        else:
            gate_prob = torch.sigmoid(final_score / max(self.tau, self.eps))

            if self.hard:
                hard_gate = (gate_prob > 0.5).float()
                gate = hard_gate.detach() - gate_prob.detach() + gate_prob
            else:
                gate = gate_prob

            logits = torch.cat([-final_score, final_score], dim=1)  # (B,2,L)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

        if valid_mask is not None:
            gate = gate * valid_mask
            logits = logits * valid_mask

        self.last_aux = {
            "raw_score": raw_score,       # (B,1,L)
            "comp_score": comp_score,     # (B,1,L)
            "final_score": final_score,   # (B,1,L)
            "valid_mask": valid_mask,     # (B,1,L) or None
        }

        return gate, logits


class VideoSelector(nn.Module):
    """
    保留占位，当前主路径不用。
    这里也补成支持 padding_mask，避免后面切换分支时再返工。
    """
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

    def forward(self, x: torch.Tensor, padding_mask=None):
        logits = self.net(x)
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        gate = torch.sigmoid(logits)

        if padding_mask is not None:
            valid_mask = padding_mask.unsqueeze(1).float().to(x.device)
            logits = logits * valid_mask
            gate = gate * valid_mask

        return gate, logits