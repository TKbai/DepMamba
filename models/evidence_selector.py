import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceSelector(nn.Module):
    """
    Proposal-aware ASG v2 / Native Proposal Selector v1

    输入:
        x: (B, D, L)
        padding_mask: (B, L), 1=valid, 0=pad

    输出:
        gate:   (B, 1, L)
        logits: (B, 2, L)

    last_aux 中保留:
        raw_score / comp_score / final_score / valid_mask
    并新增:
        proposal_scores / proposal_spans / proposal_valid_mask / proposal_scale_ids / proposal_feats
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

        # proposal/window selector
        proposal_win_sizes=(8, 16, 32),
        proposal_stride_ratio=0.5,
        proposal_comp_weight=0.3,
        max_proposals=None,
        enable_proposals=False,
        build_proposals_in_train=False,

        # native proposal selector
        use_native_proposals=True,
        proposal_hidden_dim=None,
        proposal_score_type="linear",

        # proposal -> timeline projection
        use_projected_proposal_as_main_score=True,
        proposal_project_mode="max",
        proposal_project_blend=1.0,

        
    ):
        super().__init__()

        self.tau = tau
        self.hard = hard
        self.use_gumbel = use_gumbel
        self.win_sizes = tuple(win_sizes)
        self.comp_weight = comp_weight
        self.eps = eps

        self.proposal_win_sizes = tuple(proposal_win_sizes)
        self.proposal_stride_ratio = proposal_stride_ratio
        self.proposal_comp_weight = proposal_comp_weight
        self.max_proposals = max_proposals
        self.enable_proposals = enable_proposals
        self.build_proposals_in_train = build_proposals_in_train

        self.use_native_proposals = use_native_proposals
        self.proposal_hidden_dim = d_hidden if proposal_hidden_dim is None else proposal_hidden_dim
        self.proposal_score_type = proposal_score_type

        self.use_projected_proposal_as_main_score = use_projected_proposal_as_main_score
        self.proposal_project_mode = proposal_project_mode
        self.proposal_project_blend = proposal_project_blend

        # -------- 多尺度时序建模 --------
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
            nn.ReLU(inplace=False),
        )

        # -------- point-wise evidence --------
        self.score_head = nn.Conv1d(d_hidden, 1, kernel_size=1)

        # -------- native proposal heads --------
        if self.proposal_hidden_dim == d_hidden:
            self.proposal_proj = nn.Identity()
        else:
            self.proposal_proj = nn.Linear(d_hidden, self.proposal_hidden_dim)

        self.proposal_score_heads = nn.ModuleList()
        for _ in self.proposal_win_sizes:
            if self.proposal_score_type == "linear":
                self.proposal_score_heads.append(
                    nn.Linear(self.proposal_hidden_dim, 1)
                )
            else:
                self.proposal_score_heads.append(
                    nn.Linear(self.proposal_hidden_dim, 1)
                )

        self.last_aux = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
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
        """
        if valid_mask is None:
            pad = k // 2
            return F.avg_pool1d(
                F.pad(x, (pad, pad), mode="replicate"),
                kernel_size=k,
                stride=1,
            )

        pad = k // 2

        x_pad = F.pad(x * valid_mask, (pad, pad), mode="constant", value=0.0)
        m_pad = F.pad(valid_mask, (pad, pad), mode="constant", value=0.0)

        num = F.avg_pool1d(x_pad, kernel_size=k, stride=1) * k
        den = F.avg_pool1d(m_pad, kernel_size=k, stride=1) * k

        return num / den.clamp(min=self.eps)

    def _completeness_score(self, raw_score, valid_mask=None):
        """
        raw_score:  (B,1,L)
        valid_mask: (B,1,L) or None
        """
        comp_all = []
        L = raw_score.size(-1)

        for k in self.win_sizes:
            inside = self._masked_same_avg_pool(raw_score, valid_mask, k)

            outer_k = min(2 * k + 1, L if L % 2 == 1 else max(1, L - 1))
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

    def _enumerate_spans(self, valid_len, win_size):
        """
        生成 [start, end) span
        """
        if valid_len <= 0:
            return []

        win = min(int(win_size), int(valid_len))
        if win <= 0:
            return []

        stride = max(1, int(round(win * self.proposal_stride_ratio)))

        starts = list(range(0, max(1, valid_len - win + 1), stride))
        last_start = max(0, valid_len - win)

        if len(starts) == 0 or starts[-1] != last_start:
            starts.append(last_start)

        return [(s, s + win) for s in starts]

    def _segment_mean_1d(self, score_1d, mask_1d, start, end):
        """
        score_1d: (L,)
        mask_1d:  (L,) or None
        """
        seg = score_1d[start:end]
        if seg.numel() == 0:
            return score_1d.new_zeros(())

        if mask_1d is None:
            return seg.mean()

        m = mask_1d[start:end].float()
        den = m.sum()
        if den.item() < 0.5:
            return score_1d.new_zeros(())

        return (seg * m).sum() / den.clamp(min=self.eps)

    def _proposal_completeness_1d(self, score_1d, mask_1d, start, end, valid_len):
        """
        fallback proposal completeness:
        inner mean - surrounding mean
        """
        win = end - start
        inner = self._segment_mean_1d(score_1d, mask_1d, start, end)

        left_s = max(0, start - win)
        left_e = start
        right_s = end
        right_e = min(valid_len, end + win)

        outer_parts = []
        outer_masks = []

        if left_e > left_s:
            outer_parts.append(score_1d[left_s:left_e])
            if mask_1d is not None:
                outer_masks.append(mask_1d[left_s:left_e].float())

        if right_e > right_s:
            outer_parts.append(score_1d[right_s:right_e])
            if mask_1d is not None:
                outer_masks.append(mask_1d[right_s:right_e].float())

        if len(outer_parts) == 0:
            return score_1d.new_zeros(())

        outer = torch.cat(outer_parts, dim=0)

        if mask_1d is None:
            outer_mean = outer.mean()
        else:
            m = torch.cat(outer_masks, dim=0)
            den = m.sum()
            if den.item() < 0.5:
                return score_1d.new_zeros(())
            outer_mean = (outer * m).sum() / den.clamp(min=self.eps)

        return inner - outer_mean

    def _build_proposals_from_score(self, final_score, valid_mask=None):
        """
        旧路径：从 final_score 派生 proposal（fallback）
        """
        B, _, L = final_score.shape
        device = final_score.device

        batch_scores = []
        batch_spans = []
        batch_scale_ids = []
        counts = []

        for b in range(B):
            score_1d = final_score[b, 0]
            mask_1d = valid_mask[b, 0] if valid_mask is not None else None

            if mask_1d is None:
                valid_len = L
            else:
                valid_len = int(mask_1d.sum().item())

            sample_scores = []
            sample_spans = []
            sample_scale_ids = []

            if valid_len > 0:
                for scale_id, win_size in enumerate(self.proposal_win_sizes):
                    spans = self._enumerate_spans(valid_len, win_size)

                    for start, end in spans:
                        inner_mean = self._segment_mean_1d(score_1d, mask_1d, start, end)
                        prop_comp = self._proposal_completeness_1d(
                            score_1d, mask_1d, start, end, valid_len
                        )
                        prop_score = inner_mean + self.proposal_comp_weight * prop_comp
                        prop_score = torch.clamp(prop_score, min=-10.0, max=10.0)

                        sample_scores.append(prop_score)
                        sample_spans.append((start, end))
                        sample_scale_ids.append(scale_id)

            if self.max_proposals is not None and len(sample_scores) > self.max_proposals:
                score_tensor = torch.stack(sample_scores, dim=0)
                top_idx = torch.topk(score_tensor, k=self.max_proposals, dim=0).indices
                top_idx_list = sorted(top_idx.tolist(), key=lambda i: sample_spans[i][0])

                sample_scores = [sample_scores[i] for i in top_idx_list]
                sample_spans = [sample_spans[i] for i in top_idx_list]
                sample_scale_ids = [sample_scale_ids[i] for i in top_idx_list]

            batch_scores.append(sample_scores)
            batch_spans.append(sample_spans)
            batch_scale_ids.append(sample_scale_ids)
            counts.append(len(sample_scores))

        n_max = max(1, max(counts) if len(counts) > 0 else 1)

        proposal_scores = final_score.new_zeros(B, n_max)
        proposal_spans = torch.zeros(B, n_max, 2, device=device, dtype=torch.long)
        proposal_valid_mask = torch.zeros(B, n_max, device=device, dtype=torch.bool)
        proposal_scale_ids = torch.zeros(B, n_max, device=device, dtype=torch.long)

        for b in range(B):
            n = counts[b]
            if n > 0:
                proposal_scores[b, :n] = torch.stack(batch_scores[b], dim=0)
                proposal_valid_mask[b, :n] = True
                proposal_spans[b, :n] = torch.tensor(batch_spans[b], device=device, dtype=torch.long)
                proposal_scale_ids[b, :n] = torch.tensor(batch_scale_ids[b], device=device, dtype=torch.long)

        return proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, None

    def _build_native_proposals_from_hidden(self, h, valid_mask=None):
        """
        原生 proposal selector:
        直接从隐藏特征 h 构 proposal，而不是从 final_score 派生

        Args:
            h: (B, C, L)
            valid_mask: (B, 1, L) or None

        Returns:
            proposal_scores:     (B, N_max)
            proposal_spans:      (B, N_max, 2)
            proposal_valid_mask: (B, N_max) bool
            proposal_scale_ids:  (B, N_max) long
            proposal_feats:      (B, N_max, proposal_hidden_dim)
        """
        B, C, L = h.shape
        device = h.device

        batch_scores = []
        batch_spans = []
        batch_scale_ids = []
        batch_feats = []
        counts = []

        for b in range(B):
            hb = h[b]  # (C, L)
            mask_1d = valid_mask[b, 0] if valid_mask is not None else None

            if mask_1d is None:
                valid_len = L
            else:
                valid_len = int(mask_1d.sum().item())

            sample_scores = []
            sample_spans = []
            sample_scale_ids = []
            sample_feats = []

            if valid_len > 0:
                for scale_id, win_size in enumerate(self.proposal_win_sizes):
                    spans = self._enumerate_spans(valid_len, win_size)

                    for start, end in spans:
                        seg = hb[:, start:end]  # (C, win)

                        if seg.numel() == 0:
                            continue

                        if mask_1d is not None:
                            seg_mask = mask_1d[start:end].float()
                            den = seg_mask.sum()
                            if den.item() < 0.5:
                                continue
                            pooled = (seg * seg_mask.unsqueeze(0)).sum(dim=1) / den.clamp(min=self.eps)
                        else:
                            pooled = seg.mean(dim=1)

                        pooled_proj = self.proposal_proj(pooled.unsqueeze(0)).squeeze(0)
                        score = self.proposal_score_heads[scale_id](
                            pooled_proj.unsqueeze(0)
                        ).squeeze(0).squeeze(-1)
                        score = torch.clamp(score, min=-10.0, max=10.0)

                        sample_scores.append(score)
                        sample_spans.append((start, end))
                        sample_scale_ids.append(scale_id)
                        sample_feats.append(pooled_proj)

            if self.max_proposals is not None and len(sample_scores) > self.max_proposals:
                score_tensor = torch.stack(sample_scores, dim=0)
                top_idx = torch.topk(score_tensor, k=self.max_proposals, dim=0).indices
                top_idx_list = sorted(top_idx.tolist(), key=lambda i: sample_spans[i][0])

                sample_scores = [sample_scores[i] for i in top_idx_list]
                sample_spans = [sample_spans[i] for i in top_idx_list]
                sample_scale_ids = [sample_scale_ids[i] for i in top_idx_list]
                sample_feats = [sample_feats[i] for i in top_idx_list]

            batch_scores.append(sample_scores)
            batch_spans.append(sample_spans)
            batch_scale_ids.append(sample_scale_ids)
            batch_feats.append(sample_feats)
            counts.append(len(sample_scores))

        n_max = max(1, max(counts) if len(counts) > 0 else 1)

        proposal_scores = h.new_zeros(B, n_max)
        proposal_spans = torch.zeros(B, n_max, 2, device=device, dtype=torch.long)
        proposal_valid_mask = torch.zeros(B, n_max, device=device, dtype=torch.bool)
        proposal_scale_ids = torch.zeros(B, n_max, device=device, dtype=torch.long)
        proposal_feats = h.new_zeros(B, n_max, self.proposal_hidden_dim)

        for b in range(B):
            n = counts[b]
            if n > 0:
                proposal_scores[b, :n] = torch.stack(batch_scores[b], dim=0)
                proposal_valid_mask[b, :n] = True
                proposal_spans[b, :n] = torch.tensor(batch_spans[b], device=device, dtype=torch.long)
                proposal_scale_ids[b, :n] = torch.tensor(batch_scale_ids[b], device=device, dtype=torch.long)
                proposal_feats[b, :n] = torch.stack(batch_feats[b], dim=0)

        return proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, proposal_feats


    def _project_proposals_to_timeline(
        self,
        proposal_scores,
        proposal_spans,
        proposal_valid_mask,
        seq_len,
        valid_mask=None,
    ):
        """
        将 proposal 分数投影回时间轴（无原地写版本，支持梯度）

        Args:
            proposal_scores:     (B, N)
            proposal_spans:      (B, N, 2)
            proposal_valid_mask: (B, N) bool
            seq_len:             int
            valid_mask:          (B,1,L) or None

        Returns:
            projected_score: (B,1,L)
            covered_mask:    (B,1,L) bool
        """
        B, N = proposal_scores.shape
        device = proposal_scores.device

        proj_list = []
        cover_list = []

        time_index = torch.arange(seq_len, device=device).view(1, seq_len)  # (1,T)

        for b in range(B):
            valid_idx = torch.nonzero(proposal_valid_mask[b], as_tuple=False).squeeze(-1)

            if valid_idx.numel() == 0:
                proj_b = proposal_scores.new_zeros(seq_len)
                cover_b = torch.zeros(seq_len, device=device, dtype=torch.bool)
            else:
                scores_b = proposal_scores[b, valid_idx]        # (M,)
                spans_b = proposal_spans[b, valid_idx]          # (M,2)

                starts = spans_b[:, 0].clamp(min=0, max=seq_len).view(-1, 1)  # (M,1)
                ends = spans_b[:, 1].clamp(min=0, max=seq_len).view(-1, 1)    # (M,1)

                cover = (time_index >= starts) & (time_index < ends)           # (M,T)

                if valid_mask is not None:
                    vm = valid_mask[b, 0].bool().view(1, seq_len)              # (1,T)
                    cover = cover & vm

                cover_b = cover.any(dim=0)                                     # (T,)

                if self.proposal_project_mode == "mean":
                    cover_f = cover.float()                                    # (M,T)
                    num = (scores_b.view(-1, 1) * cover_f).sum(dim=0)          # (T,)
                    den = cover_f.sum(dim=0).clamp(min=1.0)                    # (T,)
                    proj_b = num / den
                    proj_b = torch.where(cover_b, proj_b, torch.zeros_like(proj_b))
                else:
                    # max projection
                    neg_fill = torch.full(
                        (scores_b.numel(), seq_len),
                        -10.0,
                        device=device,
                        dtype=proposal_scores.dtype,
                    )
                    score_map = scores_b.view(-1, 1).expand(-1, seq_len)       # (M,T)
                    masked_score_map = torch.where(cover, score_map, neg_fill)
                    proj_b = masked_score_map.max(dim=0).values
                    proj_b = torch.where(cover_b, proj_b, torch.zeros_like(proj_b))

            proj_list.append(proj_b)
            cover_list.append(cover_b)

        projected_score = torch.stack(proj_list, dim=0).unsqueeze(1)  # (B,1,T)
        covered_mask = torch.stack(cover_list, dim=0).unsqueeze(1)    # (B,1,T)

        if valid_mask is not None:
            projected_score = projected_score * valid_mask
            covered_mask = covered_mask & valid_mask.bool()

        projected_score = torch.clamp(projected_score, min=-10.0, max=10.0)
        return projected_score, covered_mask

    def forward(self, x, padding_mask=None):
        """
        x: (B, D, L)
        padding_mask: (B, L), 1=valid, 0=pad
        """
        valid_mask = self._to_valid_mask(padding_mask, x)  # (B,1,L) or None

        if valid_mask is not None:
            x = x * valid_mask

        # -------- hidden features --------
        h3 = self.branch3(x)
        h7 = self.branch7(x)
        h15 = self.branch15(x)

        h = torch.cat([h3, h7, h15], dim=1)  # (B, d_hidden, L)
        h = self.fuse(h)

        if valid_mask is not None:
            h = h * valid_mask

        # -------- point-wise evidence path (兼容旧主干) --------
        raw_score = self.score_head(h)  # (B,1,L)
        raw_score = torch.clamp(raw_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            raw_score = raw_score * valid_mask

        comp_score = self._completeness_score(raw_score, valid_mask=valid_mask)
        comp_score = torch.clamp(comp_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            comp_score = comp_score * valid_mask

        point_final_score = raw_score + self.comp_weight * comp_score
        point_final_score = torch.clamp(point_final_score, min=-10.0, max=10.0)

        if valid_mask is not None:
            point_final_score = point_final_score * valid_mask

        # -------- proposal path --------
        proposal_scores = None
        proposal_spans = None
        proposal_valid_mask = None
        proposal_scale_ids = None
        proposal_feats = None

        need_build_proposals = (
            self.enable_proposals and
            (self.build_proposals_in_train or (not self.training))
        )

        if need_build_proposals:
            # 训练阶段如果真的要用 proposal-wise MIL，就保留梯度；
            # 验证/测试阶段用 no_grad 节省开销
            if self.training and self.build_proposals_in_train:
                if self.use_native_proposals:
                    proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, proposal_feats = \
                        self._build_proposals_from_score(point_final_score, valid_mask=valid_mask)
                else:
                    proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, proposal_feats = \
                        self._build_proposals_from_score(point_final_score, valid_mask=valid_mask)
            else:
                with torch.no_grad():
                    if self.use_native_proposals:
                        proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, proposal_feats = \
                            self._build_proposals_from_score(
                                point_final_score.detach(),
                                valid_mask=valid_mask.detach() if valid_mask is not None else None
                            )
                    else:
                        proposal_scores, proposal_spans, proposal_valid_mask, proposal_scale_ids, proposal_feats = \
                            self._build_proposals_from_score(
                                point_final_score.detach(),
                                valid_mask=valid_mask.detach() if valid_mask is not None else None
                            )

        # -------- proposal -> timeline projection --------
        projected_final_score = None
        projected_covered_mask = None
        main_final_score = point_final_score

        if (
            self.use_projected_proposal_as_main_score
            and proposal_scores is not None
            and proposal_spans is not None
            and proposal_valid_mask is not None
        ):
            projected_final_score, projected_covered_mask = self._project_proposals_to_timeline(
                proposal_scores,
                proposal_spans,
                proposal_valid_mask,
                seq_len=x.size(-1),
                valid_mask=valid_mask,
            )

            # proposal 覆盖区域用 proposal 分数，其他区域保留 point-wise 分数
            mixed_score = torch.where(
                projected_covered_mask,
                projected_final_score,
                point_final_score,
            )

            blend = float(self.proposal_project_blend)
            blend = max(0.0, min(1.0, blend))

            main_final_score = (1.0 - blend) * point_final_score + blend * mixed_score
            main_final_score = torch.clamp(main_final_score, min=-10.0, max=10.0)

            if valid_mask is not None:
                main_final_score = main_final_score * valid_mask


        # -------- gate / logits 用 main_final_score（proposal-projected main timeline score）--------
        if self.use_gumbel:
            logits = torch.cat([-main_final_score, main_final_score], dim=1)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            logits_t = logits.permute(0, 2, 1)
            probs = F.gumbel_softmax(
                logits_t,
                tau=self.tau,
                hard=self.hard,
                dim=-1,
            )
            gate = probs[..., 1].unsqueeze(1)
        else:
            gate_prob = torch.sigmoid(main_final_score / max(self.tau, self.eps))

            if self.hard:
                hard_gate = (gate_prob > 0.5).float()
                gate = hard_gate.detach() - gate_prob.detach() + gate_prob
            else:
                gate = gate_prob

            logits = torch.cat([-main_final_score, main_final_score], dim=1)
            logits = torch.clamp(logits, min=-10.0, max=10.0)

        if valid_mask is not None:
            gate = gate * valid_mask
            logits = logits * valid_mask

        self.last_aux = {
            # point-wise path
            "raw_score": raw_score,                       # (B,1,L)
            "comp_score": comp_score,                     # (B,1,L)
            "point_final_score": point_final_score,       # (B,1,L)

            # main timeline score after proposal projection
            "projected_final_score": projected_final_score,   # (B,1,L) or None
            "projected_covered_mask": projected_covered_mask, # (B,1,L) bool or None
            "final_score": main_final_score,                  # (B,1,L)  <- 以后主链统一吃这个
            "valid_mask": valid_mask,                         # (B,1,L) or None

            # proposal-level outputs
            "proposal_scores": proposal_scores,
            "proposal_spans": proposal_spans,
            "proposal_valid_mask": proposal_valid_mask,
            "proposal_scale_ids": proposal_scale_ids,
            "proposal_feats": proposal_feats,

            # debug / analysis
            "hidden_feat": h,
            "uses_native_proposals": self.use_native_proposals,
        }

        return gate, logits


class VideoSelector(nn.Module):
    """
    保留占位，当前主路径不用。
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