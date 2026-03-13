"""
DepMamba v2
-----------
New pipeline:
    modality-specific ASG
    -> unimodal encoders
    -> lightweight cross-attention (MulT-lite)
    -> fusion encoder
    -> classifier

Notes
-----
1) This version intentionally REMOVES:
   - CoSSM
   - master-slave gating
   - gate-controlled delta update
   - ESMamba / ESMMBiMamba

2) This version REUSES:
   - EvidenceSelector
   - BiMamba (from .mamba.bimamba)
   - BaseNet

3) Input convention:
   x.shape == (B, T, D_total)
   x[..., :video_input_size]                 -> video features
   x[..., video_input_size:video+audio]      -> audio features

4) padding_mask convention:
   1 = valid timestep
   0 = padded timestep
"""

import math
import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba as UniMamba
from .mamba.bimamba import Mamba as BiMamba
from .evidence_selector import EvidenceSelector
from .base import BaseNet
def assert_finite(x, name):
    if not torch.isfinite(x).all():
        print(f"[BAD] {name}")
        print("shape:", tuple(x.shape), "dtype:", x.dtype, "device:", x.device)
        print("nan:", torch.isnan(x).any().item(), "inf:", torch.isinf(x).any().item())
        raise RuntimeError(f"{name} has NaN/Inf")

# =========================================================
# Utility blocks
# =========================================================

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, ff_mult=4):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, key_padding_mask=None):
        # pre-norm
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)

        attn_out, _ = self.attn(
            query=qn,
            key=kvn,
            value=kvn,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = q + attn_out
        x = x + self.ffn(self.out_norm(x))
        return x


class CNNEncoderLayer(nn.Module):
    """
    Local temporal modeling block.
    Kept from your original design because it is still useful as a local complement.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.0,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_size, output_size, kernel_size=3, padding=1, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.skip = nn.Conv1d(input_size, output_size, kernel_size=1, bias=False)
        else:
            self.skip = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C_out, T)
        """
        out = self.net(x)
        if self.skip is not None:
            x = self.skip(x)
        out = out + x
        return out


class MambaEncoderLayer(nn.Module):
    """
    Single-modality Mamba encoder layer.
    Uses BiMamba when bidirectional=True, otherwise falls back to UniMamba.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        causal: bool = False,
        mamba_config: Optional[dict] = None,
    ):
        super().__init__()

        cfg = copy.deepcopy(mamba_config) if mamba_config is not None else {}
        bidirectional = cfg.pop("bidirectional", True)

        if causal or (not bidirectional):
            self.mamba = UniMamba(d_model=d_model, **cfg)
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type="v2",
                **cfg,
            )

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
        x_norm = self.norm(x)

        try:
            core_out = self.mamba(x_norm, inference_params=inference_params)
        except TypeError:
            try:
                core_out = self.mamba(x_norm, inference_params)
            except TypeError:
                core_out = self.mamba(x_norm)

        core_out = torch.nan_to_num(core_out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = x + self.drop(core_out)
        return out


class EnSSM(nn.Module):
    """
    Stacked local-CNN + single-modality Mamba encoder.
    This now acts as a unimodal encoder or a post-fusion encoder.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        dropout: float = 0.0,
        causal: bool = False,
        mamba_config: Optional[dict] = None,
    ):
        super().__init__()

        cnn_list = []
        mamba_list = []

        for i, out_dim in enumerate(output_sizes):
            in_dim = input_size if i == 0 else output_sizes[i - 1]

            cnn_list.append(
                CNNEncoderLayer(
                    input_size=in_dim,
                    output_size=out_dim,
                    dropout=dropout,
                )
            )
            mamba_list.append(
                MambaEncoderLayer(
                    d_model=out_dim,
                    dropout=dropout,
                    causal=causal,
                    mamba_config=mamba_config,
                )
            )

        self.cnn_layers = nn.ModuleList(cnn_list)
        self.mamba_layers = nn.ModuleList(mamba_list)

    def forward(
        self,
        x: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
        out = x
        for i, (cnn_layer, mamba_layer) in enumerate(zip(self.cnn_layers, self.mamba_layers)):
            out = cnn_layer(out.permute(0, 2, 1)).permute(0, 2, 1)
            #assert_finite(out, f"EnSSM layer {i} after CNN")

            out = mamba_layer(out, inference_params=inference_params)
            #assert_finite(out, f"EnSSM layer {i} after Mamba")

        return out


# =========================================================
# Main model
# =========================================================

class DepMamba(BaseNet):
    """
    New DepMamba:
        input projection
        -> modality-specific ASG
        -> audio encoder / video encoder
        -> cross-attention
        -> fusion encoder
        -> classifier
    """

    def __init__(
        self,
        audio_input_size: int = 161,
        video_input_size: int = 136,
        mm_input_size: int = 128,
        uni_output_sizes: Optional[List[int]] = None,
        fusion_output_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        causal: bool = False,
        mamba_config: Optional[dict] = None,
        use_gate: bool = True,

        # selector
        selector_tau: float = 1.0,
        selector_hard: bool = False,
        selector_use_gumbel: bool = False,
        selector_alpha: float = 0.5,
        selector_comp_weight: float = 0.3,
        selector_win_sizes: Optional[List[int]] = None,

        selector_use_native_proposals: bool = True,
        selector_proposal_hidden_dim: Optional[int] = None,
        selector_proposal_score_type: str = "linear",

        selector_enable_proposals: bool = False,
        selector_build_proposals_in_train: bool = False,
        selector_proposal_win_sizes: Optional[List[int]] = None,
        selector_proposal_stride_ratio: float = 0.5,
        selector_proposal_comp_weight: float = 0.3,
        selector_max_proposals: Optional[int] = None,

        # proposal-level attention
        proposal_attn_topk_ratio_a: float = 0.2,
        proposal_attn_topk_ratio_v: float = 0.2,
        proposal_attn_min_props_a: int = 2,
        proposal_attn_min_props_v: int = 2,
        proposal_attn_max_props_a: Optional[int] = 8,
        proposal_attn_max_props_v: Optional[int] = 8,
        proposal_pooling: str = "mean",

        # cross-attention
        attn_heads: int = 1,
        use_evidence_attn: bool = True,
        attn_stride: int = 4,

        attn_topk_ratio_a: float = 0.35,
        attn_topk_ratio_v: float = 0.35,
        attn_min_tokens_a: int = 4,
        attn_min_tokens_v: int = 4,
        attn_max_tokens_a: Optional[int] = None,
        attn_max_tokens_v: Optional[int] = None,

        cross_beta_a: float = 0.1,
        cross_beta_v: float = 0.1,

        # final pooling
        final_pooling: str = "mean_plus_proposal",
        proposal_cls_topk_ratio_a: float = 0.2,
        proposal_cls_topk_ratio_v: float = 0.2,
        proposal_cls_min_props_a: int = 2,
        proposal_cls_min_props_v: int = 2,
        proposal_cls_max_props_a: Optional[int] = 6,
        proposal_cls_max_props_v: Optional[int] = 6,
    ):
        super().__init__()

        if uni_output_sizes is None:
            uni_output_sizes = [256, 64]
        if fusion_output_sizes is None:
            fusion_output_sizes = [128]
        if selector_win_sizes is None:
            selector_win_sizes = [7, 15, 31]
        if selector_proposal_win_sizes is None:
            selector_proposal_win_sizes = [8, 16, 32]

        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.mm_input_size = mm_input_size
        self.use_gate = use_gate
        self.selector_alpha = selector_alpha

        self.use_evidence_attn = use_evidence_attn
        self.attn_stride = attn_stride

        self.attn_topk_ratio_a = attn_topk_ratio_a
        self.attn_topk_ratio_v = attn_topk_ratio_v
        self.attn_min_tokens_a = attn_min_tokens_a
        self.attn_min_tokens_v = attn_min_tokens_v
        self.attn_max_tokens_a = attn_max_tokens_a
        self.attn_max_tokens_v = attn_max_tokens_v

        self.cross_beta_a = cross_beta_a
        self.cross_beta_v = cross_beta_v

        self.proposal_attn_topk_ratio_a = proposal_attn_topk_ratio_a
        self.proposal_attn_topk_ratio_v = proposal_attn_topk_ratio_v
        self.proposal_attn_min_props_a = proposal_attn_min_props_a
        self.proposal_attn_min_props_v = proposal_attn_min_props_v
        self.proposal_attn_max_props_a = proposal_attn_max_props_a
        self.proposal_attn_max_props_v = proposal_attn_max_props_v
        self.proposal_pooling = proposal_pooling

        self.final_pooling = final_pooling
        self.proposal_cls_topk_ratio_a = proposal_cls_topk_ratio_a
        self.proposal_cls_topk_ratio_v = proposal_cls_topk_ratio_v
        self.proposal_cls_min_props_a = proposal_cls_min_props_a
        self.proposal_cls_min_props_v = proposal_cls_min_props_v
        self.proposal_cls_max_props_a = proposal_cls_max_props_a
        self.proposal_cls_max_props_v = proposal_cls_max_props_v

        # ---------------------------
        # 1) input projection
        # ---------------------------
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, kernel_size=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, kernel_size=1, bias=False)

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

        # ---------------------------
        # 2) modality-specific ASG
        # ---------------------------
        self.audio_selector = EvidenceSelector(
            d_in=mm_input_size,
            d_hidden=mm_input_size,
            tau=selector_tau,
            hard=selector_hard,
            use_gumbel=selector_use_gumbel,
            win_sizes=selector_win_sizes,
            comp_weight=selector_comp_weight,

            proposal_win_sizes=selector_proposal_win_sizes,
            proposal_stride_ratio=selector_proposal_stride_ratio,
            proposal_comp_weight=selector_proposal_comp_weight,
            max_proposals=selector_max_proposals,
            enable_proposals=selector_enable_proposals,
            build_proposals_in_train=selector_build_proposals_in_train,

            use_native_proposals=selector_use_native_proposals,
            proposal_hidden_dim=selector_proposal_hidden_dim,
            proposal_score_type=selector_proposal_score_type,
        )
        self.video_selector = EvidenceSelector(
            d_in=mm_input_size,
            d_hidden=mm_input_size,
            tau=selector_tau,
            hard=selector_hard,
            use_gumbel=selector_use_gumbel,
            win_sizes=selector_win_sizes,
            comp_weight=selector_comp_weight,

            proposal_win_sizes=selector_proposal_win_sizes,
            proposal_stride_ratio=selector_proposal_stride_ratio,
            proposal_comp_weight=selector_proposal_comp_weight,
            max_proposals=selector_max_proposals,
            enable_proposals=selector_enable_proposals,
            build_proposals_in_train=selector_build_proposals_in_train,

            use_native_proposals=selector_use_native_proposals,
            proposal_hidden_dim=selector_proposal_hidden_dim,
            proposal_score_type=selector_proposal_score_type,
        )

        self.last_audio_gate = None
        self.last_video_gate = None

        # ---------------------------
        # 3) unimodal encoders
        # ---------------------------
        self.audio_encoder = EnSSM(
            input_size=mm_input_size,
            output_sizes=uni_output_sizes,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )
        self.video_encoder = EnSSM(
            input_size=mm_input_size,
            output_sizes=uni_output_sizes,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )

        d_model = uni_output_sizes[-1]

        # ---------------------------
        # downsampling
        # ---------------------------

        self.attn_pool = nn.MaxPool1d(
            kernel_size=self.attn_stride,
            stride=self.attn_stride,
            ceil_mode=True,
        )

        self.audio_from_video = CrossAttentionBlock(
            d_model=d_model,
            nhead=attn_heads,
            dropout=0.0,
        )
        self.video_from_audio = CrossAttentionBlock(
            d_model=d_model,
            nhead=attn_heads,
            dropout=0.0,
        )

        # ---------------------------
        # 5) post-fusion encoder
        # ---------------------------
        self.fusion_encoder = EnSSM(
            input_size=d_model * 2,
            output_sizes=fusion_output_sizes,
            dropout=dropout,
            causal=causal,
            mamba_config=mamba_config,
        )

        # ---------------------------
        # 6) classifier
        # ---------------------------
        self.pool = nn.AdaptiveMaxPool1d(1)

        cls_in_dim = fusion_output_sizes[-1]
        if self.final_pooling == "mean_plus_proposal":
            cls_in_dim = fusion_output_sizes[-1] * 2

        self.output = nn.Linear(cls_in_dim, 1)

    # -----------------------------------------------------
    # helper functions
    # -----------------------------------------------------
    def _split_modalities(self, x: torch.Tensor):
        """
        x: (B, T, D_total)
        assumes layout [video, audio]
        """
        total_needed = self.video_input_size + self.audio_input_size
        if x.size(-1) < total_needed:
            raise ValueError(
                f"Input feature dim = {x.size(-1)} is smaller than "
                f"video_input_size + audio_input_size = {total_needed}."
            )

        xv = x[:, :, :self.video_input_size]
        xa = x[:, :, self.video_input_size:self.video_input_size + self.audio_input_size]
        return xa, xv

    def _apply_selector(
        self,
        x: torch.Tensor,
        selector: nn.Module,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, T, D)
            selector expects (B, D, T)
        Returns:
            x_reweighted: (B, T, D)
            gate: (B, 1, T)
            logits: (B, 2, T)
        """
        gate, logits = selector(
            x.permute(0, 2, 1),
            padding_mask=padding_mask,
        )  # gate: (B,1,T)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(1).float()  # (B,1,T)
            gate = gate * valid
            logits = logits * valid

        gate_t = gate.permute(0, 2, 1)  # (B,T,1)

        alpha = self.selector_alpha
        x = x * (alpha + (1.0 - alpha) * gate_t)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()

        return x, gate, logits

    def _masked_mean_pool(self, x: torch.Tensor, padding_mask: torch.Tensor):
        """
        x: (B, T, D)
        padding_mask: (B, T), 1=valid, 0=pad
        """
        mask = padding_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        x = (x * mask).sum(dim=1) / denom
        return x
    


    def _select_topk_tokens(
        self,
        x_ds: torch.Tensor,
        score_ds: torch.Tensor,
        valid_mask_ds: torch.Tensor,
        topk_ratio: float,
        min_tokens: int,
        max_tokens: Optional[int] = None,
    ):
        """
        x_ds:         (B, T_ds, D)
        score_ds:     (B, 1, T_ds)
        valid_mask_ds:(B, T_ds) bool

        返回:
            sel_x:    (B, K_max, D)
            sel_valid:(B, K_max) bool
            sel_idx:  (B, K_max) long
        """
        B, T_ds, D = x_ds.shape
        device = x_ds.device

        idx_list = []
        k_list = []
        max_k = 1

        for b in range(B):
            valid_idx = torch.nonzero(valid_mask_ds[b], as_tuple=False).squeeze(-1)

            if valid_idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=device)
                k = 0
            else:
                cur_scores = score_ds[b, 0, valid_idx]  # (valid_len,)
                valid_len = cur_scores.numel()

                k = max(min_tokens, int(math.ceil(valid_len * topk_ratio)))
                k = min(k, valid_len)

                if max_tokens is not None:
                    k = min(k, max_tokens)

                k = max(1, k)

                topk_local = torch.topk(cur_scores, k=k, dim=-1).indices
                idx = valid_idx[topk_local]

                # 恢复时间顺序，而不是按分数顺序
                idx = torch.sort(idx).values

            idx_list.append(idx)
            k_list.append(k)
            max_k = max(max_k, max(1, k))

        sel_x = torch.zeros(B, max_k, D, device=device, dtype=x_ds.dtype)
        sel_valid = torch.zeros(B, max_k, device=device, dtype=torch.bool)
        sel_idx = torch.zeros(B, max_k, device=device, dtype=torch.long)

        for b in range(B):
            k = k_list[b]
            if k > 0:
                idx = idx_list[b]
                sel_x[b, :k] = x_ds[b, idx]
                sel_valid[b, :k] = True
                sel_idx[b, :k] = idx

        return sel_x, sel_valid, sel_idx


    def _scatter_selected_tokens(
        self,
        base_shape_tensor: torch.Tensor,
        updates: torch.Tensor,
        sel_idx: torch.Tensor,
        sel_valid: torch.Tensor,
    ):
        """
        把 top-k attention 输出 scatter 回原下采样时间轴
        base_shape_tensor: 仅用于提供目标 shape，通常传 xa_ds / xv_ds
        updates:  (B, K_max, D)
        sel_idx:  (B, K_max)
        sel_valid:(B, K_max)
        返回:
            out: (B, T_ds, D)
        """
        out = torch.zeros_like(base_shape_tensor)

        B = out.size(0)
        for b in range(B):
            k = int(sel_valid[b].sum().item())
            if k > 0:
                out[b, sel_idx[b, :k]] = updates[b, :k]

        return out

    def _select_topk_proposals(
        self,
        proposal_scores: torch.Tensor,
        proposal_valid_mask: torch.Tensor,
        topk_ratio: float,
        min_props: int,
        max_props: Optional[int] = None,
    ):
        """
        proposal_scores:     (B, N)
        proposal_valid_mask: (B, N) bool

        返回:
            sel_idx:   (B, K_max) long
            sel_valid: (B, K_max) bool
        """
        B, N = proposal_scores.shape
        device = proposal_scores.device

        idx_list = []
        k_list = []
        max_k = 1

        for b in range(B):
            valid_idx = torch.nonzero(proposal_valid_mask[b], as_tuple=False).squeeze(-1)

            if valid_idx.numel() == 0:
                idx = torch.zeros(1, dtype=torch.long, device=device)
                k = 0
            else:
                cur_scores = proposal_scores[b, valid_idx]
                valid_len = cur_scores.numel()

                k = max(min_props, int(math.ceil(valid_len * topk_ratio)))
                k = min(k, valid_len)

                if max_props is not None:
                    k = min(k, max_props)

                k = max(1, k)

                topk_local = torch.topk(cur_scores, k=k, dim=-1).indices
                idx = valid_idx[topk_local]

                idx = torch.sort(idx).values

            idx_list.append(idx)
            k_list.append(k)
            max_k = max(max_k, max(1, k))

        sel_idx = torch.zeros(B, max_k, device=device, dtype=torch.long)
        sel_valid = torch.zeros(B, max_k, device=device, dtype=torch.bool)

        for b in range(B):
            k = k_list[b]
            if k > 0:
                sel_idx[b, :k] = idx_list[b]
                sel_valid[b, :k] = True

        return sel_idx, sel_valid

    def _pool_proposals(
        self,
        x: torch.Tensor,
        proposal_spans: torch.Tensor,
        sel_idx: torch.Tensor,
        sel_valid: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        x:            (B, T, D)
        proposal_spans:(B, N, 2)  [start, end)
        sel_idx:      (B, K_max)
        sel_valid:    (B, K_max) bool
        padding_mask: (B, T) or None

        返回:
            prop_feat:  (B, K_max, D)
            prop_valid: (B, K_max) bool
            prop_spans: (B, K_max, 2)
        """
        B, T, D = x.shape
        K_max = sel_idx.size(1)
        device = x.device

        prop_feat = torch.zeros(B, K_max, D, device=device, dtype=x.dtype)
        prop_valid = sel_valid.clone()
        prop_spans = torch.zeros(B, K_max, 2, device=device, dtype=torch.long)

        for b in range(B):
            for j in range(K_max):
                if not sel_valid[b, j]:
                    continue

                idx = sel_idx[b, j].item()
                start = int(proposal_spans[b, idx, 0].item())
                end = int(proposal_spans[b, idx, 1].item())

                start = max(0, min(start, T))
                end = max(start + 1, min(end, T))

                seg = x[b, start:end]  # (len, D)

                if padding_mask is not None:
                    m = padding_mask[b, start:end].float()  # (len,)
                    den = m.sum()
                    if den.item() < 0.5:
                        prop_valid[b, j] = False
                        continue

                    if self.proposal_pooling == "mean":
                        pooled = (seg * m.unsqueeze(-1)).sum(dim=0) / den.clamp(min=1.0)
                    else:
                        pooled = (seg * m.unsqueeze(-1)).sum(dim=0) / den.clamp(min=1.0)
                else:
                    pooled = seg.mean(dim=0)

                prop_feat[b, j] = pooled
                prop_spans[b, j, 0] = start
                prop_spans[b, j, 1] = end

        return prop_feat, prop_valid, prop_spans

    def _scatter_proposals_to_sequence(
        self,
        base_x: torch.Tensor,
        prop_updates: torch.Tensor,
        prop_valid: torch.Tensor,
        prop_spans: torch.Tensor,
    ):
        """
        base_x:      (B, T, D)  仅用于提供目标 shape
        prop_updates:(B, K, D)
        prop_valid:  (B, K) bool
        prop_spans:  (B, K, 2)

        返回:
            seq_out: (B, T, D)
        """
        B, T, D = base_x.shape
        device = base_x.device

        seq_out = torch.zeros(B, T, D, device=device, dtype=base_x.dtype)
        seq_cnt = torch.zeros(B, T, 1, device=device, dtype=base_x.dtype)

        for b in range(B):
            for j in range(prop_updates.size(1)):
                if not prop_valid[b, j]:
                    continue

                start = int(prop_spans[b, j, 0].item())
                end = int(prop_spans[b, j, 1].item())

                start = max(0, min(start, T))
                end = max(start + 1, min(end, T))

                seq_out[b, start:end] += prop_updates[b, j].unsqueeze(0)
                seq_cnt[b, start:end] += 1.0

        seq_out = seq_out / seq_cnt.clamp(min=1.0)
        return seq_out

    def _get_native_proposal_aux(self, aux: Optional[dict]):
        """
        只接受 native proposal selector 产生的 proposal。
        如果 proposal 来自 fallback point-derived 路径，直接返回 None。
        """
        if aux is None:
            return None

        if not aux.get("uses_native_proposals", False):
            return None

        proposal_scores = aux.get("proposal_scores", None)
        proposal_spans = aux.get("proposal_spans", None)
        proposal_valid_mask = aux.get("proposal_valid_mask", None)

        if proposal_scores is None or proposal_spans is None or proposal_valid_mask is None:
            return None

        return {
            "proposal_scores": proposal_scores,
            "proposal_spans": proposal_spans,
            "proposal_valid_mask": proposal_valid_mask,
            "proposal_scale_ids": aux.get("proposal_scale_ids", None),
            "proposal_feats": aux.get("proposal_feats", None),
        }

    def _selected_proposals_to_time_mask(
        self,
        seq_len: int,
        proposal_spans: torch.Tensor,
        sel_idx: torch.Tensor,
        sel_valid: torch.Tensor,
        device,
    ):
        """
        proposal_spans: (B, N, 2)
        sel_idx:        (B, K)
        sel_valid:      (B, K) bool

        返回:
            time_mask: (B, T) bool
        """
        B = proposal_spans.size(0)
        time_mask = torch.zeros(B, seq_len, device=device, dtype=torch.bool)

        for b in range(B):
            for j in range(sel_idx.size(1)):
                if not sel_valid[b, j]:
                    continue

                idx = int(sel_idx[b, j].item())
                start = int(proposal_spans[b, idx, 0].item())
                end = int(proposal_spans[b, idx, 1].item())

                start = max(0, min(start, seq_len))
                end = max(start + 1, min(end, seq_len))

                time_mask[b, start:end] = True

        return time_mask
    
    def _proposal_guided_pooling(
        self,
        x_fused: torch.Tensor,
        aux_a: Optional[dict],
        aux_v: Optional[dict],
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        x_fused: (B, T, D)

        返回:
            proposal_feat: (B, D)
            proposal_time_mask: (B, T) bool
        """
        B, T, D = x_fused.shape
        device = x_fused.device

        base_valid = (
            padding_mask.bool()
            if padding_mask is not None
            else torch.ones(B, T, device=device, dtype=torch.bool)
        )

        proposal_mask_a = torch.zeros(B, T, device=device, dtype=torch.bool)
        proposal_mask_v = torch.zeros(B, T, device=device, dtype=torch.bool)

        native_aux_a = self._get_native_proposal_aux(aux_a)
        native_aux_v = self._get_native_proposal_aux(aux_v)

        # -------- audio proposals --------
        if native_aux_a is not None:
            sel_idx_a, sel_valid_a = self._select_topk_proposals(
                native_aux_a["proposal_scores"],
                native_aux_a["proposal_valid_mask"],
                topk_ratio=self.proposal_cls_topk_ratio_a,
                min_props=self.proposal_cls_min_props_a,
                max_props=self.proposal_cls_max_props_a,
            )
            proposal_mask_a = self._selected_proposals_to_time_mask(
                T, native_aux_a["proposal_spans"], sel_idx_a, sel_valid_a, device
            )

        # -------- video proposals --------
        if native_aux_v is not None:
            sel_idx_v, sel_valid_v = self._select_topk_proposals(
                native_aux_v["proposal_scores"],
                native_aux_v["proposal_valid_mask"],
                topk_ratio=self.proposal_cls_topk_ratio_v,
                min_props=self.proposal_cls_min_props_v,
                max_props=self.proposal_cls_max_props_v,
            )
            proposal_mask_v = self._selected_proposals_to_time_mask(
                T, native_aux_v["proposal_spans"], sel_idx_v, sel_valid_v, device
            )

        # union mask：音频 proposal ∪ 视频 proposal
        proposal_time_mask = (proposal_mask_a | proposal_mask_v) & base_valid

        # 如果一个 proposal 都没选出来，就退回全局 valid mask
        empty_rows = proposal_time_mask.sum(dim=1) == 0
        if empty_rows.any():
            proposal_time_mask[empty_rows] = base_valid[empty_rows]

        proposal_feat = self._masked_mean_pool(x_fused, proposal_time_mask)
        return proposal_feat, proposal_time_mask

    # -----------------------------------------------------
    # main feature extractor
    # -----------------------------------------------------
    def feature_extractor(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ):
        """
        Args:
            x: (B, T, D_total)
            padding_mask: (B, T), 1=valid, 0=pad
        """

        # 1) split modalities
        xa, xv = self._split_modalities(x)  # both are (B, T, D_in)

        # 2) input projection
        xa = self.conv_audio(xa.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
        xv = self.conv_video(xv.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)

        # zero out padded positions early
        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).float()
            xa = xa * valid
            xv = xv * valid

        # 3) modality-specific ASG
        if self.use_gate:
            xa, gate_a, logits_a = self._apply_selector(xa, self.audio_selector, padding_mask)
            xv, gate_v, logits_v = self._apply_selector(xv, self.video_selector, padding_mask)
            self.last_audio_gate = gate_a
            self.last_video_gate = gate_v
        else:
            gate_a, logits_a, gate_v, logits_v = None, None, None, None
            self.last_audio_gate = None
            self.last_video_gate = None

        # 4) unimodal encoding
        xa = self.audio_encoder(xa, inference_params=None)  # (B, T, D)
        xv = self.video_encoder(xv, inference_params=None)  # (B, T, D)

        # ===== 关键修正 1：encoder 后再次清零 padding 区域 =====
        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).float()
            xa = xa * valid
            xv = xv * valid

        # 5) proposal-aware cross-attention
        xa_attn = torch.zeros_like(xa)
        xv_attn = torch.zeros_like(xv)

        use_proposal_attn = False

        if self.use_evidence_attn and self.use_gate:
            aux_a = getattr(self.audio_selector, "last_aux", None)
            aux_v = getattr(self.video_selector, "last_aux", None)

            native_aux_a = self._get_native_proposal_aux(aux_a)
            native_aux_v = self._get_native_proposal_aux(aux_v)

            if native_aux_a is not None and native_aux_v is not None:
                proposal_scores_a = native_aux_a["proposal_scores"]
                proposal_scores_v = native_aux_v["proposal_scores"]
                proposal_valid_a = native_aux_a["proposal_valid_mask"]
                proposal_valid_v = native_aux_v["proposal_valid_mask"]
                proposal_spans_a = native_aux_a["proposal_spans"]
                proposal_spans_v = native_aux_v["proposal_spans"]

                # 1) 先选 top-k proposal
                sel_idx_a, sel_valid_a = self._select_topk_proposals(
                    proposal_scores_a,
                    proposal_valid_a,
                    topk_ratio=self.proposal_attn_topk_ratio_a,
                    min_props=self.proposal_attn_min_props_a,
                    max_props=self.proposal_attn_max_props_a,
                )
                sel_idx_v, sel_valid_v = self._select_topk_proposals(
                    proposal_scores_v,
                    proposal_valid_v,
                    topk_ratio=self.proposal_attn_topk_ratio_v,
                    min_props=self.proposal_attn_min_props_v,
                    max_props=self.proposal_attn_max_props_v,
                )

                # 2) 用单模态编码后的 full-resolution feature 做 proposal pooling
                xa_prop, xa_prop_valid, xa_prop_spans = self._pool_proposals(
                    xa, proposal_spans_a, sel_idx_a, sel_valid_a, padding_mask=padding_mask
                )
                xv_prop, xv_prop_valid, xv_prop_spans = self._pool_proposals(
                    xv, proposal_spans_v, sel_idx_v, sel_valid_v, padding_mask=padding_mask
                )

                # 3) proposal <-> proposal 双向 cross-attention
                xa_prop_attn = self.audio_from_video(
                    q=xa_prop,
                    kv=xv_prop,
                    key_padding_mask=~xv_prop_valid,
                )
                xv_prop_attn = self.video_from_audio(
                    q=xv_prop,
                    kv=xa_prop,
                    key_padding_mask=~xa_prop_valid,
                )

                xa_prop_attn = xa_prop_attn * xa_prop_valid.unsqueeze(-1).float()
                xv_prop_attn = xv_prop_attn * xv_prop_valid.unsqueeze(-1).float()

                # 4) proposal attention 结果广播回时间轴
                xa_attn = self._scatter_proposals_to_sequence(
                    xa, xa_prop_attn, xa_prop_valid, xa_prop_spans
                )
                xv_attn = self._scatter_proposals_to_sequence(
                    xv, xv_prop_attn, xv_prop_valid, xv_prop_spans
                )

                use_proposal_attn = True

        if not use_proposal_attn:
            # 回退：如果 proposal 还没启用，就继续用原来的 token-level sparse attention / dense attention
            xa_ds = self.attn_pool(xa.permute(0, 2, 1)).permute(0, 2, 1)
            xv_ds = self.attn_pool(xv.permute(0, 2, 1)).permute(0, 2, 1)

            if padding_mask is not None:
                mask_ds = F.max_pool1d(
                    padding_mask.float().unsqueeze(1),
                    kernel_size=self.attn_stride,
                    stride=self.attn_stride,
                    ceil_mode=True,
                ).squeeze(1)
                valid_mask_ds = mask_ds.bool()
            else:
                valid_mask_ds = torch.ones(
                    xa_ds.size(0), xa_ds.size(1), device=xa_ds.device, dtype=torch.bool
                )

            key_padding_mask_ds = ~valid_mask_ds

            xa_attn_ds = self.audio_from_video(
                q=xa_ds,
                kv=xv_ds,
                key_padding_mask=key_padding_mask_ds,
            )
            xv_attn_ds = self.video_from_audio(
                q=xv_ds,
                kv=xa_ds,
                key_padding_mask=key_padding_mask_ds,
            )

            xa_attn = F.interpolate(
                xa_attn_ds.permute(0, 2, 1),
                size=xa.size(1),
                mode="nearest",
            ).permute(0, 2, 1)

            xv_attn = F.interpolate(
                xv_attn_ds.permute(0, 2, 1),
                size=xv.size(1),
                mode="nearest",
            ).permute(0, 2, 1)

        # 6) 残差式增强
        xa_ctx = xa + self.cross_beta_a * xa_attn
        xv_ctx = xv + self.cross_beta_v * xv_attn

        # padding 位置重新清零
        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).float()
            xa_ctx = xa_ctx * valid
            xv_ctx = xv_ctx * valid

        # 6) fusion
        x_fused = torch.cat([xa_ctx, xv_ctx], dim=-1)  # (B, T, 2D)
        x_fused = self.fusion_encoder(x_fused, inference_params=None)

        # 7) final pooling
        if padding_mask is not None:
            global_feat = self._masked_mean_pool(x_fused, padding_mask)
        else:
            global_feat = self.pool(x_fused.permute(0, 2, 1)).squeeze(-1)

        if self.final_pooling == "mean_plus_proposal":
            proposal_feat, proposal_time_mask = self._proposal_guided_pooling(
                x_fused,
                aux_a=aux_a,
                aux_v=aux_v,
                padding_mask=padding_mask,
            )
            feat = torch.cat([global_feat, proposal_feat], dim=-1)
        else:
            feat = global_feat

        if return_intermediate:
            return {
                "feat": feat,
                "audio_feat": xa,
                "video_feat": xv,
                "audio_ctx": xa_ctx,
                "video_ctx": xv_ctx,
                "audio_gate": gate_a,
                "video_gate": gate_v,
                "audio_logits": logits_a,
                "video_logits": logits_v,
                "audio_aux": aux_a,
                "video_aux": aux_v,
            }
        else:
            return feat

    # -----------------------------------------------------
    # classifier
    # -----------------------------------------------------
    def classifier(self, x: torch.Tensor):
        return self.output(x)

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_feat: bool = False,
        return_intermediate: bool = False,
    ):
        """
        Args:
            x: (B, T, D_total)
            padding_mask: (B, T), 1=valid, 0=pad
        """
        if return_intermediate:
            info = self.feature_extractor(
                x,
                padding_mask=padding_mask,
                return_intermediate=True,
            )
            logits = self.classifier(info["feat"])
            info["logits"] = logits
            return info

        feat = self.feature_extractor(
            x,
            padding_mask=padding_mask,
            return_intermediate=False,
        )
        logits = self.classifier(feat)

        if return_feat:
            return logits, feat
        else:
            return logits