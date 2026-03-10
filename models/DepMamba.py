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
        selector_tau: float = 1.0,
        selector_hard: bool = False,
        selector_alpha: float = 0.5,
        attn_heads: int = 1,
    ):
        super().__init__()

        if uni_output_sizes is None:
            uni_output_sizes = [256, 64]
        if fusion_output_sizes is None:
            fusion_output_sizes = [128]

        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.mm_input_size = mm_input_size
        self.use_gate = use_gate
        self.selector_alpha = selector_alpha

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
        )
        self.video_selector = EvidenceSelector(
            d_in=mm_input_size,
            d_hidden=mm_input_size,
            tau=selector_tau,
            hard=selector_hard,
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

        self.attn_stride = 4   # 先用 8，后面不稳可以改成 16
        self.attn_pool = nn.MaxPool1d(
            kernel_size=self.attn_stride,
            stride=self.attn_stride,
            ceil_mode=True,
        )

        # ---------------------------
        # 4) lightweight MulT
        # ---------------------------
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
        self.cross_beta_a = 0.1
        self.cross_beta_v = 0.1

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
        self.output = nn.Linear(fusion_output_sizes[-1], 1)

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
            logits: whatever selector returns
        """
        gate, logits = selector(x.permute(0, 2, 1))  # gate: (B, 1, T)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(1).float()  # (B,1,T)
            gate = gate * valid

        gate_t = gate.permute(0, 2, 1)  # (B, T, 1)

        alpha = self.selector_alpha
        x = x * (alpha + (1.0 - alpha) * gate_t)
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

        # 5) lightweight MulT on downsampled sequences (single direction)
        # 先对时间维降采样，再做 cross-attention，避免长序列 O(T^2) 不稳定

        xa_ds = self.attn_pool(xa.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T_ds, D)
        xv_ds = self.attn_pool(xv.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T_ds, D)

        if padding_mask is not None:
            # 窗口里只要有一个有效位置，就认为该 downsample token 有效
            mask_ds = F.max_pool1d(
                padding_mask.float().unsqueeze(1),
                kernel_size=self.attn_stride,
                stride=self.attn_stride,
                ceil_mode=True,
            ).squeeze(1)  # (B, T_ds)

            key_padding_mask_ds = ~mask_ds.bool()   # True = ignore
        else:
            mask_ds = None
            key_padding_mask_ds = None

        # 双向：audio <- video, video <- audio
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

        # 上采样回原始长度
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

        # 残差式增强，不直接替换
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

        # 7) pooling
        if padding_mask is not None:
            feat = self._masked_mean_pool(x_fused, padding_mask)
        else:
            feat = self.pool(x_fused.permute(0, 2, 1)).squeeze(-1)

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