
import copy
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba as UniMamba
from .mamba.bimamba import Mamba as BiMamba
from .evidence_selector import EvidenceDiscoveryHead
from .base import BaseNet


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int):
    if mask is None:
        return x.mean(dim=dim)
    m = mask.float()
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp(min=1e-6)
    return num / den


def safe_logit(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class CNNEncoderLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_size, output_size, kernel_size=3, padding=dilation, dilation=dilation, bias=False
        )
        self.norm1 = nn.LayerNorm(output_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(input_size, output_size, kernel_size=1, bias=False) if input_size != output_size else None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        if self.skip is not None:
            nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.act(out)
        out = self.drop(out)
        residual = x if self.skip is None else self.skip(x)
        return out + residual


class MambaEncoderLayer(nn.Module):
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
            self.mamba = BiMamba(d_model=d_model, bimamba_type="v2", **cfg)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        x_norm = self.norm(x)
        try:
            core_out = self.mamba(x_norm, inference_params=inference_params)
        except TypeError:
            try:
                core_out = self.mamba(x_norm, inference_params)
            except TypeError:
                core_out = self.mamba(x_norm)
        core_out = torch.nan_to_num(core_out, nan=0.0, posinf=1e4, neginf=-1e4)
        return x + self.drop(core_out)


class EnSSM(nn.Module):
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
            cnn_list.append(CNNEncoderLayer(in_dim, out_dim, dropout=dropout))
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
        padding_mask: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        out = x
        valid = padding_mask.unsqueeze(-1).float() if padding_mask is not None else None
        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out = cnn_layer(out.transpose(1, 2)).transpose(1, 2)
            if valid is not None:
                out = out * valid
            out = mamba_layer(out, inference_params=inference_params)
            if valid is not None:
                out = out * valid
        return out


class ContextualEvidenceRefiner(nn.Module):
    """
    Refine proposal evidence with projected local features + contextualized states.
    Use bounded delta_logit to avoid score saturation / overconfident evidence.
    """
    def __init__(
        self,
        d_local: int,
        d_ctx: int,
        d_hidden: int = 128,
        dropout: float = 0.1,
        delta_scale: float = 2.0,
        score_temp: float = 2.0,
        logit_clip: float = 8.0,
        proposal_scale: float = 0.5,
    ):
        super().__init__()
        
        self.local_norm = nn.LayerNorm(d_local)
        self.ctx_norm = nn.LayerNorm(d_ctx)

        self.delta_scale = delta_scale
        self.score_temp = score_temp
        self.logit_clip = logit_clip
        self.proposal_scale = proposal_scale

        self.net = nn.Sequential(
            nn.Linear(d_local + d_ctx + 1, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(
        self,
        local_feat: torch.Tensor,         # (B, T, D_local)
        ctx_feat: torch.Tensor,           # (B, T, D_ctx)
        proposal_score: torch.Tensor,     # (B, T)   保留接口兼容
        proposal_logit: torch.Tensor,     # (B, T)
        padding_mask: Optional[torch.Tensor] = None,
    ):
        local_feat = self.local_norm(local_feat)
        ctx_feat = self.ctx_norm(ctx_feat)

        # 用 bounded proposal logit 作为 refiner 的 score feature
        base_logit = proposal_logit.clamp(min=-self.logit_clip, max=self.logit_clip)
        score_feat = torch.tanh(base_logit / 4.0).unsqueeze(-1)  # (B, T, 1)

        z = torch.cat([local_feat, ctx_feat, score_feat], dim=-1)

        raw_delta = self.net(z).squeeze(-1)
        delta_logit = self.delta_scale * torch.tanh(raw_delta)   # bounded update

        refined_logit = self.proposal_scale * base_logit + delta_logit
        refined_logit = refined_logit.clamp(min=-self.logit_clip, max=self.logit_clip)

        if padding_mask is not None:
            refined_logit = refined_logit.masked_fill(padding_mask == 0, -1e4)

        refined_score = torch.sigmoid(refined_logit / self.score_temp)

        if padding_mask is not None:
            refined_score = refined_score * padding_mask.float()

        return refined_score, refined_logit


class EvidenceAggregator(nn.Module):
    """
    evidence-aware full-sequence aggregation
    with bounded evidence logit to reduce overconfident overfitting.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        agg_beta: float = 0.7,
        max_evidence_logit: float = 4.0,
    ):
        super().__init__()
        self.ctx_norm = nn.LayerNorm(d_model)
        self.score_proj = nn.Linear(d_model, d_model)
        self.score_vec = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        self.agg_beta = agg_beta
        self.max_evidence_logit = max_evidence_logit

    def forward(
        self,
        ctx_feat: torch.Tensor,           # (B, T, D)
        corroborated_score: torch.Tensor, # (B, T)
        padding_mask: Optional[torch.Tensor] = None,
    ):
        ctx_feat_norm = self.ctx_norm(ctx_feat)

        global_logit = self.score_vec(
            torch.tanh(self.score_proj(self.drop(ctx_feat_norm)))
        ).squeeze(-1)

        evidence_logit = safe_logit(corroborated_score)
        evidence_logit = evidence_logit.clamp(
            min=-self.max_evidence_logit,
            max=self.max_evidence_logit,
        )

        agg_logit = global_logit + self.agg_beta * evidence_logit

        if padding_mask is not None:
            agg_logit = agg_logit.masked_fill(padding_mask == 0, -1e4)

        attn = torch.softmax(agg_logit, dim=-1)

        if padding_mask is not None:
            attn = attn * padding_mask.float()
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        rep = torch.sum(attn.unsqueeze(-1) * ctx_feat, dim=1)
        return rep, attn, agg_logit

class CorroborationHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z = torch.cat([q, s, torch.abs(q - s), q * s], dim=-1)
        return torch.sigmoid(self.mlp(z)).squeeze(-1)


class DepMamba(BaseNet):
    """
    Single-head evidence-centric DepMamba
        input projection
        -> modality-specific Evidence Discovery
        -> unimodal contextual encoders
        -> Contextual Evidence Refinement
        -> Sparse span routing
        -> local cross-modal corroboration
        -> corroborated evidence map
        -> evidence-aware holistic decision
    """

    def __init__(
        self,
        audio_input_size: int = 161,
        video_input_size: int = 136,
        mm_input_size: int = 128,
        uni_output_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        causal: bool = False,
        mamba_config: Optional[dict] = None,
        discovery_hidden: int = 128,
        refine_hidden: int = 128,
        corroboration_hidden: int = 128,
        agg_beta: float = 0.7,
        route_thresh: float = 0.55,
        route_topk: int = 3,
        min_span_len: int = 2,
        support_radius: int = 8,
        support_bias: float = 0.5,
        corro_shift: float = 1.0,
        span_lambda_max: float = 0.5,
        span_lambda_avg: float = 0.3,
        span_lambda_c: float = 0.2,
        contrast_windows: Optional[List[int]] = None,
        use_corroboration: bool = True,
        # keep legacy-friendly kwargs so old configs do not crash
        use_gate: bool = False,
        selector_tau: float = 1.0,
        selector_hard: bool = False,
        selector_alpha: float = 0.5,
        attn_heads: int = 1,
        use_cross_attn: bool = False,
        attn_dropout: float = 0.1,
        plain_pool_only: bool = False,
        fusion_output_sizes: Optional[List[int]] = None,
    ):
        super().__init__()

        if uni_output_sizes is None:
            uni_output_sizes = [256, 64]
        if contrast_windows is None:
            contrast_windows = [3, 5]

        self.audio_input_size = audio_input_size
        self.video_input_size = video_input_size
        self.mm_input_size = mm_input_size
        self.route_thresh = route_thresh
        self.route_topk = route_topk
        self.min_span_len = min_span_len
        self.support_radius = support_radius
        self.support_bias = support_bias
        self.corro_shift = corro_shift
        self.span_lambda_max = span_lambda_max
        self.span_lambda_avg = span_lambda_avg
        self.span_lambda_c = span_lambda_c
        self.contrast_windows = contrast_windows
        self.use_corroboration = use_corroboration

        # 1) input projection
        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, kernel_size=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv_audio.weight)
        nn.init.xavier_uniform_(self.conv_video.weight)

        #audio layernorm
        self.audio_proj_norm = nn.LayerNorm(mm_input_size)
        self.video_proj_norm = nn.LayerNorm(mm_input_size)
        self.proj_drop = nn.Dropout(dropout)

        # 2) evidence discovery
        self.audio_discovery = EvidenceDiscoveryHead(d_in=mm_input_size, d_hidden=discovery_hidden)
        self.video_discovery = EvidenceDiscoveryHead(d_in=mm_input_size, d_hidden=discovery_hidden)

        # 3) unimodal encoders
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

        # 4) contextual evidence refinement
        self.audio_refiner = ContextualEvidenceRefiner(
            d_local=mm_input_size,
            d_ctx=d_model,
            d_hidden=refine_hidden,
            dropout=dropout,
        )
        self.video_refiner = ContextualEvidenceRefiner(
            d_local=mm_input_size,
            d_ctx=d_model,
            d_hidden=refine_hidden,
            dropout=dropout,
        )

        # 5) corroboration
        self.audio_corro_head = CorroborationHead(d_model=d_model, hidden=corroboration_hidden, dropout=dropout)
        self.video_corro_head = CorroborationHead(d_model=d_model, hidden=corroboration_hidden, dropout=dropout)

        # 6) evidence-aware aggregation
        self.audio_aggregator = EvidenceAggregator(d_model=d_model, dropout=dropout, agg_beta=agg_beta)
        self.video_aggregator = EvidenceAggregator(d_model=d_model, dropout=dropout, agg_beta=agg_beta)

        # 7) single-head classifier
        self.cls_drop = nn.Dropout(dropout)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(d_model, 1)

    # -----------------------------------------------------
    # basic helpers
    # -----------------------------------------------------
    def _split_modalities(self, x: torch.Tensor):
        total_needed = self.video_input_size + self.audio_input_size
        if x.size(-1) < total_needed:
            raise ValueError(
                f"Input feature dim={x.size(-1)} is smaller than "
                f"video_input_size + audio_input_size = {total_needed}."
            )
        xv = x[:, :, :self.video_input_size]
        xa = x[:, :, self.video_input_size:self.video_input_size + self.audio_input_size]
        return xa, xv

    @staticmethod
    def _masked_same_avg(score: torch.Tensor, valid: Optional[torch.Tensor], k: int):
        # score: (B, T), valid: (B, T)
        x = score.unsqueeze(1)
        if valid is None:
            m = torch.ones_like(x)
        else:
            m = valid.float().unsqueeze(1)

        pad = k // 2
        x_num = F.pad(x * m, (pad, pad), mode="constant", value=0.0)
        x_den = F.pad(m, (pad, pad), mode="constant", value=0.0)

        num = F.avg_pool1d(x_num, kernel_size=k, stride=1) * k
        den = F.avg_pool1d(x_den, kernel_size=k, stride=1) * k
        return (num / den.clamp(min=1e-6)).squeeze(1)

    def compute_contrast_cue(self, refined_score: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # refined_score: (B, T)
        valid = padding_mask if padding_mask is not None else None
        comp_all = []
        for k in self.contrast_windows:
            inside = self._masked_same_avg(refined_score, valid, k)
            outer_k = 2 * k + 1
            outer = self._masked_same_avg(refined_score, valid, outer_k)
            comp_all.append(inside - outer)
        contrast = torch.stack(comp_all, dim=0).max(dim=0).values
        if padding_mask is not None:
            contrast = contrast * padding_mask.float()
        return contrast

    def _score_to_spans(
        self,
        score: torch.Tensor,     # (T,)
        contrast: torch.Tensor,  # (T,)
        valid_len: int,
    ) -> List[Dict[str, Any]]:
        if valid_len <= 0:
            return []

        s = score[:valid_len]
        c = contrast[:valid_len]

        thr = max(self.route_thresh, float(s.mean().item() + 0.25 * s.std(unbiased=False).item()))
        active = s >= thr

        spans: List[Dict[str, Any]] = []
        start = None

        for t in range(valid_len):
            flag = bool(active[t].item())
            if flag and start is None:
                start = t

            is_boundary = (not flag) or (t == valid_len - 1)
            if is_boundary and start is not None:
                end = t if (flag and t == valid_len - 1) else (t - 1)
                if end >= start and (end - start + 1) >= self.min_span_len:
                    seg_s = s[start:end + 1]
                    seg_c = c[start:end + 1]
                    rel_anchor = int(seg_s.argmax().item())
                    anchor = start + rel_anchor
                    quality = (
                        self.span_lambda_max * float(seg_s.max().item())
                        + self.span_lambda_avg * float(seg_s.mean().item())
                        + self.span_lambda_c * float(seg_c.mean().item())
                    )
                    spans.append(
                        {
                            "start": start,
                            "end": end,
                            "anchor": anchor,
                            "quality": quality,
                        }
                    )
                start = None

        # robust fallback: if no span survives, keep the strongest token as a singleton span
        if len(spans) == 0 and valid_len > 0:
            anchor = int(s.argmax().item())
            spans.append(
                {
                    "start": anchor,
                    "end": anchor,
                    "anchor": anchor,
                    "quality": float(s[anchor].item()),
                }
            )

        spans = sorted(spans, key=lambda z: z["quality"], reverse=True)[: self.route_topk]
        return spans

    def _build_batch_spans(
        self,
        score: torch.Tensor,           # (B, T)
        contrast: torch.Tensor,        # (B, T)
        padding_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Dict[str, Any]]]:
        B, T = score.shape
        if padding_mask is None:
            lengths = [T] * B
        else:
            lengths = padding_mask.sum(dim=1).long().tolist()

        batch_spans = []
        for b in range(B):
            batch_spans.append(self._score_to_spans(score[b], contrast[b], lengths[b]))
        return batch_spans

    def _local_support_pool(self, center: int, valid_len: int):
        left = max(0, center - self.support_radius)
        right = min(valid_len, center + self.support_radius + 1)
        return left, right

    def _corroborate_direction(
        self,
        query_ctx: torch.Tensor,       # (B, Tq, D)
        query_refined: torch.Tensor,   # (B, Tq)
        support_ctx: torch.Tensor,     # (B, Ts, D)
        support_refined: torch.Tensor, # (B, Ts)
        query_spans: List[List[Dict[str, Any]]],
        support_mask: Optional[torch.Tensor],
        corro_head: nn.Module,
    ):
        B, Tq, D = query_ctx.shape
        if support_mask is None:
            support_lens = [support_ctx.size(1)] * B
        else:
            support_lens = support_mask.sum(dim=1).long().tolist()

        corroborated_logit = safe_logit(query_refined)
        corroborated_shift = torch.zeros_like(query_refined)

        span_aux = []

        for b in range(B):
            sample_aux = []
            s_len = support_lens[b]

            for span in query_spans[b]:
                st, ed, anchor = span["start"], span["end"], span["anchor"]

                q_tokens = query_ctx[b, st:ed + 1]                    # (Lq, D)
                q_score = query_refined[b, st:ed + 1]                # (Lq,)
                q_weight = q_score / q_score.sum().clamp(min=1e-6)
                q_repr = torch.sum(q_tokens * q_weight.unsqueeze(-1), dim=0)  # (D,)

                left, right = self._local_support_pool(anchor, s_len)
                if right <= left:
                    continue

                k = support_ctx[b, left:right]                       # (Ls, D)
                bias = support_refined[b, left:right]                # (Ls,)

                attn_logits = torch.matmul(k, q_repr) / math.sqrt(D)
                attn_logits = attn_logits + self.support_bias * safe_logit(bias)
                attn = torch.softmax(attn_logits, dim=0)
                s_repr = torch.sum(attn.unsqueeze(-1) * k, dim=0)

                corroboration_strength = corro_head(q_repr.unsqueeze(0), s_repr.unsqueeze(0))[0]
                quality = torch.sigmoid(torch.tensor(span["quality"], device=query_ctx.device, dtype=query_ctx.dtype))
                final_conf = (quality * corroboration_strength).clamp(min=1e-4, max=1.0 - 1e-4)

                shift = self.corro_shift * safe_logit(final_conf.unsqueeze(0))[0]
                corroborated_logit[b, st:ed + 1] = corroborated_logit[b, st:ed + 1] + shift
                corroborated_shift[b, st:ed + 1] = corroborated_shift[b, st:ed + 1] + shift

                sample_aux.append(
                    {
                        "start": st,
                        "end": ed,
                        "anchor": anchor,
                        "quality": float(span["quality"]),
                        "corro_strength": float(corroboration_strength.detach().item()),
                        "final_conf": float(final_conf.detach().item()),
                    }
                )

            span_aux.append(sample_aux)

        corroborated_score = torch.sigmoid(corroborated_logit)
        return corroborated_score, corroborated_shift, span_aux

    def _project_inputs(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        xa_raw, xv_raw = self._split_modalities(x)

        xa0 = self.conv_audio(xa_raw.transpose(1, 2)).transpose(1, 2)
        xv0 = self.conv_video(xv_raw.transpose(1, 2)).transpose(1, 2)

        xa0 = self.audio_proj_norm(xa0)
        xv0 = self.video_proj_norm(xv0)

        xa0 = self.proj_drop(xa0)
        xv0 = self.proj_drop(xv0)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).float()
            xa0 = xa0 * valid
            xv0 = xv0 * valid

        return xa0, xv0

    # -----------------------------------------------------
    # feature extractor
    # -----------------------------------------------------
    def feature_extractor(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
    ):
        xa0, xv0 = self._project_inputs(x, padding_mask=padding_mask)

        # evidence discovery on projected local features
        p_a, aux_a = self.audio_discovery(xa0.transpose(1, 2), padding_mask)
        p_v, aux_v = self.video_discovery(xv0.transpose(1, 2), padding_mask)

        # unimodal contextual modeling
        h_a = self.audio_encoder(xa0, padding_mask=padding_mask, inference_params=None)
        h_v = self.video_encoder(xv0, padding_mask=padding_mask, inference_params=None)

        if padding_mask is not None:
            valid = padding_mask.unsqueeze(-1).float()
            h_a = h_a * valid
            h_v = h_v * valid

        # contextual evidence refinement
        r_a, rlogit_a = self.audio_refiner(
            local_feat=xa0,
            ctx_feat=h_a,
            proposal_score=p_a,
            proposal_logit=aux_a["proposal_logit"],
            padding_mask=padding_mask,
        )
        r_v, rlogit_v = self.video_refiner(
            local_feat=xv0,
            ctx_feat=h_v,
            proposal_score=p_v,
            proposal_logit=aux_v["proposal_logit"],
            padding_mask=padding_mask,
        )

        # contrast cue for SES / span quality
        c_a = self.compute_contrast_cue(r_a, padding_mask)
        c_v = self.compute_contrast_cue(r_v, padding_mask)

        # sparse span routing
        spans_a = self._build_batch_spans(r_a, c_a, padding_mask)
        spans_v = self._build_batch_spans(r_v, c_v, padding_mask)

        # asynchronous cross-modal corroboration
        if self.use_corroboration:
            hat_r_a, shift_a, span_aux_a = self._corroborate_direction(
                query_ctx=h_a,
                query_refined=r_a,
                support_ctx=h_v,
                support_refined=r_v,
                query_spans=spans_a,
                support_mask=padding_mask,
                corro_head=self.audio_corro_head,
            )
            hat_r_v, shift_v, span_aux_v = self._corroborate_direction(
                query_ctx=h_v,
                query_refined=r_v,
                support_ctx=h_a,
                support_refined=r_a,
                query_spans=spans_v,
                support_mask=padding_mask,
                corro_head=self.video_corro_head,
            )
        else:
            hat_r_a = r_a
            hat_r_v = r_v
            shift_a = torch.zeros_like(r_a)
            shift_v = torch.zeros_like(r_v)
            span_aux_a = [[] for _ in range(r_a.size(0))]
            span_aux_v = [[] for _ in range(r_v.size(0))]

        if padding_mask is not None:
            hat_r_a = hat_r_a * padding_mask.float()
            hat_r_v = hat_r_v * padding_mask.float()

        # evidence-aware holistic decision
        d_a, alpha_a, agg_logit_a = self.audio_aggregator(h_a, hat_r_a, padding_mask)
        d_v, alpha_v, agg_logit_v = self.video_aggregator(h_v, hat_r_v, padding_mask)

        fused = torch.cat([d_a, d_v, torch.abs(d_a - d_v), d_a * d_v], dim=-1)

        if return_intermediate:
            return {
                "feat": fused,
                "audio_local": xa0,
                "video_local": xv0,
                "audio_ctx": h_a,
                "video_ctx": h_v,
                "audio_proposal": p_a,
                "video_proposal": p_v,
                "audio_proposal_logit": aux_a["proposal_logit"],
                "video_proposal_logit": aux_v["proposal_logit"],
                "audio_refined": r_a,
                "video_refined": r_v,
                "audio_refined_logit": rlogit_a,
                "video_refined_logit": rlogit_v,
                "audio_contrast": c_a,
                "video_contrast": c_v,
                "audio_corroborated": hat_r_a,
                "video_corroborated": hat_r_v,
                "audio_corro_shift": shift_a,
                "video_corro_shift": shift_v,
                "audio_attn": alpha_a,
                "video_attn": alpha_v,
                "audio_agg_logit": agg_logit_a,
                "video_agg_logit": agg_logit_v,
                "audio_spans": spans_a,
                "video_spans": spans_v,
                "audio_span_aux": span_aux_a,
                "video_span_aux": span_aux_v,
            }
        return fused

    def classifier(self, feat: torch.Tensor):
        feat = self.cls_drop(feat)
        hidden = self.fusion_mlp(feat)
        return self.output(hidden), hidden

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_feat: bool = False,
        return_intermediate: bool = False,
    ):
        if return_intermediate:
            info = self.feature_extractor(x, padding_mask=padding_mask, return_intermediate=True)
            logits, hidden = self.classifier(info["feat"])
            info["decision_feat"] = hidden
            info["logits"] = logits
            return info

        feat = self.feature_extractor(x, padding_mask=padding_mask, return_intermediate=False)
        logits, hidden = self.classifier(feat)

        if return_feat:
            return logits, hidden
        return logits
