"""ES-DepMamba implementation.
Authors
-------
* Yaxin Bai 2025
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .evidence_selector import EvidenceSelector

# ---- torchaudio 兼容：新版本没有 list_audio_backends ----
if not hasattr(torchaudio, "list_audio_backends"):
    def _compat_list_audio_backends():
        try:
            backend = torchaudio.get_audio_backend()
            return [backend] if backend is not None else []
        except Exception:
            return []
    torchaudio.list_audio_backends = _compat_list_audio_backends


import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

import torch.utils._pytree as _pytree

import torch.utils._pytree as _pytree

# 兼容 torch 2.1：没有 register_pytree_node，就用旧的 _register_pytree_node 包一层
if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
    def _patched_register_pytree_node(node_type, flatten_fn, unflatten_fn, *,
                                      serialized_type_name=None,
                                      serialized_ref_fn=None):
        # 忽略新参数，直接调用旧接口
        return _pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    _pytree.register_pytree_node = _patched_register_pytree_node

from mamba_ssm import Mamba  
from .es_mamba import Mamba as ESMamba 
from .mamba.bimamba import Mamba as BiMamba
from .mamba.mm_bimamba import Mamba as MMBiMamba
from .base import BaseNet



class MMMambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.a_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        
        a_out1, v_out1 = self.mamba(a_x, v_x,a_inference_params,v_inference_params)
        a_out = a_x + self.norm1(a_out1)
        v_out = v_x + self.norm2(v_out1)

        return a_out, v_out

class MMCNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.a_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.a_bn = nn.BatchNorm1d(output_size)

        self.v_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.v_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.a_drop = nn.Dropout(dropout)
        self.v_drop = nn.Dropout(dropout)

        self.a_net = nn.Sequential(self.a_conv, self.a_bn, self.relu, self.a_drop)
        self.v_net = nn.Sequential(self.v_conv, self.v_bn, self.relu, self.v_drop)

        if input_size != output_size:
            self.a_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.v_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.a_skipconv = None
            self.v_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.a_conv.weight.data)
        nn.init.xavier_uniform_(self.v_conv.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, xa, xv):
        a_out = self.a_net(xa)
        v_out = self.v_net(xv)
        if self.a_skipconv is not None:
            xa = self.a_skipconv(xa)
        if self.v_skipconv is not None:
            xv = self.v_skipconv(xv)
        a_out = a_out+xa
        v_out = v_out+xv
        return a_out, v_out

class MambaEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config is not None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        # 从 config 里拿出 bidirectional
        bidirectional = mamba_config.pop('bidirectional')

        if causal or (not bidirectional):
            # ★ 单向：改成用你自己的 ESMamba（es_mamba 里那个）
            self.mamba = ESMamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            # ★ 双向：仍然用原来的 BiMamba，不支持 gate
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )

        # 用完再放回去，防止外面复用 mamba_config 时出问题
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        inference_params=None,
        gate=None,   # 暂时没人传，默认 None
    ):
        # 只有 ESMamba 才支持 gate 这个 keyword
        if isinstance(self.mamba, ESMamba):
            core_out = self.mamba(
                x,
                gate=gate,
                inference_params=inference_params,
            )
        else:
            # BiMamba / 原 Mamba：忽略 gate，只按老接口调用
            core_out = self.mamba(
                x,
                inference_params,
            )

        out = x + self.norm1(core_out)
        return out

class CNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_size)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(output_size, output_size, 5, padding=2, dilation=dilation, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_size)
        # self.relu2 = nn.ReLU()

        self.drop = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.drop)

        if input_size != output_size:
            self.conv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.conv = None
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight.data)
        # nn.init.xavier_uniform_(self.conv2.weight.data)

    def forward(self, x):
        out = self.net(x)
        if self.conv is not None:
            x = self.conv(x)
        out = out+x
        return out

class CoSSM(nn.Module):
    """This class implements the CoSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size = 3,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(MMCNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MMMambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        a_x, v_x, 
        a_inference_params = None,
        v_inference_params = None
    ):
        a_out = a_x
        v_out = v_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            a_out, v_out  = cnn_layer(a_out.permute(0,2,1), v_out.permute(0,2,1))
            a_out = a_out.permute(0,2,1)
            v_out = v_out.permute(0,2,1)
            a_out, v_out = mamba_layer(
                a_out, v_out,
                a_inference_params = a_inference_params,
                v_inference_params = v_inference_params
            )
            
        return a_out, v_out

class EnSSM(nn.Module):
    """This class implements the EnSSM encoder.
    """
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256,512,512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')
        prev_input_size = input_size

        cnn_list = []
        mamba_list = []
        # print(output_sizes)
        for i in range(len(output_sizes)):
            cnn_list.append(CNNEncoderLayer(
                    input_size = input_size if i<1 else output_sizes[i-1],
                    output_size = output_sizes[i],
                    dropout=dropout
                ))
            mamba_list.append(MambaEncoderLayer(
                    d_model=output_sizes[i],
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    causal=causal,
                    mamba_config=mamba_config,
                ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)


    def forward(
        self,
        x,
        gate=None,
        inference_params = None,
    ):
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            out  = cnn_layer(out.permute(0,2,1))
            out = out.permute(0,2,1)
            out = mamba_layer(
                out,
                inference_params = inference_params,
                gate=gate, 
            )

        return out

class DepMamba(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, mm_output_sizes=[256,64], d_ffn=1024, num_layers=8, dropout=0.1, activation='Swish', causal=False, mamba_config=None):
        super().__init__()

        self.cossm_encoder = CoSSM(num_layers,
                                         mm_input_size,
                                    mm_output_sizes,
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        
        self.enssm_encoder = EnSSM(num_layers,
                                    mm_output_sizes[-1]*2,
                                    [mm_output_sizes[-1]*2],
                                    d_ffn,
                                    activation=activation,
                                    dropout=dropout,
                                    causal=causal,
                                    mamba_config=mamba_config)
        
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.output = nn.Linear(mm_output_sizes[-1]*2, 1)
        self.m = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)
        # ===== ES-Mamba: 新增，音频 Evidence Selector（当前只计算，不参与决策） =====
        # xa 在 Conv 之后的形状是 (B, L, mm_input_size)，Selector 期望 (B, D, L)
        selector_tau = 1.0   # 先写死，后面可以放到 config 里
        self.audio_selector = EvidenceSelector(
            d_in=mm_input_size,
            d_hidden=mm_input_size,
            tau=selector_tau,
            hard=False,        # 当前阶段只用 soft gate
        )
        # 用于在训练循环里取 gate（可选）
        self.last_audio_gate = None
        # =============================================================
        

    def feature_extractor(self, x, padding_mask=None, a_inference_params = None, v_inference_params = None):
        xa = x[:, :, 136:]
        xv = x[:, :, :136]
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1)
        # ===== ES-Mamba: 使用 Conv 后的音频特征做 Evidence Selection =====
        # 当前 xa: (B, L, mm_input_size) -> 转成 (B, D, L) 喂给 Selector
        xa_for_sel = xa.permute(0, 2, 1)           # (B, D, L)
        gate_a, logits_a = self.audio_selector(xa_for_sel)  # gate_a: (B, 1, L)

        # 先不把 gate 用到后续计算，只保存下来，后面加稀疏 loss / Mamba gating 时会用到
        self.last_audio_gate = gate_a
        # ===============================================================

        # ===== ESMamba-input：用 gate 先做一次输入级 Mask =====
        # xa: (B, L, D), gate_a: (B, 1, L) -> (B, L, 1)
        gate_a_t = gate_a.permute(0, 2, 1)         # (B, L, 1)
        xa = xa * gate_a_t                         # 广播到 D 维
        # ======================================================
        
        xa, xv = self.cossm_encoder(xa, xv, a_inference_params, v_inference_params)

        x = torch.cat([xa,xv],dim=-1)
        x = self.enssm_encoder(
            x,
            gate=self.last_audio_gate,   # 或者 gate=gate_a，等价
            inference_params=None,
        )
        
        if padding_mask is not None:
            x = x * (padding_mask.unsqueeze(-1).float())
            x = x.sum(dim=1) / (padding_mask.unsqueeze(-1).float()
                                ).sum(dim=1, keepdim=False)  # Compute average
        else:
            x = self.pool(x.permute(0,2,1)).squeeze(-1)
        return x

    def classifier(self, x):
        return self.output(x)
