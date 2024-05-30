# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import copy
import os
import onnx
# import onnx_tensorrt.backend as backend
# import onnxruntime as ort
import sys
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
# import torch_tensorrt
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import tensorrt
from cuda import cudart

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLlamaDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
try:
    from .configs import EConfig
    from .utils_c import *
    from .choices import *
except:
    from configs import EConfig
    from utils_c import *
    from choices import *
    from utils import prepare_logits_processor
top_k=10

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x):
        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

# class Linear(Module):
    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # weight: Tensor

    # def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 # device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        # super().__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # if bias:
            # self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
            # self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self) -> None:
        # # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # # https://github.com/pytorch/pytorch/issues/57109
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # init.uniform_(self.bias, -bound, bound)

    # def forward(self, input: Tensor) -> Tensor:
        # return F.linear(input, self.weight, self.bias)

    # def extra_repr(self) -> str:
        # return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

# class LinearInt4(nn.Module):
    # __constants__ = ['in_features', 'out_features']
    # in_features: int
    # out_features: int
    # weight: Tensor
    # scale: Tensor
    # zp: Tensor

    # def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 # device=None, dtype=None) -> None:
        # super().__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = Parameter(torch.empty((out_features, in_features), 'device': device, 'dtype': torch.int4))
        # self.scale = Parameter(torch.empty((out_features, in_features), 'device': device, 'dtype': dtype))
        # self.scale = Parameter(torch.empty((out_features, in_features), 'device': device, 'dtype': dtype))
        # if bias:
            # self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
            # self.register_parameter('bias', None)
        # self.reset_parameters()
        

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states_ = torch.concat((past_key_value[0], key_states), dim=2)
            value_states_ = torch.concat((past_key_value[1], value_states), dim=2)
        else:
            key_states_ = key_states
            value_states_ = value_states

        # past_key_value = (key_states, value_states) if use_cache else None

        # # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_.transpose(2, 3)) / math.sqrt(self.head_dim)

        # attn_weights with shape [bs, head_num, new_seqlen, all_seqlen]
        if attention_mask:
            attn_mask = torch.empty_like(attn_weights[0,0]).fill_(torch.finfo(attn_weights.dtype).min)
            attn_mask = torch.triu(attn_mask, diagonal=(attn_weights.shape[3]-attn_weights.shape[2]+1))
            attn_weights += attn_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, (key_states,value_states)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj, None

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        if config.num_local_experts > 0:
            self.mlp = MixtralSparseMoeBlock(config)
        else:
            self.mlp = LlamaMLP(config)
        self.index=index
        # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # @torch.compile()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: bool = False,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # print("middle attention_output", hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        outputs += (present_key_value,)

        return outputs


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))
    def forward(self,x):
        return x + self.dummy - self.dummy #(also tried x+self.dummy)

def len_list(x,n):
    return [i for i in x if len(i)<=n]

class Model(nn.Module):
    def __init__(self,config,load_emb=False,path=None,bias=True, load_checkpoint=None, training=False, ea_engine_path=None):
        super().__init__()

        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.base_hidden_size = config.base_hidden_size
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        # kv-cache shape [layer_num, 2, bs, kv_head_num, seq_len, kv_head_dim]
        self.stable_kv = torch.zeros((2, 16, config.num_key_value_heads, 1024, self.hidden_size // config.num_key_value_heads)).cuda().half()
        self.trt_k_out = torch.zeros((1, config.num_key_value_heads, 1, self.hidden_size // config.num_key_value_heads)).cuda().half()
        self.trt_v_out = torch.zeros((1, config.num_key_value_heads, 1, self.hidden_size // config.num_key_value_heads)).cuda().half()
        self.stable_kv_len = 0
        self.eagle_gen_time = 0.


        self.embed_tokens = nn.Embedding(config.vocab_size, self.base_hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path,"model.safetensors.index.json"),"r") as f:
                    index_json=json.loads(f.read())
                    emb_path=index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path,emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights=torch.load(os.path.join(path,emb_path))
                tensor=weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor


        #self.init_tree()

        if training and False:
            self.layers = nn.ModuleList([HFLlamaDecoderLayer(config,index) for index in range(config.num_hidden_layers)])
        else:
            self.layers = nn.ModuleList([LlamaDecoderLayer(config,index) for index in range(config.num_hidden_layers)])
        self.fc=nn.Linear(2*self.base_hidden_size,self.hidden_size,bias=bias)
        if (self.base_hidden_size != self.hidden_size):
            self.fc_back=nn.Linear(self.hidden_size,self.base_hidden_size,bias=bias)
        self.act=ACT2FN[config.hidden_act]
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

        if load_checkpoint:
            ea_layer_state_dict = torch.load(load_checkpoint,
                                             map_location=self.embed_tokens.weight.device)
            self.load_state_dict(ea_layer_state_dict, strict=True)

        # self.session = ort.InferenceSession("trt-org/eagle_small.onnx", providers=['CUDAExecutionProvider'])
        self.torch_stream = torch.cuda.current_stream()
        self.cuda_stream = torch.cuda.current_stream().cuda_stream
        self.trt_engine = False
        if ea_engine_path is not None:
            with open(ea_engine_path+"/context.trt", "rb") as f, tensorrt.Runtime(tensorrt.Logger()) as runtime:
                self.context_engine = runtime.deserialize_cuda_engine(f.read())
                address = CUASSERT(cudart.cudaMalloc(self.context_engine.device_memory_size))[0]
            self.context_context = self.context_engine.create_execution_context_without_device_memory()
            self.context_context.set_optimization_profile_async(0, self.cuda_stream)
            self.context_context.device_memory = address
            with open(ea_engine_path+"/prefill.trt", "rb") as f, tensorrt.Runtime(tensorrt.Logger()) as runtime:
                self.prefill_engine = runtime.deserialize_cuda_engine(f.read())
                address = CUASSERT(cudart.cudaMalloc(self.prefill_engine.device_memory_size))[0]
            self.prefill_context = self.prefill_engine.create_execution_context_without_device_memory()
            self.prefill_context.set_optimization_profile_async(0, self.cuda_stream)
            self.prefill_context.device_memory = address
            with open(ea_engine_path+"/generate.trt", "rb") as f, tensorrt.Runtime(tensorrt.Logger()) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
                address = CUASSERT(cudart.cudaMalloc(self.engine.device_memory_size))[0]
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.set_optimization_profile_async(0, self.cuda_stream)
            self.context.device_memory = address
            self._set_tensor(self.engine, self.context, "past_key_out", self.trt_k_out)
            self._set_tensor(self.engine, self.context, "past_val_out", self.trt_v_out)
            self.trt_engine = True


    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffers(self.tree,self.embed_tokens.weight.device)


    def reset(self):
        self.tree_mask=None


    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                #inputs_embeds.dtype,
                torch.float32, # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            tree_len = tree_mask.size(-1)
            combined_attention_mask[:, :, -tree_len:, -tree_len:][
                tree_mask == 0
                ] = torch.finfo(torch.float32).min


        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head=None,
    ):
        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids).reshape(hidden_states.shape)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))
        # print("cnet fc hidden_states", hidden_states)
        layer_outputs = self.layers[0](
            hidden_states,
            attention_mask=(input_ids is None),
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states, kv_cache = layer_outputs

        # print("cnet decode hidden_states", hidden_states)
        # hidden_states = self.fc_back(hidden_states[:,-1:])

        return hidden_states[:,-1:], kv_cache

    @torch.no_grad()
    def generate(self,hidden_states,input_ids,head,max_length=4,use_cache=False):
        return_input_ids=copy.deepcopy(input_ids[0].tolist())
        input_ids=input_ids[:,1:]

        #input_ids=input_ids.to(hidden_states.device)
        if use_cache:
            past_key_values=None
            for i in range(max_length):
                if past_key_values!=None:
                    out_hidden,past_key_values = self(out_hidden[:, -1:], input_ids=torch.tensor([[token]]).to(input_ids.device),past_key_values=past_key_values,use_cache=True)
                else:
                    out_hidden, past_key_values = self(hidden_states, input_ids=input_ids,use_cache=True)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                #input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                return_input_ids.append(token.item())
                if token == 2:
                    break
                #hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)
        else:
            for i in range(max_length):
                out_hidden=self(hidden_states,input_ids=input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                return_input_ids.append(token.item())
                input_ids = torch.cat((input_ids, torch.tensor([[token]]).to(input_ids.device)), dim=1)
                if token==2:
                    break
                hidden_states = torch.cat((hidden_states, out_hidden[:, -1:]), dim=1)

        return return_input_ids

    @torch.no_grad()
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self, new_len: int=0):
        self.stable_kv_len = new_len

    def revert_kv(self, revert_len: int=0):
        self.stable_kv_len -= revert_len

    @torch.no_grad()
    def repeat_hidden(self,hidden_state,repeat_num):
        new_hidden=[]
        for id,i in enumerate(repeat_num):
            new_hidden.append(hidden_state[:,id:id+1].repeat(1,i,1))
        return torch.cat(new_hidden,dim=1)

    # @torch.no_grad()
    # def sample(self,tensor,k=1,replacement=True):
    #     probabilities = torch.nn.functional.softmax(tensor, dim=1)
    #     sampled_indices = torch.multinomial(probabilities, k,replacement=replacement)
    #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
    #
    #     return  sampled_indices,sampled_probs

    def sample(self,logits, logits_processor,k=1, replacement=False):
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

        return sampled_indices, sampled_probs,probabilities

        # if replacement:
        #     sampled_indices = torch.multinomial(probabilities, k, replacement=True)
        #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
        #     return sampled_indices, sampled_probs
        # else:
        #     sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        #     sampled_probs = torch.gather(probabilities, 1, sampled_indices)
        #
        #     cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        #     cumulative_sum = torch.cat((torch.zeros(cumulative_sum.shape[0],1, device=cumulative_sum.device), cumulative_sum[:, :-1]),dim=-1)
        #
        #     sampled_probs=sampled_probs/(1-cumulative_sum)
        #     sampled_probs[torch.isinf(sampled_probs)] = -1
        #     sampled_probs[torch.isnan(sampled_probs)] = -1
        #
        #     sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
        #
        #     # has_nan = torch.isnan(sampled_probs).any()
        #     # if has_nan:
        #     #     print(1)
        #
        #     # sampled_probs_list=sampled_probs[0].tolist()
        #     # sum_list=[1-sum(sampled_probs_list[:i]) for i in range(len(sampled_probs_list))]
        #     # for i in range(len(sampled_probs_list)):
        #     #     a=sampled_probs_list[i]/(sum_list[i])
        #     #     if sum_list[i]==0:
        #     #         sampled_probs_list[i]=1.0
        #     #     else:
        #     #         sampled_probs_list[i]=sampled_probs_list[i]/(sum_list[i])
        #     # sampled_probs=torch.tensor([sampled_probs_list],device=sampled_probs.device)
        #
        #
        #
        #     return sampled_indices, sampled_probs

    def _set_tensor(self, engine, context: tensorrt.IExecutionContext,
                    name, tensor):
        if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT:
            context.set_input_shape(name, list(tensor.size()))
        context.set_tensor_address(name, tensor.data_ptr())

    def _check_tensors(self, engine, context: tensorrt.IExecutionContext) -> None:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            ptr = context.get_tensor_address(name)
            if ptr == 0:
                raise RuntimeError(f"Engine I/O tensor {name} is unbound")
            # print("tensor_name", name)
            # print("ptr", ptr)

    @torch.no_grad()
    def topOne_genrate(self, hidden_states, inputs_embeds, head, logits_processor, max_length=10, prefill=False, end_ids=None):
        ss_token = []
        # self.reset()
        # print("self.stable_kv_len", self.stable_kv_len)
        # print("hidden_states", hidden_states)
        # print("inputs_embeds", inputs_embeds)
        batch_size, seq_length, _ = hidden_states.shape
        inputs_embeds = inputs_embeds[:,self.stable_kv_len:, :].reshape(hidden_states.shape)
        position_ids = torch.arange(
            self.stable_kv_len,
            self.stable_kv_len + seq_length,
            dtype=torch.long,
            device=hidden_states.device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # if prefill:
            # torch.onnx.export(
                # self,
                # {
                  # 'hidden_states': hidden_states,
                  # 'inputs_embeds': inputs_embeds,
                  # 'position_ids': position_ids,
                # },
                # "trt-eagle/prefill.onnx",
                # input_names=["hidden_states", "inputs_embeds", "position_ids"], output_names=["out_hidden", "past_key_out", "past_val_out"],
                # dynamic_axes={
                    # "hidden_states": {1: "seq_len"},
                    # 'inputs_embeds': {1: "seq_len"},
                    # "position_ids": {1: "seq_len"},
                    # "past_key_out": {2: "seq_len"},
                    # "past_val_out": {2: "seq_len"},
                # }
            # )
            # exit()
        # if not prefill:
            # torch.onnx.export(
                # self,
                # {
                  # 'hidden_states': hidden_states,
                  # 'inputs_embeds': inputs_embeds,
                  # 'past_key_values': self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :],
                  # 'position_ids': position_ids,
                # },
                # "trt-eagle/context.onnx",
                # input_names=["hidden_states", "inputs_embeds", "past_key_values", "position_ids"], output_names=["out_hidden", "past_key_out", "past_val_out"],
                # dynamic_axes={
                    # "hidden_states": {1: "seq_len"},
                    # 'inputs_embeds': {1: "seq_len"},
                    # "past_key_values": {3: "base_seq_len"},
                    # "position_ids": {1: "seq_len"},
                    # "past_key_out": {2: "seq_len"},
                    # "past_val_out": {2: "seq_len"},
                # }
            # )
            # exit()

        # self.torch_stream.synchronize()
        start = time.time()
        if self.trt_engine:
            key_cache = torch.empty((1, self.num_key_value_heads, hidden_states.shape[1],
                                    self.hidden_size // self.num_key_value_heads),
                                    device=hidden_states.device, dtype=hidden_states.dtype)
            val_cache = torch.empty((1, self.num_key_value_heads, hidden_states.shape[1],
                                    self.hidden_size // self.num_key_value_heads),
                                    device=hidden_states.device, dtype=hidden_states.dtype)
            if prefill:
                engine = self.prefill_engine
                context = self.prefill_context
            else:
                engine = self.context_engine
                context = self.context_context
                trt_kv = self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :].detach().clone()
                self._set_tensor(engine, context, "past_key_values", trt_kv)
            self._set_tensor(engine, context, "hidden_states", hidden_states)
            self._set_tensor(engine, context, "inputs_embeds", inputs_embeds)
            self._set_tensor(engine, context, "position_ids", position_ids)
            self._set_tensor(engine, context, "out_hidden", hidden_states)
            self._set_tensor(engine, context, "past_key_out", key_cache)
            self._set_tensor(engine, context, "past_val_out", val_cache)
            self._check_tensors(engine, context)
            ok = context.execute_async_v3(self.cuda_stream)
            if not ok:
                raise RuntimeError(f"Executing TRT engine failed step={step}!")
        else:
            hidden_states, kv_cache = self(
                hidden_states,
                inputs_embeds=inputs_embeds,
                past_key_values=None if prefill else self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :],
                position_ids=position_ids,
                attention_mask=hidden_states,
            )
            key_cache, val_cache = kv_cache

        # print("1 hidden_states", hidden_states)
        # print("inputs_embeds", inputs_embeds.dtype)
        # print("position_ids", position_ids.dtype)
        # print("hidden_states", hidden_states.dtype)
        # print("self.stable_kv_len", self.stable_kv_len)
        # exit()

        self.stable_kv[0, :batch_size, :, self.stable_kv_len:self.stable_kv_len+seq_length, :] = key_cache
        self.stable_kv[1, :batch_size, :, self.stable_kv_len:self.stable_kv_len+seq_length, :] = val_cache
        self.stable_kv_len += seq_length
        last_hidden = hidden_states[:, 0]
        if self.use_small_head:
            last_headout = self.small_head(last_hidden)
        else:
            last_headout = head(last_hidden)

        hidden_states = hidden_states[:, :1]
        for i in range(max_length-1):
            input_ids = torch.argmax(last_headout, dim=-1).to(torch.int32)
            if self.use_small_head:
                input_ids = self.indexes[input_ids]
            ss_token.append(input_ids)
            # 108704 is "文本"
            # if (end_ids in ss_token) or (108704 in ss_token):
            # if end_ids in ss_token:
                # return (torch.cat(ss_token),None,None)

            batch_size, seq_length, _ = hidden_states.shape
            position_ids = torch.arange(
                self.stable_kv_len,
                self.stable_kv_len + seq_length,
                dtype=torch.long,
                device=hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length).detach().clone()

            # torch.onnx.export(
                # self,
                # {
                  # 'hidden_states': hidden_states,
                  # 'input_ids': input_ids,
                  # 'past_key_values': self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :],
                  # 'position_ids': position_ids,
                # },
                # "trt-eagle/generate.onnx",
                # input_names=["hidden_states", "input_ids", "past_key_values", "position_ids"], output_names=["out_hidden", "past_key_out", "past_val_out"],
                # dynamic_axes={
                    # "past_key_values": {3: "seq_len"},
                # }
            # )
            # exit()

            if self.trt_engine:
                trt_kv = self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :].detach().clone()
                self._set_tensor(self.engine, self.context, "hidden_states", hidden_states)
                self._set_tensor(self.engine, self.context, "input_ids", input_ids)
                self._set_tensor(self.engine, self.context, "past_key_values", trt_kv)
                self._set_tensor(self.engine, self.context, "position_ids", position_ids)
                self._set_tensor(self.engine, self.context, "out_hidden", hidden_states)
                self._check_tensors(self.engine, self.context)
                ok = self.context.execute_async_v3(self.cuda_stream)
                if not ok:
                    raise RuntimeError(f"Executing TRT engine failed step={step}!")
            else:
                hidden_states, kv_cache = self(
                    hidden_states,
                    input_ids=input_ids,
                    past_key_values=self.stable_kv[:, :batch_size, :, :self.stable_kv_len, :],
                    position_ids=position_ids,
                )
                self.trt_k_out, self.trt_v_out = kv_cache

            self.stable_kv[0, :batch_size, :, self.stable_kv_len:self.stable_kv_len+seq_length, :] = self.trt_k_out
            self.stable_kv[1, :batch_size, :, self.stable_kv_len:self.stable_kv_len+seq_length, :] = self.trt_v_out
            # print("self.stable_kv", self.stable_kv)
            # print("out_hidden", hidden_states)
            self.stable_kv_len += 1
            if self.use_small_head:
                last_headout = self.small_head(hidden_states[:, -1])
            else:
                last_headout = head(hidden_states[:, -1])

        # self.torch_stream.synchronize()
        end = time.time()
        self.eagle_gen_time += end - start
        # print("eagle_gen_time", self.eagle_gen_time)
        self.eagle_gen_time = 0.
        input_ids = torch.argmax(last_headout, dim=-1).to(torch.int32)
        if self.use_small_head:
            input_ids = self.indexes[input_ids]
        ss_token.append(input_ids)

        return (torch.cat(ss_token),None,None)


    @torch.no_grad()
    def topK_genrate(self, hidden_states, inputs_embeds, head, logits_processor,max_length=4, use_cache=True):
        # test_=input_ids
        # input_ids = torch.tensor([state[1:]])
        inputs_embeds = inputs_embeds[:, 1:, :]
        inputs_embeds = inputs_embeds.to(hidden_states.device)
        ss_token,ss_prob,ss_op = [],[],[]
        len_posi=inputs_embeds.shape[1]
        self.reset()
        
        if use_cache:
            if hasattr(self, "stable_kv") and self.stable_kv is not None:
                kv_len=self.stable_kv_len
                # print("small self.stable_kv[0][0].shape", self.stable_kv[0][0].shape)
                # print("small inputs_embeds.shape", inputs_embeds.shape)
                out_hidden, past_key_values = self(hidden_states, inputs_embeds=inputs_embeds[:,kv_len:, :], past_key_values=self.stable_kv,use_cache=True)
            else:
                out_hidden, past_key_values = self(hidden_states, inputs_embeds=inputs_embeds, use_cache=True)
            # self.stable_kv=past_key_values
            # print("small 2 self.stable_kv[0][0].shape", self.stable_kv[0][0].shape)
            last_hidden = out_hidden[:, -1]
            if not self.diff_device:
                last_headout = head(last_hidden)
            else:
                if hasattr(self, "layer_device"):
                    last_headout = head(last_hidden)
                    last_headout=last_headout.to(self.layer_device)
                else:
                    last_headout=F.linear(last_hidden,self.headweight)



            for i in range(len(self.tree_buffer['tree_indices'])):
                if logits_processor is not None:
                    topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
                else:
                    top=torch.topk(last_headout, top_k, dim=-1)
                    topk_index,topk_prob = top.indices,top.values
                    op=None

                ss_token.append(topk_index)
                ss_prob.append(topk_prob)
                ss_op.append(op)
                #topk_index = torch.topk(last_headout, top_k, dim=-1).indices
                topk_index = topk_index.view(-1)
                select_index=topk_index[self.tree_buffer['tree_indices'][i]]
                #len_sq=select_index.shape[0]
                input_ids=select_index[None,:]
                if i==0:
                    hidden_states = out_hidden[:, -1:]
                else:
                    hidden_states=out_hidden
                hidden_states=self.repeat_hidden(hidden_states,self.tree_buffer["repeat_nums"][i])
                #hidden_states = hidden_states.repeat(1,len_sq,1)
                self.tree_mask=self.tree_buffer['attn_mask'][i]
                position_ids=len_posi+self.tree_buffer["position_ids"][i]
                # print("topk_index", topk_index)
                # print("select_index", select_index)
                # print("input_ids", input_ids)
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, past_key_values=past_key_values,
                                                   position_ids=position_ids,use_cache=True)
                len_posi += 1

                if not self.diff_device:
                    last_headout = head(out_hidden[0])
                else:
                    if hasattr(self, "layer_device"):
                        last_headout = head(out_hidden[0])
                        last_headout = last_headout.to(self.layer_device)
                    else:
                        last_headout = F.linear(out_hidden[0], self.headweight)
                #last_headout = head(out_hidden[0])
                #sslogits.append(last_headout)
                #print(select_index)

            if logits_processor is not None:
                topk_index,topk_prob,op=self.sample(last_headout,logits_processor,k=top_k,)
            else:
                top = torch.topk(last_headout, top_k, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                op=None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

        else:
            # TODO
            pass

        return (torch.cat(ss_token),torch.cat(ss_prob),ss_op)




    @torch.no_grad()
    def acc(self,data,head,max_length=5):
        hidden_states=data["hidden_states"]
        input_ids=data["input_ids"]
        #attention_mask=data["attention_mask"]
        loss_mask=data["loss_mask"]
        sample_mask=data["sample_mask"]
        target=data["target"]
        total=[0 for _ in range(max_length)]
        correct=[0 for _ in range(max_length)]
        bs,sl=hidden_states.shape[0],hidden_states.shape[1]
        target_headout = head(target)
        hidden_states_headout=head(hidden_states)

        for i in range(bs):
            for j in range(sl):
                if loss_mask[i,j]==0:
                    continue
                single_hidden_states=hidden_states[i,:j]
                single_input_ids=input_ids[i,:j]


                single_hidden_states = single_hidden_states[None, :, :]
                single_input_ids = single_input_ids[None, :]
                for k in range(max_length):
                    tmp_in_target_headout = hidden_states_headout[i,single_hidden_states.shape[1]-1]
                    tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1]-1]
                    target_in_token = torch.argmax(tmp_in_target_headout)
                    target_out_token = torch.argmax(tmp_out_target_headout)
                    tmp_token=input_ids[i,single_hidden_states.shape[1]-1]
                    tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                    if not (target_in_token==tmp_token):
                        break
                    out_hidden = self(single_hidden_states, input_ids=single_input_ids)
                    last_hidden = out_hidden[:, -1]
                    last_headout = head(last_hidden)
                    token = torch.argmax(last_headout)
                    total[k] += 1
                    if token==target_out_token:
                        correct[k]+=1
                    else:
                        for kk in range(k,max_length):
                            total[kk]+=1
                        break

                    single_hidden_states=torch.cat((single_hidden_states,out_hidden[:,-1:]),dim=1)
                    single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)), dim=1)


        acc=[correct[i]/total[i] for i in range(len(correct))]
        return acc





class Vhead(nn.Module):
    def __init__(self,ins=6566,outs=32000):
        super().__init__()
        self.fc = nn.Linear(ins,outs,bias=False)
    def forward(self,x):
        return self.fc(x)



import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__=="__main__":
    config = EConfig.from_pretrained('config.json')
    model = Model(config,load_emb=True,path="/home/lyh/weights/hf/vicuna_v13/7B/")
    print(model)
