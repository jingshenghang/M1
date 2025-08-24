# Copyright (c) 2023, Tri Dao, Albert Gu.

import math  
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
from transformers import LlamaConfig
# from transformers.cache_utils import Cache  #, DynamicCache, StaticCache
from transnormer.cache import CustomDynamicCache
from transformers.activations import ACT2FN

import os
do_eval = eval(os.environ.get("do_eval", default="False"))
BLOCK = 256

# from .configuration_transnormer import TransnormerConfig
# from .norm import SimpleRMSNorm as SimpleRMSNorm_torch
# from .srmsnorm_triton import SimpleRMSNorm as SimpleRMSNorm_triton
# from .utils import (
#     get_activation_fn,
#     get_norm_fn,
#     logging_info,
#     print_module,
#     print_params,
# )

# logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TransnormerConfig"

# TODO: fix environment: https://huggingface.co/OpenNLPLab/TransNormerLLM-7B/discussions/1
use_triton = eval(os.environ.get("use_triton", default="True"))
debug = eval(os.environ.get("debug", default="False"))
do_eval = eval(os.environ.get("do_eval", default="False"))
eval_and_not_generate = eval(os.environ.get("eval_and_not_generate", default="False"))
BLOCK = 256

if use_triton:
    try:
        from .lightning_attention2 import lightning_attention

        has_lightning_attention = True
    except (ImportError, ModuleNotFoundError):
        has_lightning_attention = False
else:
    has_lightning_attention = False

if debug:
    logger.info(f"Use triton: {use_triton}")
    logger.info(f"Use lightning attention: {has_lightning_attention}")
    logger.info(f"Debug mode: {debug}, {type(debug)}")

if not has_lightning_attention:
    def linear_attention(q, k, v, attn_mask):
        energy = torch.einsum("... n d, ... m d -> ... n m", q, k)
        energy = energy * attn_mask
        output = torch.einsum("... n m, ... m d -> ... n d", energy.to(dtype=v.dtype), v)

        return output

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):   
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output

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


class NormLinearAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None): #, gating=False):   #no gating gives better performance
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        # self.intermediate_size = config.intermediate_size
        self.intermediate_size = config.hidden_size   #need to do it for post-training linearization, otherwise hard to modify

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        expands = 2
        self.head_dim = self.hidden_size // self.num_heads * expands # Qwen3 version

        # set bias True to match Qwen
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size * expands, self.hidden_size, bias=False)

        self.normalize = SimpleRMSNorm(self.intermediate_size)     
        self.act_fn = ACT2FN["silu"]

        ### Adding convolution - TODO uncomment if need conv
        self.conv_kernel_size = 4
        self.use_conv_bias = True
        self.q_conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=self.num_key_value_heads * self.head_dim, #self.intermediate_size,
            out_channels=self.num_key_value_heads * self.head_dim, #self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.num_key_value_heads * self.head_dim, #self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )
        #initialize the conv weights to be identity
        self._init_conv_to_one()

        # for inference only
        self.offset = 0

    def _init_conv_to_one(self):
        with torch.no_grad():
            self.q_conv1d.weight.zero_()
            self.k_conv1d.weight.zero_()
            center_index = self.conv_kernel_size - 1
            for channel in range(self.intermediate_size):
                self.q_conv1d.weight[channel, 0, center_index] = 1
                self.q_conv1d.weight[channel, 0, :] += torch.randn_like(self.q_conv1d.weight[channel, 0, :]) * 0.0001
            for channel in range(self.num_key_value_heads * self.head_dim):
                self.k_conv1d.weight[channel, 0, center_index] = 1
                self.k_conv1d.weight[channel, 0, :] += torch.randn_like(self.k_conv1d.weight[channel, 0, :]) * 0.0001
            if self.use_conv_bias:
                self.q_conv1d.bias.zero_()    
                self.k_conv1d.bias.zero_()    

    def forward(
        self,
        hidden_states: torch.Tensor,
        lin_attention_mask: Optional[torch.Tensor] = None, # (b, h, n, m)
        padding_mask: Optional[torch.Tensor] = None,  # (b, m)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None, 
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (not self.training) and (not do_eval):
            return self.inference(
                hidden_states,
                lin_attention_mask,
                padding_mask,
                output_attentions,
                past_key_value,
                use_cache,
                slope_rate,
            )
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)      
        #apply convolution 
        q = rearrange(q, "b l d -> b d l")
        k = rearrange(k, "b l d -> b d l")
        q = self.q_conv1d(q)[...,:q.shape[-1]]
        k = self.k_conv1d(k)[...,:k.shape[-1]]
        q = rearrange(q, "b d l -> b l d")
        k = rearrange(k, "b d l -> b l d")      
        # reshape
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)   
        k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_key_value_heads),
            [k, v])      
        # act
        q = self.act_fn(q)
        k = self.act_fn (k)

        # import pdb
        # pdb.set_trace()
        key_states, value_states = past_key_value.update(k, v, self.layer_idx, {}) if use_cache else None
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
       
        # #if attention_mask is not None:  # no matter the length, we just slice it
        # causal_mask = lin_attention_mask[:, :, :, : key_states.shape[-2]]
        if padding_mask is not None:
            v = v.masked_fill(
                (1 - padding_mask).unsqueeze(1).unsqueeze(-1).to(
                    torch.bool), 0)

        if not has_lightning_attention:
            # if slope_rate is not None:
            lin_mask = torch.exp(slope_rate * lin_attention_mask)
            output = linear_attention(q, k, v, lin_mask)
        else:
            output = lightning_attention(q, k, v, True,
                                         slope_rate.squeeze(-1).squeeze(-1))

        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.normalize(output)
        # # gate
        # if self.gating:
        #     output = u * output
        
        # outproj
        output = self.o_proj(output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = torch.einsum("... n d, ... m d -> ... n m", q, k)

        return output, attn_weights, past_key_value

    def inference(
            self,
            hidden_states,
            lin_attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
            padding_mask: Optional[torch.Tensor] = None,  # (b, m)
            output_attentions: bool = False,
            past_key_value: Optional[torch.Tensor] = None,  #corrected original
            use_cache: bool = False,
            slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)      
        #apply convolution 
        q = rearrange(q, "b l d -> b d l")  #size bdl
        k = rearrange(k, "b l d -> b d l")  #size bd'l (smaller because of group query)

        if past_key_value.conv_states_q[self.layer_idx] is not None:
            # q
            conv_states_q = past_key_value.conv_states_q[self.layer_idx]  # b d 4
            conv_states_q = torch.roll(conv_states_q, shifts=-1, dims=-1)
            conv_states_q[:, :, -1] = q[:, :, 0]
            past_key_value.conv_states_q[self.layer_idx] = conv_states_q.clone()
            q = torch.sum(conv_states_q * self.q_conv1d.weight[:, 0, :], dim=-1)

            if self.use_conv_bias:
                q += self.q_conv1d.bias
            q = q.unsqueeze(dim=-1)

            # k
            conv_states_k = past_key_value.conv_states_k[self.layer_idx]
            conv_states_k = torch.roll(conv_states_k, shifts=-1, dims=-1)
            conv_states_k[:, :, -1] = k[:, :, 0]
            past_key_value.conv_states_k[self.layer_idx] = conv_states_k.clone()
            k = torch.sum(conv_states_k * self.k_conv1d.weight[:, 0, :], dim=-1)

            if self.use_conv_bias:
                k += self.k_conv1d.bias
            k = k.unsqueeze(dim=-1)

        else:
            past_key_value.conv_states_q[self.layer_idx] = nn.functional.pad(
                q,
                (self.conv_kernel_size - q.shape[-1], 0)
            )
            past_key_value.conv_states_k[self.layer_idx] = nn.functional.pad(
                k,
                (self.conv_kernel_size - k.shape[-1], 0)
            )
            q = self.q_conv1d(q)[..., :q.shape[-1]]
            k = self.k_conv1d(k)[..., :k.shape[-1]]

        q = rearrange(q, "b d l -> b l d")
        k = rearrange(k, "b d l -> b l d")   

        # reshape
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)   
        k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_key_value_heads),
            [k, v]) 

        # act
        q = self.act_fn(q)
        k = self.act_fn (k)
        #repeat for group query
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        if past_key_value == None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1
        
        ratio = torch.exp(-slope_rate)

        b, h, n, d = q.shape

        # only use for the first time
        if (
            past_key_value == None or len(past_key_value) <= self.layer_idx or
            past_key_value[self.layer_idx] is None or len(past_key_value[self.layer_idx])==0 
        ):
            slope_rate = slope_rate.to(torch.float32)
            if padding_mask is not None:
                v = v.masked_fill(
                    (1 - padding_mask).unsqueeze(1).unsqueeze(-1).to(
                        torch.bool), 0)
                
            e = v.shape[-1]
            NUM_BLOCK = (n + BLOCK - 1) // BLOCK

            # other
            array = torch.arange(BLOCK).to(q) + 1  ## !!!! important
            q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
            index = array[:, None] - array[None, :]
            s_index = slope_rate * index[
                None,
                None,
            ]
            s_index = torch.where(index >= 0, -s_index, float("-inf"))
            diag_decay = torch.exp(s_index)

            kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si

                qi = q[:, :, si:ei].contiguous()
                ki = k[:, :, si:ei].contiguous()
                vi = v[:, :, si:ei].contiguous()
                qkv_none_diag = torch.matmul(qi * q_decay[:, :m],
                                             kv).to(torch.float32)
                                             

                # diag
                qk = torch.matmul(qi, ki.transpose(-1, -2)).to(
                    torch.float32) * diag_decay[:, :, :m, :m]
                qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                block_decay = torch.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + torch.matmul(
                    (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)
        else:
            kv = past_key_value[self.layer_idx][0]

            output = []
            for i in range(n):
                kv = ratio * kv + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i:i + 1],
                    v[:, :, i:i + 1],
                )
                qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :,
                                                                    i:i + 1],
                                   kv.to(q.dtype))
                output.append(qkv)
            output = torch.concat(output, dim=-2)

        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.normalize(output)
        # outproj
        output = self.o_proj(output)

        attn_weights = None
        past_key_value.update(kv, None, self.layer_idx, {})
        return output, attn_weights, past_key_value