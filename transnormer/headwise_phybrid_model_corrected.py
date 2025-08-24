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

from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb


import os
do_eval = eval(os.environ.get("do_eval", default="False"))
BLOCK = 256


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


def linear_attention(q, k, v, attn_mask):   
        energy = torch.einsum("... n d, ... m d -> ... n m", q, k)    
        energy = energy * attn_mask 
        output = torch.einsum("... n m, ... m d -> ... n d", energy.to(dtype=v.dtype), v)

        return output


class HeadWisePhybridAttention(nn.Module):
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

        # set bias True to match Qwen
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.normalize = SimpleRMSNorm(self.intermediate_size//2)     
        self.act_fn = ACT2FN["silu"]

        ### Adding convolution - TODO uncomment if need conv
        self.conv_kernel_size = 4
        self.use_conv_bias = True
        self.q_conv1d = nn.Conv1d(
            in_channels=self.intermediate_size//2,
            out_channels=self.intermediate_size//2,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size//2,
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

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings= 2048, #comes from training_args.max_seq_length, #TODO
            base=1000000.0,  #standard in Qwen
        )

    def _init_conv_to_one(self):
        with torch.no_grad():
            self.q_conv1d.weight.zero_()
            self.k_conv1d.weight.zero_()
            center_index = self.conv_kernel_size - 1
            for channel in range(self.intermediate_size//2):
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
        attention_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
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

        # split over heads for phybrid - to sizes b l (h d)//2
        query_states = q[..., :self.hidden_size // 2]  # Take first half of embedding to ATTN
        q_tnorm = q[..., self.hidden_size // 2:] # Take second half to TRANSNORMER
        
        ### Qwen Attention 

        bsz, q_len, _ = hidden_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads//2, self.head_dim).transpose(1, 2)
        key_states = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     if self.layer_idx is None:
        #         raise ValueError(
        #             f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
        #             "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
        #             "with a layer index."
        #         )
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)[:,:self.num_heads//2, ...]
        value_states = repeat_kv(value_states, self.num_key_value_groups)[:,:self.num_heads//2, ...]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads//2, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads//2, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads//2, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads//2, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size//2)


        ### Linear Attention 

        #apply convolution 
        q_tnorm = rearrange(q_tnorm, "b l d -> b d l")
        k = rearrange(k, "b l d -> b d l")
        q_tnorm = self.q_conv1d(q_tnorm)[...,:q_tnorm.shape[-1]]
        k = self.k_conv1d(k)[...,:k.shape[-1]]
        q_tnorm = rearrange(q_tnorm, "b d l -> b l d")
        k = rearrange(k, "b d l -> b l d")      
        # reshape
        q_tnorm = rearrange(q_tnorm, "b l (h d) -> b h l d", h=self.num_heads//2)   
        k, v = map(
            lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_key_value_heads),
            [k, v])      
        # act
        q_tnorm = self.act_fn(q_tnorm)
        k = self.act_fn (k)

        # import pdb
        # pdb.set_trace()
        key_states, value_states = past_key_value.update(k, v, self.layer_idx, {}) if use_cache else None
        
        k = repeat_kv(k, self.num_key_value_groups)[:,self.num_heads//2:, ...]
        v = repeat_kv(v, self.num_key_value_groups)[:,self.num_heads//2:, ...]
       
        # #if attention_mask is not None:  # no matter the length, we just slice it
        # causal_mask = lin_attention_mask[:, :, :, : key_states.shape[-2]]
        if padding_mask is not None:
            v = v.masked_fill(
                (1 - padding_mask).unsqueeze(1).unsqueeze(-1).to(
                    torch.bool), 0)

        
        lin_mask = torch.exp(slope_rate * lin_attention_mask)
        
        output_tnorm = linear_attention(q_tnorm, k, v, lin_mask)  
       

        # reshape
        output_tnorm = rearrange(output_tnorm, "b h n d -> b n (h d)")

        # normalize
        output_tnorm = self.normalize(output_tnorm)
        

        ### MERGE
        output = torch.cat([attn_output, output_tnorm], dim=-1)  # (b, n, h)

        # outproj
        output = self.o_proj(output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = torch.einsum("... n d, ... m d -> ... n m", q, k)

        return output, attn_weights, past_key_value

    # TODO rewrite inference
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



### Define Decoder Layer 

class LlamaMLP(nn.Module):  #original llama MLP
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
        if False : #self.config.pretraining_tp > 1:
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

        return down_proj


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




class TransnormerDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, device, dtype):
        #attn_type is in ["attn", "lin_attn"]
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = HeadWisePhybridAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lin_attention_mask: Optional[torch.Tensor] = None,
        padding_mask : Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        slope_rate: Optional[torch.Tensor] = None, 
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # if "padding_mask" in kwargs:
        #     warnings.warn(
        #         "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        #     )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                lin_attention_mask=lin_attention_mask,
                padding_mask = padding_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                slope_rate = slope_rate, 
                cache_position=cache_position,
                **kwargs,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



