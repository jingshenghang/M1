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

# def create_custom_causal_mask(
#     attention_mask: torch.Tensor,
#     key_states: torch.Tensor,
#     first_tokens: int = 0,
#     last_tokens: int = 0,
# ) -> torch.Tensor:
#     """
#     Build a mask that, for every query position, allows attention to:
#       • the first  `first_tokens` *valid* keys (prefix / sink tokens); and
#       • the last   `last_tokens`  *valid* keys (SWA sliding window).

#     Args
#     ----
#     attention_mask : 4-D tensor [B, H, Q, K] with 1 for usable keys and 0 for padding
#                      and (usually) the standard causal triangle already applied.
#     key_states     : key tensor – used only to obtain the actual key length K.  
#     first_tokens   : how many tokens from the start to keep (≥0).
#     last_tokens    : how many tokens from the end   to keep (≥0).

#     Returns
#     -------
#     Tensor shaped [B, H, Q, K] with **0** on allowed positions
#     and **-inf** on masked positions, ready to be added to attention scores.
#     """

#     if first_tokens < 0 or last_tokens < 0:
#         raise ValueError("`first_tokens` and `last_tokens` must be non-negative.")

#     batch, heads, query_len, _ = attention_mask.shape
#     key_len = key_states.shape[-2]

#     # If both windows are 0, simply return the (already causal) attention mask.
#     if first_tokens == 0 and last_tokens == 0:
#         return attention_mask[:, :, :, :key_len]

#     device, dtype = attention_mask.device, attention_mask.dtype

#     # Boolean mask of currently *valid* key positions (padding & future tokens are False)
#     valid_keys = attention_mask.bool()                               # [B, H, Q, K]

#     # How many *valid* keys are visible for each (B, H, Q)
#     valid_counts = valid_keys.sum(dim=-1)                            # [B, H, Q]

#     # Index helper – shape [1,1,1,K] → 0 … K-1
#     arange_idx = torch.arange(key_len, device=device)\
#                         .view(1, 1, 1, key_len)

#     # --- prefix window ---------------------------------------------------------
#     allow_first = arange_idx < first_tokens if first_tokens > 0 else None  # [1,1,1,K] or None

#     # --- suffix window ---------------------------------------------------------
#     if last_tokens > 0:
#         # Start index of the *last*-window for every (B,H,Q):  valid_len - last_tokens
#         start_last = (valid_counts - last_tokens).clamp(min=0)             # [B,H,Q]
#         # Broadcast so it becomes [B,H,Q,K]
#         allow_last = arange_idx >= start_last.unsqueeze(-1)
#     else:
#         allow_last = None

#     # Combine the two windows (logical OR), then intersect with `valid_keys`
#     if allow_first is not None and allow_last is not None:
#         allowed = (allow_first | allow_last) & valid_keys
#     elif allow_first is not None:   # only prefix
#         allowed = allow_first & valid_keys
#     else:                           # only suffix
#         allowed = allow_last & valid_keys

#     # Build the final additive mask: 0 for allowed, -inf for everything else
#     mask = torch.full_like(attention_mask[:, :, :, :key_len], float("-inf"), dtype=dtype)
#     mask.masked_fill_(allowed, 0.0)

#     return mask

from typing import Sequence
def create_custom_causal_mask(
   causal_mask: Tensor,
   key_states: Tensor,
   *,
   first_tokens: int = 0,
   last_tokens: int = 0,
   head_policy: Union[Sequence[int], Tensor, None] = None,
) -> Tensor:
   # ---- basic checks --------------------------------------------------------
   if first_tokens < 0 or last_tokens < 0:
       raise ValueError("`first_tokens` and `last_tokens` must be >= 0")

   B, _, Q, K_mask = causal_mask.shape
   H = len(head_policy)
   K = key_states.shape[-2]
   device = causal_mask.device
   dtype = causal_mask.dtype
   neg_inf = torch.finfo(dtype).min

   # ---- 1. normalise head_policy -------------------------------------------
   if head_policy is None:
       head_policy = torch.zeros(H, dtype=torch.bool, device=device)
   else:
       head_policy = torch.as_tensor(head_policy, dtype=torch.bool, device=device)
       if head_policy.numel() != H:
           raise ValueError(f"`head_policy` must have length H={H}")
   build_new = (~head_policy).view(1, H, 1, 1)  # True -> build new mask for this head

   # ---- 2. build 2-D causal pattern (Q x K) ---------------------------------
   q_idx = torch.arange(Q, device=device).unsqueeze(1)  # shape (Q, 1)
   k_idx = torch.arange(K, device=device).unsqueeze(0)  # shape (1, K)

   if last_tokens == 0:
       band = k_idx <= q_idx
   else:
       band = (k_idx <= q_idx) & ((q_idx - k_idx) < last_tokens)

   if first_tokens > 0:
       global_prefix = (k_idx < first_tokens) & (q_idx >= first_tokens)
       band |= global_prefix  # union of conditions

   mask2d = torch.where(
       band,
       torch.zeros((), dtype=dtype, device=device),
       torch.full((), neg_inf, dtype=dtype, device=device),
   )
   new_mask = mask2d.expand(B, H, -1, -1)  # shape (B, H, Q, K)

   # ---- 3. align original mask to new K -------------------------------------
   if K_mask < K:  # need to pad on the right
       pad = causal_mask.new_full((B, H, Q, K - K_mask), neg_inf)
       orig = torch.cat([causal_mask, pad], dim=-1)
   else:           # or crop if K shrank
       orig = causal_mask[..., :K]

   # print(torch.where(build_new, new_mask, orig))
   # ---- 4. mix old and new per head -----------------------------------------
   return torch.where(build_new, new_mask, orig)

# def create_custom_causal_mask(
#    attention_mask: Tensor,
#    key_states: Tensor,
#    *,
#    first_tokens: int = 0,
#    last_tokens: int = 0,
# ) -> Tensor:
#    """
#    Build an *additive* causal mask where each query may
#      • attend to the first `first_tokens` keys globally, and
#      • attend to its own `last_tokens` most-recent keys (a sliding window).
#    All other positions are masked (−∞).

#    Parameters
#    ----------
#    attention_mask : (B, H, Q, K_mask)
#        Any additive mask whose 0/−∞ values we will **overwrite**.
#        (Only its shape is used; content is discarded.)
#    key_states : Tensor
#        Tensor whose penultimate dim is the true key length K.
#    first_tokens : int
#        How many leading key positions are global.
#    last_tokens : int
#        Sliding-window size (≥ 1 ⇒ query can see itself).

#    Returns
#    -------
#    additive_mask : (B, H, Q, K)  with   0.0 = keep,  −∞ = mask
#    """
#    if last_tokens < 0 or first_tokens < 0:
#        raise ValueError("`first_tokens` and `last_tokens` must be ≥ 0.")

#    B, H, Q, _ = attention_mask.shape
#    K = key_states.shape[-2]
#    device, dtype = attention_mask.device, attention_mask.dtype
#    neg_inf = torch.finfo(dtype).min

#    # --- 1. Pad / trim the reference mask to the true KV length -------------
#    if _ < K:
#        pad = attention_mask.new_full((B, H, Q, K - _), neg_inf)
#        ref = torch.cat([attention_mask, pad], dim=-1)
#    else:
#        ref = attention_mask[..., :K]

#    # We'll throw away `ref`'s data, but its batch/head shape is handy.
#    # ------------------------------------------------------------------------

#    # --- 2. Build the (Q, K) sliding-window mask ----------------------------
#    #   keep if  0 ≤ (q − k) < last_tokens
#    q_idx = torch.arange(Q, device=device).unsqueeze(-1)         # (Q,1)
#    k_idx = torch.arange(K, device=device).unsqueeze(0)          # (1,K)

#    if last_tokens > 0:
#        keep_band = (k_idx <= q_idx) & ((q_idx - k_idx) < last_tokens)
#    else:  # last_tokens == 0 ⇒ keep nothing from the sliding part
#        keep_band = torch.zeros(Q, K, dtype=torch.bool, device=device)

#    # --- 3. Add global-prefix columns ---------------------------------------
#    if first_tokens > 0:
#        keep_band[:, :first_tokens] = True                       # broadcast

#    # --- 4. Convert to additive form and broadcast to (B,H, Q, K) -----------
#    mask_2d = torch.where(keep_band,
#                          torch.zeros((), dtype=dtype, device=device),
#                          torch.full((), neg_inf, dtype=dtype, device=device))

#    additive_mask = mask_2d.expand(B, H, -1, -1).clone()

#    return additive_mask


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



class HeadWisePhybridAttention(nn.Module):
    def __init__(self, config: LlamaConfig,
                 layer_idx: Optional[int] = None):  # , gating=False):   #no gating gives better performance
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        # self.intermediate_size = config.intermediate_size
        self.intermediate_size = config.hidden_size  # need to do it for post-training linearization, otherwise hard to modify

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

        self.act_fn = ACT2FN["silu"]

        if self.config.head_perm_dict and str(layer_idx) in self.config.head_perm_dict:
            perm_lst = self.config.head_perm_dict[str(layer_idx)]
            assert len(perm_lst) == self.num_heads
            inverse_perm = [0] * len(perm_lst)
            for i, p in enumerate(perm_lst):
                inverse_perm[p] = i
            # Use register_buffer so these move automatically with .to(), .cuda(), etc.
            self.register_buffer("perm_tensor", torch.tensor(perm_lst, dtype=torch.long))
            self.register_buffer("inverse_perm_tensor", torch.tensor(inverse_perm, dtype=torch.long))
        else:
            self.perm_tensor = None
            self.inverse_perm_tensor = None

        if self.config.head_split_dict and str(layer_idx) in self.config.head_split_dict:
            self.attn_head_num = self.config.head_split_dict[str(layer_idx)]
        else:
            raise ValueError(f"Hybrid Layer {layer_idx} don't have `attn_head_num`")

        # transnormer parameters
        self.tnorm_head_num = self.num_heads - self.attn_head_num
        self.normalize = SimpleRMSNorm(self.tnorm_head_num * self.head_dim)

        # for inference only
        self.offset = 0

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,  # comes from training_args.max_seq_length, #TODO
            base=1000000.0,  # standard in Qwen
        )


    def forward(
            self,
            hidden_states: torch.Tensor,
            lin_attention_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
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
        bsz, q_len, _ = hidden_states.size()
        
        # FIXME
        if self.perm_tensor is not None:
            q = q.view(bsz, q_len, self.num_heads, self.head_dim)
            q = torch.index_select(q, dim=2, index=self.perm_tensor)    
            q = q.view(bsz, q_len, self.hidden_size)

        s_idx = self.head_dim * self.attn_head_num
        # hidden_ze = hn * hd //3
        # split over heads for phybrid - to sizes b l (h d)//2
        query_states = q[..., :s_idx]  # Take first half of embedding to ATTN
        q_tnorm = q[..., s_idx:]  # Take second half to TRANSNORMER

        ### Qwen Attention
        query_states = query_states.view(bsz, q_len, self.attn_head_num, self.head_dim).transpose(1, 2)
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
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.perm_tensor is not None:
            key_states = torch.index_select(key_states, dim=1, index=self.perm_tensor)
            value_states = torch.index_select(value_states, dim=1, index=self.perm_tensor)
        key_states = key_states[:, :self.attn_head_num, ...]
        value_states = value_states[:, :self.attn_head_num, ...]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.attn_head_num, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.attn_head_num, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.attn_head_num, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.attn_head_num, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        ### SWA Attention
        query_states = q_tnorm.view(bsz, q_len, self.tnorm_head_num, self.head_dim).transpose(1, 2)
        key_states = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
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
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.perm_tensor is not None:
            key_states = torch.index_select(key_states, dim=1, index=self.perm_tensor)
            value_states = torch.index_select(value_states, dim=1, index=self.perm_tensor)
        key_states = key_states[:, self.attn_head_num:, ...]
        value_states = value_states[:, self.attn_head_num:, ...]

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.tnorm_head_num, q_len, kv_seq_len):
            # print(self.layer_idx)
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.tnorm_head_num, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            # TODO: add swa mask
            # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            # print(attention_mask)
            causal_mask = create_custom_causal_mask(attention_mask, key_states, first_tokens=10, last_tokens=128, head_policy=[0] * self.tnorm_head_num)
            # print(causal_mask)
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights = nn.functional.dropout(attn_weights, p=0.0, training=self.training)
        swa_output = torch.matmul(attn_weights, value_states)

        if swa_output.size() != (bsz, self.tnorm_head_num, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.tnorm_head_num, q_len, self.head_dim)}, but is"
                f" {swa_output.size()}"
            )


        ### MERGE 

        output = torch.cat([attn_output, swa_output], dim=1)
        if self.perm_tensor is not None:
            output = torch.index_select(output, dim=1, index=self.inverse_perm_tensor)

        output = output.transpose(1, 2).contiguous()
        output = output.reshape(bsz, q_len, self.hidden_size)

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
            past_key_value: Optional[torch.Tensor] = None,  # corrected original
            use_cache: bool = False,
            slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # apply convolution
        q = rearrange(q, "b l d -> b d l")  # size bdl
        k = rearrange(k, "b l d -> b d l")  # size bd'l (smaller because of group query)

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
        k = self.act_fn(k)
        # repeat for group query
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
                past_key_value[self.layer_idx] is None or len(past_key_value[self.layer_idx]) == 0
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

class LlamaMLP(nn.Module):  # original llama MLP
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
        if False:  # self.config.pretraining_tp > 1:
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
        # attn_type is in ["attn", "lin_attn"]
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
            padding_mask: Optional[torch.Tensor] = None,
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
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            slope_rate=slope_rate,
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



