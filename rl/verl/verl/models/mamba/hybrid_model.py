from torch import Tensor
import torch.nn as nn

from verl.models.mamba.hybrid_mamba_config import MambaConfig
from verl.models.mamba.hybrid_mamba_layer import Mamba
from verl.models.mamba.mha import MHA

from mamba_ssm.ops.triton.layer_norm import RMSNorm
from transformers.activations import ACT2FN

class MLP(nn.Module):
    def __init__(self, d_model, intermediate_size, hidden_act, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, **factory_kwargs)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MHADecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MHADecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mha = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_key_value_heads,
            layer_idx=layer_idx,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=config.hidden_size//config.num_attention_heads,
            rotary_emb_base=config.rope_theta,
            causal=True,
            device=device,
            dtype=dtype,
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mha.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, position_ids = None, *args, **kwargs):
        dtype = hidden_states.dtype
        inference_params = kwargs.pop("inference_params", None)
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        max_seqlen = kwargs.pop("max_seqlen", None)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if cu_seqlens is None:
            hidden_states = self.mha(hidden_states, inference_params=inference_params)
        else:
            hidden_states = self.mha(hidden_states.squeeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inference_params=inference_params)
            hidden_states = hidden_states.unsqueeze(0)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if inference_params is None:
            return (hidden_states, None, None)
        else:
            return hidden_states

class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MambaDecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba(
            d_model=config.d_model, d_inner=config.d_inner, d_xb=config.d_xb, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = MLP(config.d_model, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, position_ids = None, *args, **kwargs):
        inference_params = kwargs.pop("inference_params", None)
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        seq_idx = kwargs.pop("seq_idx", None)

        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.mamba(hidden_states, position_ids=position_ids, cu_seqlens=cu_seqlens, seq_idx=seq_idx, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if inference_params is None:
            return (hidden_states, None, None)
        else:
            return hidden_states
    