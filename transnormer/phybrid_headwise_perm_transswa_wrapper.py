# Copyright (c) 2023, Albert Gu, Tri Dao.
import os 
import json 
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


from typing import List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM
from transformers.utils.hub import cached_file

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from transnormer.headwise_perm_transswa_phybrid_model import TransnormerDecoderLayer #main change

from transnormer.hybrid_transnormer_config import TransnormerConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transnormer.cache import CustomDynamicCache

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast

from util import load_safetensors_to_dict

from transformers import PreTrainedModel, GenerationMixin

TRANSNORMER_CONFIG_NAME = "transnormer_config.json"

class TransnormerTransformerHybridModelWrapper(nn.Module, GenerationMixin):

    def __init__(self, checkpoint_path, transformer_model, transnormer_config, attn_layers, dtype, init_with_kqvo, load_from_hub=False, **kwargs):
        super(TransnormerTransformerHybridModelWrapper, self).__init__()
        self.transnormer_config = transnormer_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        self.num_hidden_layers = transnormer_config.num_hidden_layers
        
        for layer_idx in range(self.num_hidden_layers):
            if layer_idx not in attn_layers:
                transnormer_encoder = TransnormerDecoderLayer(
                    transnormer_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                # print("transnormer", transnormer_encoder,transnormer_encoder.self_attn.v_proj.state_dict() )
                # print("transformer", transformer_model.model.layers._modules[f'{layer_idx}'], transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.state_dict())
                
                if init_with_kqvo:
                    # init weights using attention weights
                    transnormer_encoder.mlp.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                    transnormer_encoder.input_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                    transnormer_encoder.post_attention_layernorm.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].post_attention_layernorm.state_dict())
                    transnormer_encoder.self_attn.v_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.state_dict())
                    transnormer_encoder.self_attn.k_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.state_dict())
                    transnormer_encoder.self_attn.q_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.state_dict())
                    transnormer_encoder.self_attn.o_proj.load_state_dict(transformer_model.model.layers._modules[f'{layer_idx}'].self_attn.o_proj.state_dict())  
                    # keep dtype to be the same
                    transnormer_encoder.mlp = transnormer_encoder.mlp.to(dtype)
                    transnormer_encoder.input_layernorm = transnormer_encoder.input_layernorm.to(dtype)
                    transnormer_encoder.post_attention_layernorm = transnormer_encoder.post_attention_layernorm.to(dtype)
                
                self.model.model.layers[layer_idx] = transnormer_encoder

        # for item in self.model.model:
        #     print(item)
        # import pdb
        # pdb.set_trace()
    
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                self.model.load_state_dict(load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype))
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    self.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu")))
                else:
                    # support save from safetensors
                    self.load_state_dict(load_safetensors_to_dict(checkpoint_path))
        
        self.model = self.model.to(dtype).cuda()

        #compute slopes for linear attention
        self._linear_attn_mask_dict = dict() # torch.empty(0)
        # self.slopes = self._build_slope_tensor(transnormer_config.num_attention_heads//2) #TODO because only half 

        self.generation_config = self.model.generation_config  #generation
        self.main_input_name = "input_ids" #generation
        self._supports_cache_class = True
        
        #TODO print conv weight at layer1

    ######################################################################
    ############### START TRANSNORMER MASKING ############################
    ######################################################################
    @staticmethod
    def _build_slope_tensor(n_attention_heads: int):

        def get_slopes(n):

            def get_slopes_power_of_2(n):
                start = 2**(-(2**-(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2])

        # h, 1, 1
        slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
            n_attention_heads, 1, 1)

        return slopes

    def _get_mask(self, n):
        mask = torch.triu(
            torch.zeros(n, n).float().fill_(float("-inf")), 1)
        # no slope version
        # -n, ..., -2, -1, 0
        for i in range(n):
            x = torch.arange(i + 1)
            y = x
            mask[i, :i + 1] = -torch.flip(y, [0])

        return mask
    
    
    def _prepare_decoder_linear_attn_mask(self, input_shape, inputs_embeds,
                                          past_key_values_length,slopes, idx):
        bsz, tgt_len = input_shape
        src_len = tgt_len + past_key_values_length

        def power_log(x):
            return 2**(math.ceil(math.log(x, 2)))

        n = power_log(max(tgt_len, src_len))
        if idx not in self._linear_attn_mask_dict or self._linear_attn_mask_dict[idx].shape[-1] < n:
        # if self._linear_attn_mask.shape[-1] < n:  #TODO - for generation
            def get_mask(n):
                mask = torch.triu(
                    torch.zeros(n, n).float().fill_(float("-inf")), 1)
                # no slope version
                # -n, ..., -2, -1, 0
                for i in range(n):
                    x = torch.arange(i + 1)
                    y = x
                    mask[i, :i + 1] = -torch.flip(y, [0])

                return mask

            arr = []
            mask = get_mask(n)
            for _ in slopes:
                arr.append(mask)
            # _linear_attn_mask = torch.stack(arr, dim=0) #.to(inputs_embeds)
            self._linear_attn_mask_dict[idx] = torch.stack(arr, dim=0).to(inputs_embeds)

        num_heads = self._linear_attn_mask_dict[idx].shape[0]
        if self._linear_attn_mask_dict[idx].shape[-1] != src_len:
            linear_attn_mask = self._linear_attn_mask_dict[idx][:, -tgt_len:, -src_len:]
            return linear_attn_mask[None, :, :, :].expand(bsz, num_heads, tgt_len, src_len)    
        else:
            return self._linear_attn_mask_dict[idx][None, :, :, :].expand(bsz, num_heads, tgt_len, src_len)
         
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if False: #hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    ######################################################################
    ############### END TRANSNORMER MASKING ##############################
    ######################################################################
    

    # def allocate_mamba_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    #     return {
    #         i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    #         for i, layer in enumerate(self.model.model.layers)
    #         if isinstance(layer, MambaDecoderLayer)
    #     }
    def can_generate(self):
        return True  # Required for GenerationMixin
    @property
    def device(self):
        # Dynamically determine the device from the model parameters
        return next(self.parameters()).device

    # def forward(
    #     self,
    #     input_ids,
    #     **kwargs,
    # ):
    #     return self.model(input_ids, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  #[b,seqlen] padding
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        
        # print("Custom forward method called")
        # import pdb
        # pdb.set_trace()
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape   
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
        

        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = CustomDynamicCache.from_legacy_cache(past_key_values=past_key_values,attn_layers=(self.attn_layers, self.num_hidden_layers))
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens) #for llama

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # linear_attn_mask = self._prepare_decoder_linear_attn_mask(
        #     (batch_size, seq_length), inputs_embeds, 0)
        
        # linear_attn_mask = linear_attn_mask + causal_mask # when implement inference, TODO 
        # slope_rates = [
        #     self.slopes.to(input_ids.device) for _ in range(self.num_hidden_layers)
        # ]
        # bsz, tgt_len = batch_size, seq_length
        # src_len = tgt_len + 0
        # def power_log(x):
        #     return 2**(math.ceil(math.log(x, 2)))
        # n = power_log(max(tgt_len, src_len))
        # mask_helper = self._get_mask(n)
        
        for idx, decoder_layer in enumerate(self.model.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # import pdb
            # pdb.set_trace()
            if not isinstance(decoder_layer, TransnormerDecoderLayer): #vanilla attention
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            else:
                # slope_rate = slope_rates[idx]
                tnorm_head_num = self.transnormer_config.num_attention_heads - self.transnormer_config.head_split_dict[str(idx)]
                slopes = self._build_slope_tensor(tnorm_head_num)

                linear_attn_mask = self._prepare_decoder_linear_attn_mask((batch_size, seq_length), inputs_embeds, 0, slopes, idx)
                
                slope_rate = slopes.to(hidden_states.device)
                slope_rate = slope_rate * (1 - idx / (self.num_hidden_layers - 1) + 1e-5)

                layer_outputs = decoder_layer(
                    hidden_states,
                    lin_attention_mask=linear_attn_mask, 
                    padding_mask = attention_mask,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    slope_rate=slope_rate,
                    cache_position=cache_position,)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        
        if False: # self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.model.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + tuple(v for v in [next_cache, all_hidden_states, all_self_attns] if v is not None)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
  
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):  
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def gradient_checkpointing_enable(self,**kwargs):
        """
        Enable gradient checkpointing for the model to save memory during training.
        """
        # self.model.gradient_checkpointing_enable(**kwargs)

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)
            print("Gradient checkpointing enabled for the main model.")
        else:
            print("Gradient checkpointing is not supported by the main model.")

        # Check and enable gradient checkpointing for all layers
        for name, module in self.model.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                print(f"Gradient checkpointing enabled for layer: {name}")
        # import pdb
        # pdb.set_trace()

    # def generate(
    #     self,
    #     input_ids,
    #     **kwargs,
    # ):
    #     output = self.model.generate(
    #         input_ids,
    #         use_cache=True,
    #         **kwargs,
    #     )
    #     return output

    def generate(
        self,
        input_ids,
        max_length=1024,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):

        if kwargs is not None:
            max_new_tokens = kwargs.pop('max_new_tokens', None)
            if max_new_tokens is not None:
                max_length = max_new_tokens + input_ids.shape[1]
            do_sample = kwargs.pop('do_sample', True)
            if not do_sample:
                top_k, top_p, min_p = 1, 0.0, 0.0
            
            cg = kwargs.pop('cg', True)
            eos_token_id = kwargs.pop('eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = self.config.eos_token_id

            attention_mask = kwargs.pop('attention_mask', None)
            pad_token_id = kwargs.pop('pad_token_id', None)
            no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', None)
            length_penalty = kwargs.pop('length_penalty', None)
            num_return_sequences = kwargs.pop('num_return_sequences', None)
            num_beams = kwargs.pop('num_beams', None)
            low_memory = kwargs.pop('low_memory', None)
            stopping_criteria = kwargs.pop('stopping_criteria', None)

        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            # cg=cg,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            eos_token_id=eos_token_id,
            **kwargs,
        )
    
    
    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        transnormer_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)

        return TransnormerTransformerHybridModelWrapper(checkpoint_path, transformer_model, transnormer_config, attn_layers, dtype, init_with_kqvo)
    
    @staticmethod
    def init_random_model(
        checkpoint_path,
        tranformer_name,
        transnormer_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model = AutoModelForCausalLM.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        for param in transformer_model.parameters():
            if param.requires_grad:  # Only reinitialize trainable parameters
                if param.dim() >= 2:  # For weights with 2 or more dimensions
                    nn.init.xavier_uniform_(param)  # Random initialization    
                elif param.dim() == 1:  # For 1D tensors (biases)
                    nn.init.uniform_(param, a=-0.1, b=0.1)  # Uniform distribution in range [-0.1, 0.1]
                else:
                    param.data.fill_(0)  # For scalars, if present (optional)
        return MambaTransformerHybridModelWrapper(checkpoint_path, transformer_model, transnormer_config, attn_layers, dtype, init_with_kqvo=False)



    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        with open(f'{pretrained_model_name}/{TRANSNORMER_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        transnormer_config = TransnormerConfig(**config_dict)
        return TransnormerTransformerHybridModelWrapper(pretrained_model_name, transformer_model, transnormer_config, transnormer_config.attn_layers, torch_dtype, init_with_kqvo=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, TRANSNORMER_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        transnormer_config = MambaConfig(**config_dict)
        return MambaTransformerHybridModelWrapper(pretrained_model_name, transformer_model, transnormer_config, transnormer_config.attn_layers, torch_dtype, init_with_kqvo=False, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"):
        if os.path.exists(pretrained_model_name):
            return TransnormerTransformerHybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation)
        else:
            return MambaTransformerHybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'transnormer_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.transnormer_config.__dict__, f, indent=4)
