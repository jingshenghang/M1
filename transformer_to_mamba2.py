import argparse
import json
import logging
import os

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from mamba2.hybrid_mamba_config import MambaConfig
from mamba2.hybrid_model import MambaDecoderLayer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a pretrained model into a hybrid Mamba model from a Transformer."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Name or path of the pretrained model.",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="Llama-3.2B-Math-Mamba-Untrained",
        help="Directory to save the converted model and configuration.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch data type to use.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation to use.",
    )
    parser.add_argument(
        "--attn_layers",
        type=str,
        default="3,8,13,18,23,27",
        help="Comma-separated list of layers to use MHADecoderLayer (others will use MambaDecoderLayer).",
    )
    parser.add_argument(
        "--init_with_kqvo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize non-attention layers with kqvo from attention weights.",
    )
    parser.add_argument(
        "--attn_bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        # default=True,
        help="Whether to copy attention bias values.",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=1,
        help="Expansion factor for inner dimension (d_inner).",
    )
    return parser.parse_args()


def save_config(mamba_config, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    config_path = os.path.join(save_directory, "mamba_config.json")
    with open(config_path, "w") as f:
        json.dump(mamba_config.__dict__, f, indent=4)
    logging.info("Mamba configuration saved to %s", config_path)


def convert_layers(
    transformer, mamba_config, attn_layers, init_with_kqvo, attn_bias, torch_dtype
):
    config = transformer.config
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    num_heads_kv = config.num_key_value_heads
    head_dim = embed_dim // num_heads
    q_dim = head_dim * num_heads
    kv_dim = head_dim * num_heads_kv

    for layer_idx in tqdm(range(config.num_hidden_layers)):
        # logging.info("Converting layer %d...", layer_idx)
        # Fetch the layer module for easier access
        layer_module = transformer.model.layers._modules[f"{layer_idx}"]
        if layer_idx not in attn_layers:
            # Use MambaDecoderLayer for the remaining layers
            mamba_encoder = MambaDecoderLayer(
                mamba_config,
                layer_idx,
                device="cpu",
                dtype=torch_dtype,
            )

            mamba_encoder.mlp.load_state_dict(layer_module.mlp.state_dict())
            mamba_encoder.mlp = mamba_encoder.mlp.to(torch_dtype)

            mamba_encoder.input_layernorm.load_state_dict(
                layer_module.input_layernorm.state_dict()
            )
            mamba_encoder.input_layernorm = mamba_encoder.input_layernorm.to(
                torch_dtype
            )

            mamba_encoder.post_attention_layernorm.load_state_dict(
                layer_module.post_attention_layernorm.state_dict()
            )
            mamba_encoder.post_attention_layernorm = (
                mamba_encoder.post_attention_layernorm.to(torch_dtype)
            )

            mamba_encoder.mamba.out_proj.load_state_dict(
                layer_module.self_attn.o_proj.state_dict()
            )
            mamba_encoder.mamba.out_proj = mamba_encoder.mamba.out_proj.to(torch_dtype)

            if init_with_kqvo:
                mamba_encoder.mamba.in_proj.weight.data[
                    mamba_config.d_inner : mamba_config.d_inner + mamba_config.d_xb, :
                ].copy_(layer_module.self_attn.v_proj.weight.data)
                mamba_encoder.mamba.in_proj.weight.data[
                    mamba_config.d_inner
                    + mamba_config.d_xb : mamba_config.d_inner
                    + 2 * mamba_config.d_xb,
                    :,
                ].copy_(layer_module.self_attn.k_proj.weight.data)
                mamba_encoder.mamba.in_proj.weight.data[
                    mamba_config.d_inner
                    + 2 * mamba_config.d_xb : 2 * mamba_config.d_inner
                    + 2 * mamba_config.d_xb,
                    :,
                ].copy_(layer_module.self_attn.q_proj.weight.data)

            transformer.model.layers[layer_idx] = mamba_encoder


def main():
    args = parse_args()

    # Set the appropriate torch data type
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Process comma-separated attention layers into a list of ints
    attn_layers = [int(x.strip()) for x in args.attn_layers.split(",")]

    logging.info("Loading model: %s", args.model_name)
    transformer = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    print(transformer)

    config = transformer.config
    logging.info("Model has %d hidden layers", config.num_hidden_layers)

    # # Calculate derived dimensions for the Mamba configuration
    # d_model = config.hidden_size
    # d_inner = args.expand * config.num_attention_heads * config.head_dim
    # d_xb = config.num_key_value_heads * config.head_dim
    # d_state = config.head_dim
    # ssm_cfg = {
    #     "expand": args.expand,
    #     "d_state": d_state,
    #     "ngroups": config.num_attention_heads,
    # }

    # Calculate derived dimensions for the Mamba configuration
    d_model = config.hidden_size
    d_inner = args.expand * d_model
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    d_xb = config.num_key_value_heads * head_dim
    ssm_cfg = {"expand": args.expand, "ngroups": 4}

    mamba_config = MambaConfig(
        d_model=config.hidden_size,
        ssm_cfg=ssm_cfg,
        rms_norm_eps=config.rms_norm_eps,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        n_layer=config.num_hidden_layers,
        attn_layers=attn_layers,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        norm_topk_prob=config.norm_topk_prob,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
    )

    logging.info("Starting layer conversion...")

    print("args.init_with_kqvo:", args.init_with_kqvo)
    convert_layers(
        transformer,
        mamba_config,
        attn_layers,
        args.init_with_kqvo,
        args.attn_bias,
        torch_dtype,
    )

    # Save the model and config
    logging.info(transformer)
    logging.info("Saving converted model to %s", args.checkpoint_folder)
    transformer.save_pretrained(args.checkpoint_folder, safe_serialization=True)
    save_config(mamba_config, args.checkpoint_folder)
    logging.info("Model conversion complete.")

    logging.info("Loading tokenizer for %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.checkpoint_folder)
    logging.info("Tokenizer saved to %s", args.checkpoint_folder)


if __name__ == "__main__":
    main()
