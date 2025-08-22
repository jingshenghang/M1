#!/bin/bash
python  transformer_to_mamba2.py --model_name /home/ascend-vllm/model/Qwen3-30B-A3B-Instruct-2507 --checkpoint_folder  /home/ascend-vllm/model/Qwen3-30B-A3B-Instruct-2507-Mamba2 --expand 2 --attn_layers "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47"
