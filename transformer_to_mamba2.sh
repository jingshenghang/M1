#!/bin/bash
python transformer_to_mamba2.py --model_name /home/data/models/Qwen2.5-0.5B --checkpoint_folder  /home/data/models/Qwen2.5-0.5B-Mamba2-v2 --expand 2 --attn_layers "0,2,4,6,8,10,12,14,16,17"
