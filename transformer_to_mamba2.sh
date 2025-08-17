#!/bin/bash
python transformer_to_mamba2.py --model_name /home/data/models/Qwen2.5-0.5B --checkpoint_folder  /home/data/models/Qwen2.5-0.5B-Mamba2-v3 --expand 1 --attn_layers "0,2,4,6,8,10,12,14,16,18,20,22,23"
