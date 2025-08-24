#!/bin/bash
python  transformer_to_transnormer.py --model_name /home/data/models/Qwen3-30B-A3B --checkpoint_folder  /home/data/models/Qwen3-30B-A3B-transnormer --expand 2 --attn_layers "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47"
