#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
cd /home2/daiyang/BEHAVIOR
python code/visualize_env_a.py \
    --run-file code/outputs/online_deltasg_2/online_env_a_0002.json \
    --output-dir code/outputs/vis_captures
