#!/bin/bash

if [ $# -ge 1 ]; then
  model="$1"
fi
if [ $# -ge 2 ]; then
  dataset_path="$2"
fi
if [ $# -ge 3 ]; then
  metric="$3"
fi
CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type gpt4_cn_v3 \
    --model_name_or_path ${model}\
    --dataset_path ${dataset_path} \
    --use_ram_optimized_load False \
    --deepspeed examples/ds_config.json \
    --metric ${metric}
