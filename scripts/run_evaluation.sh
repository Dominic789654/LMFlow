#!/bin/bash
model_name_or_path="$1"
eval_dataset_path="$2"
deepspeed_args="$3"

# CUDA_VISIBLE_DEVICES=0 \
    deepspeed ${deepspeed_args} examples/evaluation.py \
    --answer_type text_only \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${eval_dataset_path} \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric nll \
    --prompt_structure "###Human: {input}###Assistant:" 

