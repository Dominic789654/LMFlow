#!/bin/bash

model=gpt2
lora_args=""
if [ $# -ge 1 ]; then
  model=$1
fi
if [ $# -ge 2 ]; then
  lora_args="--lora_model_path $2"
fi
      # --prompt_structure "###Human: {input_text}###Assistant:" \


CUDA_VISIBLE_DEVICES=1 \
  deepspeed examples/chatbot_test_zh_wiki.py \
      --deepspeed configs/ds_config_chatbot.json \
      --model_name_or_path ${model} \
      ${lora_args} \
      --use_ram_optimized_load False \
      --prompt_structure "{input_text}" \
      --end_string "\n\n"
