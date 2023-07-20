#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 14 ]; then
  deepspeed_args="${15}"
fi

# exp_id=xl_001_sharegpt_v3_0.1_vicuna7b_lora_3epcoh_lr1e-4
exp_id="$1"
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}
dataset_path="$2"
lr="$3"
bs="$4"
model_name_or_path="$5"
use_lora="$6"
ds_config="$7"
num_train_epochs="$8"
gradient_checkpointing="$9"
gradient_accumulation_steps="${10}"
lora_r="${11}"
eval_dataset_path="${12}"
block_size="${13}"
per_device_eval_batch_size="${14}"
mkdir -p ${output_dir} ${log_dir}

# no save 
    # --lora_target_modules q_proj k_proj v_proj o_proj \
    # --gradient_checkpointing ${gradient_checkpointing} \
deepspeed ${deepspeed_args} \
  examples/finetune.py  \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --max_step 100 \
    --use_flash_attention 0 \
    --block_size ${block_size} \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --use_lora ${use_lora} \
    --lora_r ${lora_r} \
    --use_qlora 0 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj  down_proj up_proj\
    --save_aggregated_lora 0 \
    --deepspeed ${ds_config} \
    --bf16 \
    --run_name ${exp_id}\
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --evaluation_strategy "no" \
    --eval_steps 50 \
    --eval_dataset_path ${eval_dataset_path} \
    --ddp_timeout 72000 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --activation_checkpointing ${gradient_checkpointing} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err