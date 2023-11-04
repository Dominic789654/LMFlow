#!/bin/bash
# Please run this script under ${project_id} in project directory of

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 18 ]; then
  deepspeed_args="${19}"
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
warmup_ratio="${15}"
num_portions="${16}"
selected_portion="${17}"
optimizer_name="${18}"
mkdir -p ${output_dir} ${log_dir}

# no save 
  # --lora_target_modules 
    #     --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj\ # llama
    #     --lora_target_modules c_attn c_proj c_fc c_proj \ # gpt2
    #     --lora_target_modules Wqkv out_proj fc1 fc2 \ # phi 1.5
  #       --lora_target_modules k_proj v_proj q_proj out_proj fc_in fc_out \gptj 6b

    # --gradient_checkpointing ${gradient_checkpointing} \
        # --max_steps 150 \
            # --optimizer_name "Lion" \
    # --lr_scheduler_type "cosine" \
        # --config_name ${model_name_or_path} \
    # --tokenizer_name pinkmanlove/llama-7b-hf \
# --model_name_or_path ${model_name_or_path} \
# s    # --lr_scheduler_type "cosine" \
    # --max_steps 300 \


deepspeed ${deepspeed_args} \
  examples/finetune.py  \
    --model_name_or_path ${model_name_or_path} \
    --optimizer_name ${optimizer_name} \
    --dataset_path ${dataset_path} \
    --min_x 1e-2 \
    --max_x 1e2 \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs ${num_train_epochs} \
    --learning_rate ${lr} \
    --use_flash_attention 1 \
    --block_size ${block_size} \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --use_lora ${use_lora} \
    --lora_r ${lora_r} \
    --use_qlora 0 \
    --lora_target_modules c_attn c_proj c_fc c_proj \
    --save_aggregated_lora 1 \
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
    --weight_decay 0.01 \
    --warmup_ratio ${warmup_ratio} \
    --selected_portion ${selected_portion} \
    --num_portions ${num_portions} \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --activation_checkpointing ${gradient_checkpointing} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err