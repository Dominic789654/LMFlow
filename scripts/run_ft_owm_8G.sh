#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
model_name_or_path=meta-llama/Llama-2-7b-hf
dataset_path=./data/OWM_8G
# val_path=./data/gpt4_v2_val/
output_dir=output_models/con_pretrain
deepspeed_args="--master_port=11000"

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_model_path)
      output_dir="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
export WANDB_PROJECT=lisa_continue_pretrain
exp_id=llama2-7b_sft_lr4e-5_owm8g
output_dir=output_models/acl_rebuttal/${exp_id}
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --block_size 512 \
    --per_device_train_batch_size 50 \
    --use_flash_attention 1 \
    --deepspeed configs/ds_config_zero2.json \
    --fp16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --lr_scheduler_type cosine \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err

    # --warmup_ratio 0.06 \
    # --weight_decay 0.01 \

    # --eval_dataset_path ${val_path} \
    # --do_eval \
    # --evaluation_strategy steps \
    # --eval_steps 5 \
        # --use_lisa 1 \
    # --lisa_step_interval 1 \