#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

# Parses arguments
model_name_or_path=gpt2
run_name=Llama2-7b_5e-5_bsz50_lisa_n2_k20
# dataset_path=data/alpaca/train
dataset_path=data/gpt4_v2
output_dir=output_models/${run_name}
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
exp_id=finetune
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path /home/liuxiang/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --deepspeed configs/ds_config_zero3.json \
    --fp16 \
    --run_name ${run_name} \
    --validation_split_percentage 0 \
    --logging_steps 10 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 5000 \
    --dataloader_num_workers 1 \
    --gradient_checkpointing False \
    --torch_dtype float16 \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err


    # --use_lora 1 \
    # --lora_r 256 \
    # --lora_target_modules c_attn c_proj c_fc c_proj wtp wpe lm_head \