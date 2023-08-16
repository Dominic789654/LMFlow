#!/bin/bash
# Please run this script under ${project_id} in project directory of
#   https://github.com/shizhediao/llm-ft
#     COMMIT: d5fecf30ba8011067b10cf51fede53a5ab6574e4

#export HF_DATASETS_IN_MEMORY_MAX_SIZE=1e5

deepspeed_args="--master_port=11000"      # Default argument
if [ $# -ge 1 ]; then
  deepspeed_args="$1"
fi

exp_id=xl_0153_ft_1
project_dir=$(cd "$(dirname $0)"/..; pwd)
output_dir=${project_dir}/output_models/${exp_id}
log_dir=${project_dir}/log/${exp_id}

# dataset_path=${project_dir}/data/143G_zh_cc100_sci_tbfull_trans_wiki_en_bookc_wiki/train_10_parts/
dataset_path=${project_dir}/data/high_SFT_data_0501

eval_dataset_path=${project_dir}/data/143G_zh_cc100_sci_tbfull_trans_wiki_en_bookc_wiki/eval/

mkdir -p ${output_dir} ${log_dir}

    # --block_size 512 \
    # --resume_from_checkpoint="/home/xiangliu/experiments/lmflow-001-multimachine/LMFlow/output_models/xl_0153_test_5/checkpoint-100" \

deepspeed ${deepspeed_args} \
  examples/finetune.py \
    --model_name_or_path /home/xiangliu/experiments/lmflow-001-multimachine/LMFlow/output_models/xl_0153/checkpoint-31826 \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} --overwrite_output_dir \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --block_size 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --use_lora 0 \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 \
    --run_name ${exp_id} \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --evaluation_strategy "no" \
    --eval_steps 15 \
    --eval_dataset_path ${eval_dataset_path} \
    --ddp_timeout 72000 \
    --save_strategy "no" \
    --save_steps 1 \
    --weight_decay 0 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err