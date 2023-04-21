exp_id=${1}
metric=${2}
model_name=${3}
lora_model_name=${4}

log_dir=output_dir/${exp_id}_${metric}

if [ -d "${log_dir}" ]; then
 echo "${log_dir} exists"
else
 mkdir -p ${log_dir}
fi

lora_args=""
if [ $# -ge 3 ]; then
  model=$2
fi
if [ $# -ge 4 ]; then
  lora_args="--lora_model_path ${lora_model_name}"
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text2text \
    --model_name_or_path ${model_name} \
    ${lora_args} \
    --dataset_path /home/xiangliu/LMFlow/data/gpt4_eval/ \
    --deepspeed examples/ds_config.json \
    --use_ram_optimized_load False \
    --metric ${metric} \
    --prompt_structure "###Human:{input}###Assistant:" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err