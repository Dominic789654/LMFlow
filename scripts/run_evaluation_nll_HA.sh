exp_id=${1}
metric=${2}
evaluation_path=${3}
model_name=${4}
lora_model_name=${5}


log_dir=output_dir/${exp_id}_${metric}

if [ -d "${log_dir}" ]; then
 echo "${log_dir} exists"
else
 mkdir -p ${log_dir}
fi

if [ $# -ge 5 ]; then
  lora_args="--lora_model_path ${lora_model_name}"
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text2text \
    --model_name_or_path ${model_name} \
    ${lora_args} \
    --dataset_path ${evaluation_path} \
    --deepspeed examples/ds_config.json \
    --use_ram_optimized_load False \
    --metric ${metric} \
    --prompt_structure "###Human: {input}###Assistant:" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err