model_name=${2}
lora_model_name=${3}
log_dir=output_dir/${exp_id}_ppl

if [ -d "${log_dir}" ]; then
 echo "${log_dir} exists"
else
 mkdir -p ${log_dir}
fi

lora_args=""
if [ $# -ge 2 ]; then
  model=$1
fi
if [ $# -ge 3 ]; then
  lora_args="--lora_model_path $3"
fi

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluate.py \
    --answer_type text2text \
    --model_name_or_path ${model_name} \
    ${lora_args} \
    --dataset_path /home/xiangliu/LMFlow/data/gpt4_eval/ \
    --deepspeed examples/ds_config.json \
    --metric ppl \
    --prompt_structure "###Human: {input}###Assistant:" \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err