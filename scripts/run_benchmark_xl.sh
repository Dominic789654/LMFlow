exp_id=${1}
dataset_name=${2}
model_name=${3}
log_dir=output_dir/${exp_id}_${dataset_name}_nll

if [ -d "${log_dir}" ]; then
 echo "${log_dir} exists"
else
 mkdir -p ${log_dir}
fi

lora_args=""
if [ $# -ge 3 ]; then
  model=$1
fi

if [ $# -ge 4 ]; then
  deepspeed_args="$4"
fi

deepspeed $deepspeed_args examples/benchmarking.py \
  --answer_type text2text \
  --use_ram_optimized_load False \
  --model_name_or_path ${model_name} \
  --dataset_name ${dataset_name}\
  --deepspeed examples/ds_config.json \
  --metric nll \
  --prompt_structure "###Human: {input}###Assistant:" \
  | tee ${log_dir}/train.log \
  2> ${log_dir}/train.err