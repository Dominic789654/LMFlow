
# 30 part 

# gpt2 c4 continue freeze

k=30

python scripts/data_preprocess/split.py \
  --dataset_path data/c4_10G/c4_sampled_data_10G.json \
  --output_path  data/c4_10G_split \
  --seed 1 \
  --k ${k}

#!/bin/bash

# 生成长度为10的随机列表，元素范围是1到11
# random_list=($(shuf -i 1-11 -n 20))
random_list=($(echo {0,2,3,4,5,6,7,8,9,10} | tr ' ' '\n' | shuf -n 30 -r))

# 打印生成的随机列表
echo "Generated list: ${random_list[*]}"


run_name="ft_gpt2_adamw_C4_3e-3_continue_freeze_experiment_activate1-11_total_30"
lr=3e-3
bs=200
per_device_eval_batch_size=1
use_lora=0
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=8
block_size=512
ds_config=configs/ds_config_zero2_custom_optimizer.json
# model_name_or_path=configs/llama_350m.json
# model_name_or_path=pinkmanlove/llama-7b-hf
# model_name_or_path=meta-llama/Llama-2-7b-hf
# model_name_or_path=microsoft/phi-1_5
model_name_or_path=gpt2
exp_name="${run_name}_0";
# data_path="data/gpt4_v2"
data_path="data/c4_10G_split/split_0"
warmup_ratio=0.01
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
random_layer=${random_list[$((selected_portion-1))]}
echo $random_layer
activate_layers="--activate_layers  1 $random_layer 11"
echo $activate_layers
freeze_percentage=90
freeze_strategy="random"
local_seed=$RANDOM
bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 


for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    selected_portion=$((i+1))
    # echo ${selected_portion}
    random_layer=${random_list[$((i))]}
    echo $random_layer
    activate_layers="--activate_layers  1 $random_layer 11"
    echo $activate_layers
    exp_name="${run_name}_${i}"
    data_path="data/c4_10G_split/split_${i}/"
    bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
done