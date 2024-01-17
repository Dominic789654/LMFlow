k=3

python scripts/data_preprocess/split.py \
  --dataset_path data/gpt4_v2/gpt4_v2_text_only.json \
  --output_path  data/gpt4_v2_split \
  --seed 1 \
  --k ${k}


# 生成长度为10的随机列表，元素范围是1到11
# random_list=($(shuf -i 1-11 -n 20))
generate_unique_random_pairs() {
  local list1=()
  local list2=()
  for ((i=0; i<3; i++)); do
    while : ; do
      local num1=$(( RANDOM % 31 + 1 ))
      local num2=$(( RANDOM % 31 + 1 ))
      if [ "$num1" -ne "$num2" ]; then
        list1+=($num1)
        list2+=($num2)
        break
      fi
    done
  done
  echo "${list1[*]}"
  echo "${list2[*]}"
}

# Generate the random lists and store them in variables
mapfile -t generated_lists < <(generate_unique_random_pairs)
random_list_1=(${generated_lists[0]})
random_list_2=(${generated_lists[1]})

# Print the generated lists
echo "Generated list 1: ${random_list_1[*]}"
echo "Generated list 2: ${random_list_2[*]}"



run_name="ft_mistral_adamw_gpt4_v2_5e-5_activate_random*2_no_1_k3_warm0.06_openheadAndemb"
lr=5e-5
bs=50
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
# model_name_or_path=gpt2
model_name_or_path=mistralai/Mistral-7B-v0.1
exp_name="${run_name}_0";
data_path="data/gpt4_v2_split/split_0"
# data_path="data/c4_10G"
warmup_ratio=0.06 # 0.03
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
random_layer_1=${random_list_1[$((selected_portion-1))]}
random_layer_2=${random_list_2[$((selected_portion-1))]}
echo "random_layer_1: $random_layer_1, random_layer_2: $random_layer_2"
activate_layers="--activate_layers  $random_layer_1 $random_layer_2 "
# activate_layers="--activate_layers 0  21"
echo "$activate_layers"
freeze_percentage=80
freeze_strategy="random"
local_seed=$RANDOM
bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 


for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    selected_portion=$((i+1))
    # echo ${selected_portion}
    random_layer_1=${random_list_1[$((selected_portion-1))]}
    random_layer_2=${random_list_2[$((selected_portion-1))]}
    echo "random_layer_1: $random_layer_1, random_layer_2: $random_layer_2"
    activate_layers="--activate_layers  $random_layer_1 $random_layer_2 "
    # activate_layers="--activate_layers 0  21"
    echo "$activate_layers"
    exp_name="${run_name}_${i}"
    local_seed=$RANDOM
    data_path="data/gpt4_v2_split/split_${i}/"
    bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
done
