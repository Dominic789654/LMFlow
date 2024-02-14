


# k=1
# run_name="ft_llama2-70b_adamw_GSM8K_5e-6_epoch3"
# lr=5e-6
# bs=8
# per_device_eval_batch_size=1
# use_lora=0
# epochs=3
# gradient_checkpointing=True
# gradient_accumulation_steps=1
# lora_r=128
# block_size=512
# # ds_config=configs/ds_config_zero2_custom_optimizer.json
# ds_config=configs/ds_config_zero3.json
# model_name_or_path=/nfshome/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939
# # model_name_or_path=/hpc2hdd/home/xliu886/workspace/models/llama-2-7b
# # model_name_or_path=microsoft/phi-1_5
# exp_name="${run_name}_0";
# # data_path="data/c4_10G_split/split_0/"
# # data_path="data/c4_10G"
# data_path="data/GSM8K/train_text_only/"
# warmup_ratio=0.01
# echo ${data_path}
# eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
# test_dataset_path="data/continue_half_news_wiki_formated_test"
# num_portions=${k}
# selected_portion=1
# optimizer_name=Adamw
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  




# gsm8k epoch=3 k=100 lisa
k=100

python scripts/data_preprocess/split.py \
  --dataset_path data/GSM8K/train_text_only/train_text_only.json \
  --output_path  data/GSM8K_split \
  --seed 1 \
  --k ${k}


generate_unique_random_layers() {
  local total_layers=80  # Total number of layers in the model
  local layer_numbers=($(seq 0 $((total_layers - 1))))
  local unique_pairs=()

  # Shuffle the layer numbers
  layer_numbers=($(shuf -e "${layer_numbers[@]}"))

  # Split the shuffled numbers into chunks of 4
  for ((i=0; i<${#layer_numbers[@]}; i+=4)); do
    unique_pairs+=("${layer_numbers[i]},${layer_numbers[i+1]},${layer_numbers[i+2]},${layer_numbers[i+3]}")
  done

  echo "${unique_pairs[@]}"
}


# Store generated sets of layers in an array
IFS=' ' read -r -a random_layers_sets <<< "$(generate_unique_random_layers)"

run_name="ft_Llama2-70B_GSM8K_lr5e-5_lisa_random4_openEmdAndHead_k100_3epoch"
lr=5e-5
bs=8
per_device_eval_batch_size=1
use_lora=0
epochs=3
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=128
block_size=512
# ds_config=configs/ds_config_zero2_custom_optimizer.json
ds_config=configs/ds_config_zero3.json
model_name_or_path=/nfshome/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939
# model_name_or_path=codellama/CodeLlama-70b-hf
# model_name_or_path=/hpc2hdd/home/xliu886/workspace/share_models/microsoft/phi-2
# model_name_or_path=/hpc2hdd/home/xliu886/workspace/models/llama-2-7b
# model_name_or_path=microsoft/phi-1_5
exp_name="${run_name}_0";
data_path="data/GSM8K_split/split_0/"
# data_path="data/c4_10G"
# data_path="data/gpt4_v2"
warmup_ratio=0.0
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
# IFS=',' read -r -a random_layers <<< "${random_layers_sets[0]}"
echo "Random layers to activate: $(eval echo ${random_layers_sets[0]} | sed 's/,/ /g')"
activate_layers="--activate_layers $(eval echo ${random_layers_sets[0]} | sed 's/,/ /g')"
echo "$activate_layers"
freeze_percentage=80
freeze_strategy="random"
local_seed=$RANDOM
bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
# exp_name="${run_name}_31";

for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    # model_name_or_path=./output_models/${run_name}_18
    selected_portion=$((i+1))
    # echo ${selected_portion}
    echo "Random layers to activate: $(eval echo ${random_layers_sets[i%20]} | sed 's/,/ /g')"
    activate_layers="--activate_layers $(eval echo ${random_layers_sets[i%20]} | sed 's/,/ /g')"
    echo "$activate_layers"
    exp_name="${run_name}_${i}"
    local_seed=$RANDOM
    data_path="data/GSM8K_split/split_${i}/"
    bash ./scripts/run_finetune_relora_freeze.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "${activate_layers}" ${freeze_strategy} ${freeze_percentage} ${local_seed} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7" 
    # 检查上一个命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Script failed for split ${i}. Exiting loop."
        break # 或者使用 'continue' 来跳过此次循环
    fi
done


