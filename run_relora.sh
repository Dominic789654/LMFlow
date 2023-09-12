k=3


python scripts/data_preprocess/split.py \
  --dataset_path data/gpt4_v2/gpt4_v2_text_only.json \
  --output_path  data/gpt4_v2_split \
  --seed 1 \
  --k ${k}

run_name="A100-40*2_conlora_gpt2_1024_lionlamb_ft_gpt4_v2_5e-3"
lr=5e-3
bs=200
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=1024
block_size=512
ds_config=configs/ds_config_zero2_custom_optimizer.json
model_name_or_path=gpt2
exp_name="${run_name}_0";
data_path="data/gpt4_v2_split/split_0/"
warmup_ratio=0.03
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=LionLamb
# bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"


run_name="A100-40*2_conlora_gpt2_1024_adamw_ft_gpt4_v2_6e-4"
lr=6e-3
bs=200
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=1024
block_size=512
ds_config=configs/ds_config_zero2_custom_optimizer.json
model_name_or_path=gpt2
exp_name="${run_name}_0";
data_path="data/gpt4_v2_split/split_0/"
warmup_ratio=0.03
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"

for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    selected_portion=$((i+1))
    echo ${selected_portion}
    exp_name="${run_name}_${i}"
    data_path="data/gpt4_v2_split/split_${i}/"
    bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"
done
