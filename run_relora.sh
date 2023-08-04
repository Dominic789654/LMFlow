k=5


python scripts/data_preprocess/split.py \
  --dataset_path data/continue_half_news_wiki_formated_train/train_half_news_wiki_formated.json \
  --output_path  data/continue_half_news_wiki_formated_train_split \
  --k ${k}

run_name="relora_A100-40*8_relora_llama_350m"
lr=1e-3
bs=40
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=1024
block_size=512
ds_config=configs/ds_config_zero2.json
model_name_or_path=configs/llama_350m.json
exp_name="${run_name}_0";
data_path="data/continue_half_news_wiki_formated_train_split/split_0/"
warmup_ratio=0.03
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"


bash ./scripts/run_evaluation.sh ./output_models/${exp_name}  ${test_dataset_path} "--master_port=10005 --include localhost:0"


for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}
    selected_portion=$((i+1))
    echo ${selected_portion}
    exp_name="${run_name}_${i}"
    data_path="data/continue_half_news_wiki_formated_train_split/split_${i}/"
    bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"

    bash ./scripts/run_evaluation.sh ./output_models/${exp_name} ${test_dataset_path} "--master_port=10005 --include localhost:0"
done

# bash ./scripts/run_all_benchmark.sh --model_name_or_path ./output_models/continue_lora_A100-40*8_rank128_split_0_aggregated_scheduler_flash_9e-4_4

# bash ./scripts/run_all_benchmark.sh --model_name_or_path ./output_models/continue_lora_A100-40*8_rank1024_split_0_aggregated_scheduler_flash_9e-4_4

# python /homes/rpan/xiangL/lmflow_xl/llama_eval.py --device_id 0 1 2 3 4 5 6  --gb 70 
