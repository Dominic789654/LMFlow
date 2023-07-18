k=5


# python scripts/data_preprocess/split.py \
#   --dataset_path data/high_SFT_data_0501/sharegpt_en5w_cn1w+belle_cn_8w+gpt4_5w+water_degpt4eval.json \
#   --output_path  data/high_SFT_data_0501_split \
#   --k ${k}


lr=1e-4
bs=8
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=1
lora_r=128
block_size=2048
ds_config=configs/ds_config_zero2.json
model_name_or_path=pinkmanlove/llama-7b-hf
exp_name="continue_lora_A40*8_rank128_split_0";
data_path="data/high_SFT_data_0501_split/split_0/"
echo ${data_path}
eval_dataset_path="data/eval_760/"
test_dataset_path="data/eval_760"
bash ./scripts/run_finetune_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10001 --include localhost:0,1,2,3,4,5,6,7" 

bash ./scripts/merge_lora.sh ${model_name_or_path} ./output_models/${exp_name} 
echo "finish merge"
bash ./scripts/run_evaluation.sh ./output_models/${exp_name}_merged  ${test_dataset_path} "--master_port=10005 --include localhost:0"

# bash ./scripts/run_evaluation.sh  ${model_name_or_path}  ${test_dataset_path} "--master_port=10005 --include localhost:0,1,2,3,4,5,6,7"

for i in $(seq 1 $((k-1))) 
do
    echo "current ${i} split"
    model_name_or_path=./output_models/${exp_name}_merged
    exp_name="continue_lora_A40*6_rank128_split_${i}"
    data_path="data/high_SFT_data_0501_split/split_${i}/"
    # bash ./scripts/run_finetune_xl.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10005 --include localhost:0,1,2,3,4,5,6,7" 
    # bash ./scripts/merge_lora.sh ${model_name_or_path} ./output_models/${exp_name} 
    # echo "finish merge"
    # bash ./scripts/run_evaluation.sh ./output_models/${exp_name}_merged ${test_dataset_path} "--master_port=10005 --include localhost:4"
done