
lr=1e-4
bs=8
per_device_eval_batch_size=8
use_lora=0
epochs=1
gradient_checkpointing=True
gradient_accumulation_steps=8
lora_r=128
block_size=512
ds_config=configs/ds_config_zero3.json
model_name_or_path=bigscience/bloom-7b1
exp_name="xl_090";
data_path="/home/xiangliu/LMFlow/data/high_SFT_data_0501"
eval_dataset_path="/home/xiangliu/LMFlow/data/eval_760"
bash ./scripts/run_finetune_xl_lora_modules.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} "--master_port=10065 --num_gpus=8" 


# bash ./scripts/run_chatbot_vicuna_test_HA.sh ./output_models/${exp_name} > ./chatbot_logs/${exp_name}.log 2>&1 

# eval_dataset_name=lmflow_chat_nll_eval
# bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 

eval_dataset_name=commonsense_qa_eval
bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 

eval_dataset_name=all_nll_eval
bash ./scripts/run_benchmark_xl.sh ${exp_name} ${eval_dataset_name} ./output_models/${exp_name} "--master_port=10065 --num_gpus=1" 
