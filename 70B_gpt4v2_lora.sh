k=1
run_name="ft_llama2-70b_adamw_gpt4v2_5e-5_lora_rank128"
lr=5e-5
bs=8
per_device_eval_batch_size=1
use_lora=1
epochs=1
gradient_checkpointing=False
gradient_accumulation_steps=1
lora_r=128
block_size=512
ds_config=configs/ds_config_zero3.json
model_name_or_path=/nfshome/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939
exp_name="${run_name}_0";
data_path="data/gpt4_v2"
warmup_ratio=0.01
echo ${data_path}
eval_dataset_path="data/continue_half_news_wiki_formated_eval/"
test_dataset_path="data/continue_half_news_wiki_formated_test"
num_portions=${k}
selected_portion=1
optimizer_name=Adamw
bash ./scripts/run_finetune_relora.sh ${exp_name} ${data_path} ${lr} ${bs} ${model_name_or_path} ${use_lora} ${ds_config} ${epochs} ${gradient_checkpointing} ${gradient_accumulation_steps} ${lora_r} ${eval_dataset_path} ${block_size} ${per_device_eval_batch_size} ${warmup_ratio} ${num_portions} ${selected_portion} ${optimizer_name} "--master_port=10002 --include localhost:0,1,2,3,4,5,6,7"  


